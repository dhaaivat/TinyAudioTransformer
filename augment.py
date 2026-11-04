# augment.py
# Training-time augmentations, batched on GPU.
# Waveform stage (shift, mixup, background noise) runs before fbank, because mixing sounds is additive
# in the waveform, not in log-mel. SpecAugment runs after fbank, on the spectrogram it was designed for.
import torch


def _per_clip(v, x):
    # [B] → broadcastable against x ([B, N] or [B, T, F])
    return v.view(-1, *([1] * (x.dim() - 1)))


def time_shift(wav, max_shift):
    # wav: [B, N]. Each clip moves by its own integer in [-max_shift, max_shift]; vacated samples are zeros.
    # Teaches the model that *when* the word starts within the 1 s window doesn't matter.
    B, N = wav.shape
    shift = torch.randint(-max_shift, max_shift + 1, (B, 1), device=wav.device)
    src = torch.arange(N, device=wav.device).unsqueeze(0) - shift          # out[t] = wav[t - shift]
    valid = (src >= 0) & (src < N)
    return torch.where(valid, wav.gather(1, src.clamp(0, N - 1)), torch.zeros((), device=wav.device))


def mixup(x, targets, alpha, p):
    # A fraction p of clips is blended with a random partner: x = λ·x1 + (1-λ)·x2, and so are their labels.
    # λ ~ Beta(alpha, alpha); AST's speech-commands recipe uses p=0.6 and Beta(10, 10) (λ near 0.5).
    B = x.shape[0]
    a = torch.tensor(float(alpha), device=x.device)
    lam = torch.distributions.Beta(a, a).sample((B,))
    lam = torch.where(torch.rand(B, device=x.device) < p, lam, torch.ones_like(lam))
    perm = torch.randperm(B, device=x.device)
    x = _per_clip(lam, x) * x + _per_clip(1 - lam, x) * x[perm]
    targets = lam[:, None] * targets + (1 - lam[:, None]) * targets[perm]
    return x, targets


def add_background_noise(wav, noise_bank, snr_db=(0.0, 20.0), p=0.8):
    # Adds a random 1 s segment of real background noise (_background_noise_) at a random SNR.
    # SNR (dB) = 10·log10(P_signal / P_noise): 20 dB is faint noise, 0 dB is noise as loud as the speech.
    B = wav.shape[0]
    seg = noise_bank[torch.randint(len(noise_bank), (B,), device=wav.device)]
    snr = torch.empty(B, 1, device=wav.device).uniform_(*snr_db)
    apply = (torch.rand(B, 1, device=wav.device) < p).to(wav.dtype)
    return wav + apply * (mix_at_snr(wav, seg, snr) - wav)


def mix_at_snr(wav, noise, snr_db):
    # wav, noise: [B, N]; snr_db: [B, 1] or a number. Scales each noise row so that
    # 10·log10(P_wav / P_scaled_noise) = snr_db exactly, then adds it.
    p_sig = wav.pow(2).mean(1, keepdim=True).clamp_min(1e-10)
    p_noise = noise.pow(2).mean(1, keepdim=True).clamp_min(1e-10)
    return wav + torch.sqrt(p_sig / (p_noise * 10 ** (torch.as_tensor(snr_db, device=wav.device) / 10))) * noise


def spec_augment(x, freqm, timem, value, valid_frames=None):
    # x: [B, T, F]. Blanks one band of mel bins and one band of frames per clip, widths uniform in
    # [0, freqm] / [0, timem] (same sampling as torchaudio's FrequencyMasking/TimeMasking, used by AST).
    # Forces the model to recognise a word even when part of its spectrum or duration is missing.
    # value = the train mean, so masked cells become exactly 0 after the model's normalisation.
    # valid_frames: time masks stay inside the real audio (98 of 128 frames for 1 s); AST also masks the
    # zero padding, which wastes ~1/4 of its time masks.
    B, T, F = x.shape
    for width_max, length, dim in ((freqm, F, 2), (timem, valid_frames or T, 1)):
        if width_max <= 0:
            continue
        w = torch.randint(0, width_max + 1, (B, 1), device=x.device)
        start = (torch.rand(B, 1, device=x.device) * (length - w + 1)).long()
        pos = torch.arange(x.shape[dim], device=x.device).unsqueeze(0)
        band = (pos >= start) & (pos < start + w)                         # [B, T or F]
        x = x.masked_fill(band[:, None, :] if dim == 2 else band[:, :, None], value)
    return x


def smooth_labels(targets, eps):
    # Label smoothing: the true class gets 1-eps, the rest share eps. Discourages over-confident logits.
    return targets * (1 - eps) + eps / targets.shape[1]


def soft_cross_entropy(logits, targets):
    # Cross-entropy against a probability vector (needed once labels are mixed or smoothed)
    return -(targets * torch.log_softmax(logits, dim=1)).sum(1).mean()
