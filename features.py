# features.py
# Kaldi-compatible log-mel filterbank in plain torch, matching
# torchaudio.compliance.kaldi.fbank(wav, sample_frequency=16000, window_type="hanning", num_mel_bins=128)
# which is what AST (and HF's ASTFeatureExtractor) use. Owning it removes the torchaudio dependency,
# works on batches/GPU, and gives one reference to port when the feature pipeline moves to the browser.
# tests/test_features.py checks it against torchaudio and the HF extractor.
import math

import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
FRAME_LENGTH = 400      # 25 ms
FRAME_SHIFT = 160       # 10 ms
N_FFT = 512             # frame length rounded up to a power of two
PREEMPHASIS = 0.97
LOW_FREQ = 20.0
EPS = torch.finfo(torch.float32).eps


def _mel(f):
    return 1127.0 * torch.log1p(f / 700.0)


def mel_banks(n_mels=128, sample_rate=SAMPLE_RATE, n_fft=N_FFT, low_freq=LOW_FREQ):
    # Triangles evenly spaced on the mel scale, evaluated at FFT bin centres; Nyquist bin gets weight 0
    mel_lo = 1127.0 * math.log1p(low_freq / 700.0)
    mel_hi = 1127.0 * math.log1p((sample_rate / 2) / 700.0)
    delta = (mel_hi - mel_lo) / (n_mels + 1)

    b = torch.arange(n_mels).unsqueeze(1)
    left, center, right = (mel_lo + (b + k) * delta for k in (0.0, 1.0, 2.0))
    mel = _mel(sample_rate / n_fft * torch.arange(n_fft // 2)).unsqueeze(0)

    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    banks = torch.clamp(torch.min(up, down), min=0.0)        # [n_mels, n_fft/2]
    return F.pad(banks, (0, 1))                               # [n_mels, n_fft/2 + 1]


def fbank(wav, n_mels=128, n_frames=None):
    """wav: float waveform in [-1, 1], shape [N] or [B, N] at 16 kHz.
    Returns log-mel [T, n_mels] (or [B, T, n_mels]); if n_frames is given, zero-pads/crops time to it."""
    squeeze = wav.dim() == 1
    wav = wav.reshape(-1, wav.shape[-1]).float()

    frames = wav.unfold(-1, FRAME_LENGTH, FRAME_SHIFT)                         # [B, T, 400], snip_edges=True
    frames = frames - frames.mean(dim=-1, keepdim=True)                         # remove DC per frame
    prev = torch.cat((frames[..., :1], frames[..., :-1]), dim=-1)
    frames = frames - PREEMPHASIS * prev                                        # pre-emphasis
    frames = frames * torch.hann_window(FRAME_LENGTH, periodic=False, device=wav.device)
    frames = F.pad(frames, (0, N_FFT - FRAME_LENGTH))

    power = torch.fft.rfft(frames).abs().pow(2)                                 # [B, T, 257]
    mel = power @ mel_banks(n_mels).to(wav.device).T                            # [B, T, n_mels]
    out = torch.clamp(mel, min=EPS).log()

    if n_frames is not None:
        out = pad_or_crop(out, n_frames)
    return out[0] if squeeze else out


def num_frames(n_samples):
    # Frames kaldi produces with snip_edges=True: 16000 samples → 98
    return 1 + (n_samples - FRAME_LENGTH) // FRAME_SHIFT


def pad_or_crop(x, n_frames):
    # x: [..., T, F]; zero-pad at the end of time like AST's ZeroPad2d, or crop
    T = x.shape[-2]
    if T < n_frames:
        return F.pad(x, (0, 0, 0, n_frames - T))
    return x[..., :n_frames, :]
