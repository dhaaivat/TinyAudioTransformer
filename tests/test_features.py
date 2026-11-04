import math

import pytest
import torch

from features import SAMPLE_RATE, fbank


def _test_signal(seconds=1.0, seed=0):
    # Chirp + noise: exercises the whole spectrum, unlike pure noise
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    chirp = 0.3 * torch.sin(2 * math.pi * (100 + 3000 * t) * t)
    return chirp + 0.05 * torch.randn(len(t), generator=g)


@pytest.mark.parametrize("seconds,real_frames", [(0.3, 28), (1.0, 98), (1.5, 148)])
def test_fixed_output_shape(seconds, real_frames):
    out = fbank(_test_signal(seconds), n_mels=128, n_frames=128)
    assert out.shape == (128, 128)
    if real_frames < 128:
        assert torch.all(out[real_frames:] == 0), "padding must be zeros, as in AST"
        assert torch.all(out[:real_frames] != 0)


def test_batch_matches_single():
    wavs = torch.stack([_test_signal(seed=s) for s in range(3)])
    batched = fbank(wavs, n_frames=128)
    for i in range(3):
        torch.testing.assert_close(batched[i], fbank(wavs[i], n_frames=128))


def test_matches_torchaudio_kaldi():
    kaldi = pytest.importorskip("torchaudio.compliance.kaldi")
    wav = _test_signal()
    ref = kaldi.fbank(wav.unsqueeze(0), sample_frequency=SAMPLE_RATE, window_type="hanning", num_mel_bins=128)
    torch.testing.assert_close(fbank(wav), ref, atol=1e-3, rtol=1e-4)


def test_matches_hf_ast_feature_extractor():
    transformers = pytest.importorskip("transformers")
    try:
        fe = transformers.ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-speech-commands-v2")
    except OSError as e:
        pytest.skip(f"cannot reach the Hub: {e}")
    wav = _test_signal()
    hf = fe(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values[0]
    hf_raw = hf * (2 * fe.std) + fe.mean  # undo AST normalisation; ours lives inside the model
    torch.testing.assert_close(fbank(wav, 128, fe.max_length), hf_raw, atol=1e-3, rtol=1e-4)
