import wave

import numpy as np
import torch

import augment
from analyze import parent_of
from baselines import CNNBaseline
from dataset import background_noise_bank
from eval_noisy import load_model, noisy_features
from model import TATConfig, TinyAudioTransformer


def test_noise_holdout_never_overlaps_training_noise(tmp_path):
    # First 80% of the recording is +1000, last 20% is −1000: each bank must see only its own part
    (tmp_path / "_background_noise_").mkdir()
    pcm = np.full(160000, 1000, dtype=np.int16)
    pcm[128000:] = -1000
    with wave.open(str(tmp_path / "_background_noise_" / "hum.wav"), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    train, test = background_noise_bank(tmp_path, "train"), background_noise_bank(tmp_path, "test")
    assert len(train) and len(test)
    assert (train == 1000).all() and (test == -1000).all()


def test_mix_at_snr_is_exact():
    wav, noise = torch.randn(4, 16000), torch.randn(4, 16000) * 3
    out = augment.mix_at_snr(wav, noise, 5.0)
    measured = 10 * torch.log10(wav.pow(2).mean(1) / (out - wav).pow(2).mean(1))
    torch.testing.assert_close(measured, torch.full((4,), 5.0), atol=1e-3, rtol=0)


def test_noisy_features_are_repeatable_and_get_worse_with_lower_snr():
    pcm = (torch.randn(6, 16000) * 3000).to(torch.int16)
    bank = torch.randn(10, 16000)
    a, b = noisy_features(pcm, bank, 10, seed=1), noisy_features(pcm, bank, 10, seed=1)
    assert torch.equal(a, b), "every model must be scored on identical noisy clips"
    clean = noisy_features(pcm, bank, 60, seed=1).float()
    err10 = (a.float() - clean).abs().mean()
    err0 = (noisy_features(pcm, bank, 0, seed=1).float() - clean).abs().mean()
    assert err0 > err10


def test_load_model_round_trip(tmp_path):
    x = torch.randn(2, 128, 128)
    for m in (TinyAudioTransformer(TATConfig(depth=1)), CNNBaseline()):
        m.eval()
        kind = "cnn" if isinstance(m, CNNBaseline) else "tat"
        torch.save({"model": kind, "config": m.config_dict(), "state_dict": m.state_dict()}, tmp_path / "best.pt")
        torch.testing.assert_close(load_model(tmp_path / "best.pt", "cpu")(x), m(x))


def test_parent_is_longest_extended_config():
    names = ["tat_none", "tat_shift", "tat_shift_specaug", "tat_shift_specaug_mixup"]
    assert parent_of("tat_shift_specaug_mixup", names) == "tat_shift_specaug"
    assert parent_of("tat_shift_specaug", names) == "tat_shift"
    assert parent_of("tat_noise", names) == "tat_none"
