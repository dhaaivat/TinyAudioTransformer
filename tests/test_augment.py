import math

import pytest
import torch

import augment
from metrics import expected_calibration_error, mcnemar_p


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_time_shift_moves_content_and_zero_fills():
    wav = torch.randn(16, 200)
    out = augment.time_shift(wav, max_shift=20)
    for w, o in zip(wav, out):
        matches = [s for s in range(-20, 21)
                   if torch.equal(o, torch.roll(w, s) * ((torch.arange(200) - s >= 0) & (torch.arange(200) - s < 200)))]
        assert matches, "each output row must be its input shifted by some s in [-20, 20], zero-filled"


def test_time_shift_zero_is_identity():
    wav = torch.randn(4, 100)
    torch.testing.assert_close(augment.time_shift(wav, 0), wav)


@pytest.mark.parametrize("snr", [0.0, 10.0, 20.0])
def test_noise_hits_requested_snr(snr):
    wav, bank = torch.randn(8, 16000) * 0.1, torch.randn(5, 16000)
    out = augment.add_background_noise(wav, bank, (snr, snr), p=1.0)
    noise = out - wav
    measured = 10 * torch.log10(wav.pow(2).mean(1) / noise.pow(2).mean(1))
    torch.testing.assert_close(measured, torch.full((8,), snr), atol=1e-3, rtol=0)


def test_noise_p0_is_identity():
    wav = torch.randn(8, 16000)
    torch.testing.assert_close(augment.add_background_noise(wav, torch.randn(3, 16000), p=0.0), wav)


def test_mixup_is_convex_and_labels_sum_to_one():
    x = torch.rand(32, 100)
    t = torch.nn.functional.one_hot(torch.randint(0, 35, (32,)), 35).float()
    xm, tm = augment.mixup(x, t, alpha=10.0, p=1.0)
    torch.testing.assert_close(tm.sum(1), torch.ones(32))
    assert (xm >= x.min() - 1e-6).all() and (xm <= x.max() + 1e-6).all()
    assert not torch.equal(xm, x)


def test_mixup_p0_is_identity():
    x, t = torch.rand(8, 50), torch.eye(8)
    xm, tm = augment.mixup(x, t, 10.0, p=0.0)
    torch.testing.assert_close(xm, x)
    torch.testing.assert_close(tm, t)


def test_spec_augment_masks_one_contiguous_band_per_axis():
    x = torch.randn(64, 128, 128)
    out = augment.spec_augment(x, freqm=24, timem=24, value=99.0)
    masked = out == 99.0
    for m in masked:
        freq_cols = m.all(0).nonzero().flatten()   # mel bins masked across all frames
        time_rows = m.all(1).nonzero().flatten()   # frames masked across all bins
        for band in (freq_cols, time_rows):
            assert len(band) <= 24
            if len(band):
                assert band.max() - band.min() + 1 == len(band), "band must be contiguous"
        # nothing outside those two bands is touched
        expected = torch.zeros_like(m)
        expected[:, freq_cols] = True
        expected[time_rows, :] = True
        assert torch.equal(m, expected)


def test_spec_augment_time_mask_stays_in_real_frames():
    x = torch.randn(256, 128, 128)
    masked_rows = (augment.spec_augment(x, 0, 24, value=99.0, valid_frames=98) == 99.0).all(2)
    assert masked_rows.any(), "some time masks expected"
    assert not masked_rows[:, 98:].any(), "padding frames must never be masked"


def test_soft_ce_equals_ce_on_one_hot():
    logits, y = torch.randn(10, 35), torch.randint(0, 35, (10,))
    soft = augment.soft_cross_entropy(logits, torch.nn.functional.one_hot(y, 35).float())
    torch.testing.assert_close(soft, torch.nn.functional.cross_entropy(logits, y))


def test_label_smoothing_matches_pytorch():
    logits, y = torch.randn(10, 35), torch.randint(0, 35, (10,))
    t = augment.smooth_labels(torch.nn.functional.one_hot(y, 35).float(), 0.1)
    torch.testing.assert_close(augment.soft_cross_entropy(logits, t),
                               torch.nn.functional.cross_entropy(logits, y, label_smoothing=0.1))


def test_ece_zero_when_calibrated_and_large_when_overconfident():
    y = torch.zeros(1000, dtype=torch.long)
    # Model says class 0 with 80% confidence and is right 80% of the time → ECE ≈ 0
    logit0 = math.log(0.8 / 0.2)
    logits = torch.tensor([[logit0, 0.0]]).repeat(1000, 1)
    y[800:] = 1
    assert expected_calibration_error(logits, y) < 0.5
    # Same 80% accuracy but 99.9% confident → ECE ≈ 19.9 points
    logits = torch.tensor([[math.log(999), 0.0]]).repeat(1000, 1)
    assert expected_calibration_error(logits, y) == pytest.approx(19.9, abs=0.2)


def test_mcnemar():
    a = torch.ones(100, dtype=torch.bool)
    assert mcnemar_p(a.numpy(), a.numpy()) == 1.0
    b = a.clone()
    b[:20] = False                       # 20 disagreements, all favouring a
    assert mcnemar_p(a.numpy(), b.numpy()) < 1e-4
