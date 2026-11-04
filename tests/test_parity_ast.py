# Our architecture loaded with the authors' AST weights must compute the same function as their model.
import os
from pathlib import Path

import pytest
import torch

from conftest import hf_or_skip
from features import SAMPLE_RATE, fbank


@pytest.fixture(scope="module")
def models():
    return hf_or_skip()


def test_logits_match_on_random_input(models):
    ours, hf, fe = models
    g = torch.Generator().manual_seed(0)
    hf_in = torch.randn(2, fe.max_length, 128, generator=g)          # normalised, as HF expects
    raw = hf_in * (2 * fe.std) + fe.mean                             # what our model expects
    with torch.no_grad():
        torch.testing.assert_close(ours(raw), hf(hf_in).logits, atol=1e-3, rtol=1e-3)


def test_end_to_end_on_audio(models):
    # Waveform → our fbank → our model  vs  waveform → HF extractor → HF model
    ours, hf, fe = models
    root = os.environ.get("SC_ROOT")
    if root and Path(root, "testing_list.txt").exists():
        from dataset import load_wav
        rels = Path(root, "testing_list.txt").read_text().split()[::1000][:8]
        wavs = [load_wav(Path(root, r)) for r in rels]
        labels = [r.split("/")[0] for r in rels]
    else:
        g = torch.Generator().manual_seed(0)
        wavs, labels = [0.1 * torch.randn(SAMPLE_RATE, generator=g) for _ in range(2)], None

    with torch.no_grad():
        ours_logits = torch.stack([ours(fbank(w, 128, fe.max_length)[None])[0] for w in wavs])
        hf_logits = hf(**fe([w.numpy() for w in wavs], sampling_rate=SAMPLE_RATE, return_tensors="pt")).logits
    torch.testing.assert_close(ours_logits, hf_logits, atol=1e-3, rtol=1e-3)

    if labels:  # the reference itself should be right on real test clips, or we are comparing to garbage
        id2label = hf.config.id2label
        preds = [id2label[i] for i in ours_logits.argmax(1).tolist()]
        assert sum(p == l for p, l in zip(preds, labels)) >= len(labels) - 1, list(zip(preds, labels))
