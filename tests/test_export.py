import pytest
import torch

from export_onnx import WaveformClassifier, WaveformFrontend, export
from features import fbank
from model import TATConfig, TinyAudioTransformer


def clips():
    # Loud to near-silent: log() magnifies tiny differences in quiet frames, so cover them
    g = torch.Generator().manual_seed(0)
    return torch.randn(8, 16000, generator=g) * torch.logspace(-4, 0, 8)[:, None]


def test_frontend_matches_training_fbank():
    # The gather + matmul-DFT rewrite must compute features.fbank (unfold + rfft)
    wav = clips()
    torch.testing.assert_close(WaveformFrontend()(wav), fbank(wav, 128, 128), atol=1e-3, rtol=1e-4)


def test_onnx_matches_pytorch(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    model = TinyAudioTransformer(TATConfig(depth=1)).eval()
    session = export(WaveformClassifier(model).eval(), tmp_path / "m.onnx")
    wav = clips()
    logits, feats = session.run(None, {"waveform": wav.numpy()})
    with torch.no_grad():
        torch.testing.assert_close(torch.from_numpy(logits), model(fbank(wav, 128, 128)), atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(torch.from_numpy(feats), fbank(wav, 128, 128), atol=1e-3, rtol=1e-4)
    assert session.run(None, {"waveform": wav[:3].numpy()})[0].shape == (3, 35), "batch size must be dynamic"
