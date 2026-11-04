# export_onnx.py
# Phase 3: packages a trained checkpoint as ONE ONNX graph: raw 16 kHz waveform [B, 16000] → logits [B, 35],
# plus the log-mel features the model saw [B, 128, 128] (so the demo can show them).
# The Kaldi fbank goes *inside* the graph, so a browser can't compute features differently from training.
# features.fbank is rewritten here in export-friendly form (same maths, checked by tests/test_export.py):
#   - framing is a gather with a fixed index table instead of Tensor.unfold
#   - the FFT is a matmul with precomputed cos/sin tables (window folded in) instead of torch.fft.rfft
# The exported graph is then checked with onnxruntime against the ORIGINAL training pipeline
# (features.fbank → model), so one check covers both the rewrite and the export.
#   python export_onnx.py results/phase2b/runs_phase2b/tat_shift_specaug_noise_s0/best.pt web/model \
#       --real-audio mini_speech_commands
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import CLIP_SAMPLES, load_wav
from eval_noisy import load_model
from features import EPS, FRAME_LENGTH, FRAME_SHIFT, N_FFT, PREEMPHASIS, fbank, mel_banks, num_frames


class WaveformFrontend(nn.Module):
    """features.fbank(wav, n_mels, n_frames) for fixed-length clips, built only from gather, matmul and
    elementwise ops."""

    def __init__(self, n_mels=128, n_frames=128, n_samples=CLIP_SAMPLES):
        super().__init__()
        T = num_frames(n_samples)
        self.pad_frames = n_frames - T
        self.register_buffer("frame_idx", torch.arange(T)[:, None] * FRAME_SHIFT + torch.arange(FRAME_LENGTH))
        # Real DFT of a zero-padded 512-point frame = only the first 400 rows of the 512-point tables matter.
        # The Hann window is a per-sample scale, so it can be folded into the tables.
        n, k = torch.arange(FRAME_LENGTH, dtype=torch.float64)[:, None], torch.arange(N_FFT // 2 + 1)
        angle = 2 * math.pi * n * k / N_FFT
        window = torch.hann_window(FRAME_LENGTH, periodic=False, dtype=torch.float64)[:, None]
        self.register_buffer("dft_cos", (window * torch.cos(angle)).float())    # [400, 257]
        self.register_buffer("dft_sin", (window * -torch.sin(angle)).float())
        self.register_buffer("mel_t", mel_banks(n_mels).T.contiguous())          # [257, n_mels]

    def forward(self, wav):
        frames = wav[:, self.frame_idx]                                           # [B, T, 400]
        frames = frames - frames.mean(dim=-1, keepdim=True)
        prev = torch.cat((frames[..., :1], frames[..., :-1]), dim=-1)
        frames = frames - PREEMPHASIS * prev
        power = (frames @ self.dft_cos).pow(2) + (frames @ self.dft_sin).pow(2)   # [B, T, 257]
        out = torch.clamp(power @ self.mel_t, min=EPS).log()
        return nn.functional.pad(out, (0, 0, 0, self.pad_frames))               # zero-pad time to n_frames


class WaveformClassifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.frontend, self.model = WaveformFrontend(), model

    def forward(self, wav):
        x = self.frontend(wav)
        return self.model(x), x


def real_clips(root, limit):
    files = sorted(Path(root).glob("*/*.wav"))[:limit] if limit else sorted(Path(root).glob("*/*.wav"))
    wavs = []
    for f in files:
        w = load_wav(f)[:CLIP_SAMPLES]
        wavs.append(np.pad(w, (0, CLIP_SAMPLES - len(w))))
    return torch.from_numpy(np.stack(wavs)).float(), [f.parent.name for f in files]


def compare(session, reference, wav, batch_size=256):
    ours, theirs = [], []
    with torch.no_grad():
        for i in range(0, len(wav), batch_size):
            b = wav[i:i + batch_size]
            ours.append(reference(b).numpy())
            theirs.append(session.run(None, {"waveform": b.numpy()})[0])
    ours, theirs = np.concatenate(ours), np.concatenate(theirs)
    return np.abs(ours - theirs).max(), (ours.argmax(1) == theirs.argmax(1)).mean() * 100, theirs


def strip_debug_info(onnx_path):
    # The exporter records a Python stack trace per node (with local file paths); drop it before publishing
    import onnx
    m = onnx.load(str(onnx_path))
    for node in m.graph.node:
        del node.metadata_props[:]
        node.doc_string = ""
    del m.metadata_props[:]
    m.graph.doc_string = ""
    onnx.save(m, str(onnx_path))


def export(wrapped, onnx_path):
    import onnxruntime as ort
    torch.onnx.export(wrapped, (torch.randn(2, CLIP_SAMPLES) * 0.1,), str(onnx_path), dynamo=True,
                      input_names=["waveform"], output_names=["logits", "fbank"],
                      dynamic_shapes={"wav": {0: torch.export.Dim("batch")}}, external_data=False)
    strip_debug_info(onnx_path)
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("out_dir")
    p.add_argument("--real-audio", default=None, help="folder of <word>/<clip>.wav to check ONNX vs PyTorch on")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tolerance", type=float, default=1e-3)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = load_model(args.checkpoint, "cpu")
    wrapped = WaveformClassifier(model).eval()
    reference = lambda wav: model(fbank(wav, 128, 128))  # exactly what train.py evaluated

    onnx_path = out / "tat.onnx"
    session = export(wrapped, onnx_path)
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")

    # 1. random audio: includes near-silent and loud clips, the cases where log() amplifies small differences
    g = torch.Generator().manual_seed(0)
    rand = torch.randn(16, CLIP_SAMPLES, generator=g) * torch.logspace(-4, 0, 16)[:, None]
    diff, agree, _ = compare(session, reference, rand)
    report = {"random": {"max_abs_logit_diff": float(diff), "argmax_agreement": agree}}
    print(f"random audio: max |Δlogit| {diff:.2e}, argmax agreement {agree:.1f}%")
    assert diff < args.tolerance, "ONNX output differs from the PyTorch training pipeline"
    feat_diff = np.abs(session.run(None, {"waveform": rand.numpy()})[1] - fbank(rand, 128, 128).numpy()).max()
    report["random"]["max_abs_fbank_diff"] = float(feat_diff)
    print(f"random audio: max |Δfbank| {feat_diff:.2e}")
    assert feat_diff < args.tolerance

    # 2. real speech, also checking the original (unfold + rfft) pipeline against the exported one
    if args.real_audio:
        wav, words = real_clips(args.real_audio, args.limit)
        diff, agree, logits = compare(session, reference, wav)
        classes = ck["classes"]
        acc = np.mean([classes[i] == w for i, w in zip(logits.argmax(1), words)]) * 100
        report["real"] = {"clips": len(wav), "max_abs_logit_diff": float(diff), "argmax_agreement": agree,
                          "accuracy_not_a_benchmark": acc}
        print(f"real audio ({len(wav)} clips): max |Δlogit| {diff:.2e}, argmax agreement {agree:.2f}%, "
              f"accuracy {acc:.2f}% (sanity only: these speakers may overlap with training)")
        assert diff < args.tolerance and agree == 100.0

    meta = {"classes": ck["classes"], "sample_rate": 16000, "clip_samples": CLIP_SAMPLES,
            "input": "waveform [batch, 16000] float32 in [-1, 1]", "outputs": {"logits": "[batch, 35]", "fbank": "[batch, 128 frames, 128 mel bins], log-mel before normalisation"},
            "checkpoint": str(args.checkpoint), "epoch": ck["epoch"], "val_acc": ck["val_acc"],
            "parity": report}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out / 'meta.json'}")


if __name__ == "__main__":
    main()
