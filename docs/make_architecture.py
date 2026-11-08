# docs/make_architecture.py
# Draws the model's architecture diagram (light + dark), waveform → logits with every tensor shape.
# Shapes and parameter counts are measured by running the real modules, not typed in by hand.
#   python docs/make_architecture.py   →   docs/assets/architecture_{light,dark}.png
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export_onnx import WaveformFrontend  # noqa: E402
from model import TATConfig, TinyAudioTransformer  # noqa: E402

OUT = ROOT / "docs" / "assets"
CKPT = ROOT / "results" / "phase2b" / "runs_phase2b" / "tat_shift_specaug_noise_s0" / "best.pt"

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "line": "#9a9893",
              "stage": ["#2a78d6", "#eb6834", "#1baf7a"], "fill_alpha": 0.10},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "line": "#77756f",
             "stage": ["#3987e5", "#d95926", "#199e70"], "fill_alpha": 0.16},
}


def measure():
    """Runs one dummy clip through the real modules and records shapes and parameter counts."""
    cfg = TATConfig()
    model = TinyAudioTransformer(cfg).eval()
    mean, std = -6.79, 5.56
    if CKPT.exists():
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
        mean, std = sd["norm_in.mean"].item(), sd["norm_in.std"].item()
    fe = WaveformFrontend()
    wav = torch.randn(1, 16000) * 0.1
    s = {}
    with torch.no_grad():
        frames = wav[:, fe.frame_idx]
        s["frames"] = tuple(frames.shape)
        power = (frames @ fe.dft_cos).pow(2) + (frames @ fe.dft_sin).pow(2)
        s["power"] = tuple(power.shape)
        s["mel"] = tuple((power @ fe.mel_t).shape)
        x = fe(wav)
        s["fbank"] = tuple(x.shape)
        x = model.norm_in(x)
        conv = model.patch_embed.proj(x.unsqueeze(1).transpose(2, 3))
        s["conv_in"] = tuple(x.unsqueeze(1).transpose(2, 3).shape)
        s["conv"] = tuple(conv.shape)
        tokens = model.patch_embed(x)
        s["tokens"] = tuple(tokens.shape)
        s["seq"] = (1, tokens.shape[1] + cfg.num_prefix_tokens, cfg.embed_dim)
        s["logits"] = tuple(model(fe(wav)).shape)
    n = lambda m: sum(p.numel() for p in m.parameters())
    p = {"patch": n(model.patch_embed), "cls": model.prefix_tokens.numel(), "pos": model.pos_embed.numel(),
         "block": n(model.blocks[0]), "blocks": n(model.blocks), "norm": n(model.norm), "head": n(model.head),
         "total": n(model), "attn": n(model.blocks[0].attn), "mlp": n(model.blocks[0].mlp)}
    return cfg, s, p, mean, std


def shape(t):
    return "[" + ", ".join(["B"] + [str(d) for d in t[1:]]) + "]"


def draw(theme, path):
    cfg, s, p, mean, std = measure()
    t = THEMES[theme]
    fig = plt.figure(figsize=(18, 11.5), dpi=170)
    fig.patch.set_facecolor(t["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 115)
    ax.axis("off")

    def box(x, y, w, h, title, body, colour, shape_txt=None, params=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=colour, alpha=t["fill_alpha"], edgecolor="none"))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor="none", edgecolor=colour, linewidth=1.6))
        ax.text(x + 1.6, y + h - 1.5, title, color=t["text"], fontsize=11.5, fontweight="bold", va="top")
        ax.text(x + 1.6, y + h - 4.6, body, color=t["muted"], fontsize=9.2, va="top", linespacing=1.35)
        if shape_txt:
            ax.text(x + w - 1.6, y + 1.3, shape_txt, color=t["text"], fontsize=10, ha="right", va="bottom",
                    family="monospace", fontweight="bold")
        if params:
            ax.text(x + 1.6, y + 1.3, params, color=t["muted"], fontsize=8.6, va="bottom", style="italic")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=t["line"], lw=1.6,
                                                            shrinkA=0, shrinkB=0, mutation_scale=14))

    def elbow(x0, y0, x1, y1):
        # right from (x0, y0) to the gutter, up/down to y1, then right into (x1, y1)
        gx = (x0 + x1) / 2
        ax.plot([x0, gx, gx], [y0, y0, y1], color=t["line"], lw=1.6, solid_capstyle="butt")
        arrow(gx, y1, x1, y1)

    def column(x, title, colour, sub):
        ax.text(x, 103.5, title, color=colour, fontsize=13.5, fontweight="bold", va="bottom")
        ax.text(x, 102.2, sub, color=t["muted"], fontsize=9.5, va="top")

    ax.text(3, 112.3, "TinyAudioTransformer: architecture", color=t["text"], fontsize=19, fontweight="bold", va="center")
    ax.text(3, 109.2, f"{p['total']:,} parameters · input: 1 s of 16 kHz audio · output: scores for 35 words · "
            "B = batch size · shapes measured from the code", color=t["muted"], fontsize=10.5, va="center")

    W, H, GAP = 54, 12.2, 3.6
    xs = [3, 63, 123]
    c1, c2, c3 = t["stage"]

    # ── Column 1: front-end ─────────────────────────────────────────────
    column(xs[0], "① Front-end: sound → log-mel", c1, "Kaldi filterbank, inside the ONNX graph")
    col1 = [
        ("Waveform", "1 second of microphone audio,\n16,000 samples/s, values in [−1, 1]", shape((1, 16000))),
        ("Framing", "25 ms windows (400 samples), a new one\nevery 10 ms (160 samples) → 98 frames",
         shape(s["frames"])),
        ("Clean up each frame", "remove DC offset · pre-emphasis\nx[n] − 0.97·x[n−1] · Hann window",
         shape(s["frames"])),
        ("FFT → power spectrum", "zero-pad to 512 points, DFT, |X|²\n257 frequency bins, 0–8 kHz",
         shape(s["power"])),
        ("Mel filterbank", "128 triangular filters, dense at low pitch,\nsparse at high (20 Hz – 8 kHz)",
         shape(s["mel"])),
        ("Log + pad", "log(max(energy, ε)) compresses loudness;\nzero-pad 98 → 128 frames (time × mel)",
         shape(s["fbank"])),
    ]
    tops = []
    for i, (title, body, sh) in enumerate(col1):
        y = 99 - (i + 1) * H - i * GAP
        box(xs[0], y, W, H, title, body, c1, sh)
        tops.append(y)
        if i:
            arrow(xs[0] + W / 2, tops[i - 1], xs[0] + W / 2, y + H)

    # ── Column 2: tokenisation ──────────────────────────────────────────
    column(xs[1], "② Tokens: spectrogram → sequence", c2, "AST-style patch embedding")
    col2 = [
        ("Normalise", f"(x − μ) / (2σ), training-set stats\nμ = {mean:.2f}, σ = {std:.2f} (stored in the model)",
         shape(s["fbank"]), None),
        ("Treat as a 1-channel image", "transpose to (mel × time) and add a\nchannel axis, like a grey-scale picture",
         shape(s["conv_in"]), None),
        ("Patch embedding", f"Conv2d(1 → {cfg.embed_dim}, kernel {cfg.patch_size}×{cfg.patch_size}, "
         f"stride {cfg.patch_stride})\noverlapping patches: 12 × 12 grid, each → 128-d",
         shape(s["conv"]), f"{p['patch']:,} params"),
        ("Flatten to a sequence", "12 × 12 grid → 144 tokens,\neach a 128-number vector",
         shape(s["tokens"]), None),
        ("Add [CLS] + positions", "prepend a learned [CLS] summary token;\nadd a learned position vector to every token",
         shape(s["seq"]), f"{p['cls'] + p['pos']:,} params"),
        ("Dropout", "randomly zero 10% of values\n(training only)", shape(s["seq"]), None),
    ]
    prev = None
    for i, (title, body, sh, prm) in enumerate(col2):
        y = 99 - (i + 1) * H - i * GAP
        box(xs[1], y, W, H, title, body, c2, sh, prm)
        if prev is not None:
            arrow(xs[1] + W / 2, prev, xs[1] + W / 2, y + H)
        prev = y
    # column 1 → column 2
    elbow(xs[0] + W, tops[-1] + H / 2, xs[1], 99 - H / 2)

    # ── Column 3: encoder block (expanded) + head ──────────────────────
    column(xs[2], f"③ Transformer encoder × {cfg.depth}  →  classifier", c3, "pre-norm ViT blocks")
    bx, by, bw, bh = xs[2], 51.5, W, 47.5
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor="none", edgecolor=c3, linewidth=1.6, linestyle=(0, (5, 3))))
    ax.text(bx + 1.6, by + bh - 1.4, f"Encoder block  (repeated {cfg.depth}×, {p['block']:,} params each)",
            color=t["text"], fontsize=11.5, fontweight="bold", va="top")
    ax.text(bx + bw - 1.6, by + 1.2, shape(s["seq"]) + " in and out", color=t["text"], fontsize=10,
            family="monospace", fontweight="bold", ha="right", va="bottom")
    inner = [
        ("LayerNorm", "normalise each token", None),
        ("Multi-head self-attention", f"{cfg.num_heads} heads × {cfg.embed_dim // cfg.num_heads} dims: every token\n"
         "looks at all 145 tokens and mixes in what's relevant", f"{p['attn']:,}"),
        ("+ residual", "add the block's input back (+ dropout)", None),
        ("LayerNorm", "normalise each token", None),
        ("MLP", f"Linear {cfg.embed_dim}→{int(cfg.embed_dim * cfg.mlp_ratio)} · GELU · "
         f"Linear {int(cfg.embed_dim * cfg.mlp_ratio)}→{cfg.embed_dim}\nprocesses each token on its own",
         f"{p['mlp']:,}"),
        ("+ residual", "add the input back (+ dropout)", None),
    ]
    heights = [4.2, 7.6, 4.2, 4.2, 7.6, 4.2]
    y = by + bh - 6.0
    prev = None
    for (title, body, prm), h in zip(inner, heights):
        y -= h
        ix, iw = bx + 3, bw - 6
        ax.add_patch(FancyBboxPatch((ix, y), iw, h - 0.4, boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor=c3, alpha=t["fill_alpha"] * (1.6 if "residual" not in title else 0.6),
                                    edgecolor="none"))
        ax.text(ix + 1.2, y + h - 0.9, title, color=t["text"], fontsize=10, fontweight="bold", va="top")
        ax.text(ix + 17 if h < 5 else ix + 1.2, y + h - 0.9 if h < 5 else y + h - 3.4, body,
                color=t["muted"], fontsize=8.8, va="top", linespacing=1.3)
        if prm:
            ax.text(ix + iw - 1.2, y + h - 0.9, f"{prm} params", color=t["muted"], fontsize=8.4, va="top",
                    ha="right", style="italic")
        if prev is not None:
            arrow(ix + iw / 2, prev, ix + iw / 2, y + h - 0.4)
        prev = y
        y -= 1.0

    head = [
        ("Final LayerNorm → take [CLS]", "the [CLS] token has gathered information\nfrom all patches: it summarises the clip",
         shape((1, cfg.embed_dim)), f"{p['norm']:,} params"),
        ("Linear classifier", f"{cfg.embed_dim} → {cfg.num_classes}: one score (logit) per word",
         shape(s["logits"]), f"{p['head']:,} params"),
        ("Softmax → probabilities", "scores → percentages that sum to 100%;\nthe highest is the answer",
         shape(s["logits"]), None),
    ]
    hy = by - 4.2
    prev = by
    for title, body, sh, prm in head:
        hy -= 11.2
        box(xs[2], hy, W, 11.2, title, body, c3, sh, prm)
        arrow(xs[2] + W / 2, prev, xs[2] + W / 2, hy + 11.2)
        prev = hy
        hy -= 2.6
    # column 2 → column 3
    last2 = 99 - 6 * H - 5 * GAP
    elbow(xs[1] + W, last2 + H / 2, xs[2], by + bh - 6)

    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return p


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        params = draw(theme, OUT / f"architecture_{theme}.png")
    print("wrote architecture_{light,dark}.png;", params)
