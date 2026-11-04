# docs/make_figures.py
# Draws the README figures (light + dark versions) from the saved results and one example clip.
#   python docs/make_figures.py   →   docs/assets/*.png
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import load_wav  # noqa: E402
from features import fbank  # noqa: E402

OUT = ROOT / "docs" / "assets"
RUNS = ROOT / "results" / "phase2b" / "runs_phase2b"
CLIP = ROOT / "mini_speech_commands" / "yes" / "004ae714_nohash_0.wav"

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#33322f",
             "series": ["#3987e5", "#d95926", "#199e70"]},
}
CONFIGS = [("tat_none", "No augmentation"),
           ("tat_shift_specaug", "Shift + SpecAugment"),
           ("tat_shift_specaug_noise", "Shift + SpecAugment + noise (deployed)")]
LEVELS = ["clean", "20", "10", "5", "0"]


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=10)


def noise_chart(t, path):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    x = np.arange(len(LEVELS))
    for (name, label), colour in zip(CONFIGS, t["series"]):
        accs = np.array([[json.loads((RUNS / f"{name}_s{s}" / "noisy.json").read_text())["test"][l] for l in LEVELS]
                         for s in range(3)])
        mean, std = accs.mean(0), accs.std(0, ddof=1)
        ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.18, linewidth=0)
        ax.plot(x, mean, color=colour, linewidth=2, marker="o", markersize=6,
                markeredgecolor=t["surface"], markeredgewidth=2, label=label)
        ax.annotate(f"{mean[-1]:.0f}%", (x[-1], mean[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", color=t["text"], fontsize=10, fontweight="bold")
    ax.set_xticks(x, ["clean", "20 dB", "10 dB", "5 dB", "0 dB"])
    ax.set_xlim(-0.2, len(LEVELS) - 0.55)
    ax.set_ylim(50, 100)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("background noise  (quieter → louder; 0 dB = noise as loud as the voice)", color=t["muted"], fontsize=10)
    fig.suptitle("Accuracy on unseen speakers as background noise gets louder", x=0.065, ha="left",
                 color=t["text"], fontsize=13, fontweight="bold")
    ax.set_title("Official test set, 11,005 clips. Mean of 3 training seeds; band = ±1 std.",
                 loc="left", color=t["muted"], fontsize=10)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    fig.tight_layout()
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def hearing_figure(t, path):
    wav = torch.as_tensor(load_wav(CLIP)).float()
    wav = torch.nn.functional.pad(wav, (0, 16000 - len(wav)))
    spec = fbank(wav, 128, 128).numpy()
    real = spec[:98]
    vmax = real.max()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=200, gridspec_kw={"width_ratios": [1.1, 1, 1]})
    fig.patch.set_facecolor(t["surface"])
    for ax in axes:
        style(ax, t)

    tt = np.arange(16000) / 16000
    axes[0].plot(tt, wav.numpy(), color=t["series"][0], linewidth=0.6)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("time (s)", color=t["muted"], fontsize=10)
    axes[0].set_yticks([])
    axes[0].set_title("① The sound: 16,000 numbers per second", loc="left", color=t["text"], fontsize=11, fontweight="bold")

    for ax, title in ((axes[1], "② Spectrogram: what the model reads"),
                      (axes[2], "③ Cut into 16×16 patches")):
        ax.imshow(real.T, origin="lower", aspect="auto", cmap="magma", vmin=vmax - 12, vmax=vmax,
                  extent=(0, 98, 0, 128))
        # Frames 98–127 are zero padding; drawn as a neutral block instead of on the colour scale
        ax.add_patch(plt.Rectangle((98, 0), 30, 128, facecolor=t["grid"], edgecolor="none"))
        ax.text(113, 64, "padding", rotation=90, ha="center", va="center", color=t["muted"], fontsize=9)
        ax.set_xlim(0, 128)
        ax.set_ylim(0, 128)
        ax.set_xlabel("time (10 ms frames)", color=t["muted"], fontsize=10)
        ax.set_title(title, loc="left", color=t["text"], fontsize=11, fontweight="bold")
    axes[1].set_ylabel("pitch (mel bin)", color=t["muted"], fontsize=10)
    # A new patch starts every 10 frames/bins and each is 16 wide, so neighbours overlap by 6
    for i, x0 in enumerate((40, 50, 60)):
        axes[2].add_patch(plt.Rectangle((x0, 80 + 3 * i), 16, 16, fill=False, edgecolor="white", linewidth=1.4))
    axes[2].text(4, 8, "a new patch every 10 steps\n(overlap 6) → 12 × 12 = 144",
                 color="white", fontsize=8.5, va="bottom")
    fig.suptitle('How the model "hears" the word "yes"', x=0.012, ha="left", color=t["text"], fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        noise_chart(t, OUT / f"noise_robustness_{mode}.png")
        hearing_figure(t, OUT / f"how_it_hears_{mode}.png")
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
