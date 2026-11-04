# eval_noisy.py
# Robustness check: re-scores every trained checkpoint on val and test with *held-out* background noise
# (noise_test.pt, never used in training) mixed in at fixed SNRs. Each clip gets one fixed noise segment,
# the same for every SNR and every model, so comparisons between models are paired like the clean ones.
# Decisions (e.g. which noise setting to keep) should be made on val; test is for reporting.
#   python eval_noisy.py runs_phase2b --cache-dir /tmp/sc_cache   → noisy.json per run + noisy_summary.md
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from augment import mix_at_snr
from baselines import CNNBaseline, CNNConfig
from dataset import load_cache, pcm_to_float
from features import fbank
from metrics import accuracy
from model import TATConfig, TinyAudioTransformer
from train import N_FRAMES, N_MELS, all_features, predict

NOISE_SEED = {"val": 1, "test": 2}


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["model"] == "cnn":
        model = CNNBaseline(CNNConfig(**{**ck["config"], "channels": tuple(ck["config"]["channels"])}))
    else:
        model = TinyAudioTransformer(TATConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def noisy_features(pcm, noise_bank, snr_db, seed, batch_size=1024):
    # The generator depends only on the split, so clip i always gets the same segment; only the SNR varies
    idx = torch.randint(len(noise_bank), (len(pcm),), generator=torch.Generator().manual_seed(seed))
    out = []
    for i in range(0, len(pcm), batch_size):
        wav = pcm_to_float(pcm[i:i + batch_size])
        wav = mix_at_snr(wav, noise_bank[idx[i:i + batch_size]].to(wav.device), float(snr_db))
        out.append(fbank(wav, N_MELS, N_FRAMES).half())  # no clipping: we want the exact SNR
    return torch.cat(out)


def fmt(xs):
    return f"{np.mean(xs):.2f} ± {np.std(xs, ddof=1) if len(xs) > 1 else 0:.2f}"


def summarise(root, levels):
    runs = defaultdict(dict)
    for f in sorted(Path(root).glob("*/noisy.json")):
        m = re.fullmatch(r"(.+)_s(\d+)", f.parent.name)
        if m:
            runs[m.group(1)][int(m.group(2))] = json.loads(f.read_text())
    out = []
    for split in ("val", "test"):
        out += [f"\n### {split}: accuracy with held-out noise (mean ± std over seeds; Δ vs none, paired by seed)\n",
                "| config | " + " | ".join(f"{l} dB" if l != "clean" else "clean" for l in levels) + " |",
                "|---|" + "---|" * len(levels)]
        for name, by_seed in runs.items():
            base = runs.get(name.split("_")[0] + "_none", {})
            cells = []
            for l in levels:
                accs = [by_seed[s][split][l] for s in sorted(by_seed)]
                cell = fmt(accs)
                common = [s for s in by_seed if s in base]
                if base and not name.endswith("_none") and common:
                    cell += f" ({np.mean([by_seed[s][split][l] - base[s][split][l] for s in common]):+.2f})"
                cells.append(cell)
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    Path(root, "noisy_summary.md").write_text(text)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs_root")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--snrs", type=float, nargs="+", default=[20, 10, 5, 0])
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"

    run_dirs = sorted(d.parent for d in Path(args.runs_root).glob("*/best.pt"))
    models = {d: load_model(d / "best.pt", device) for d in run_dirs}
    noise_bank = pcm_to_float(torch.load(Path(args.cache_dir) / "noise_test.pt"))
    levels = ["clean"] + [f"{s:g}" for s in args.snrs]
    print(f"{len(models)} checkpoints, {len(noise_bank)} held-out noise segments, levels {levels}")

    scores = {d: {} for d in run_dirs}
    for split in ("val", "test"):
        data = load_cache(args.cache_dir, split)
        pcm, y = data["pcm"].to(device), data["y"].to(device)
        for level in levels:
            x = all_features(pcm) if level == "clean" else noisy_features(pcm, noise_bank, float(level), NOISE_SEED[split])
            for d, model in models.items():
                scores[d].setdefault(split, {})[level] = accuracy(predict(model, x, 1024, amp), y)
            print(f"{split} {level:>5}: done", flush=True)
        del pcm, x

    for d in run_dirs:
        (d / "noisy.json").write_text(json.dumps(scores[d], indent=2))
        # Sanity check: the clean test score must reproduce what train.py reported for this checkpoint
        reported = json.loads((d / "results.json").read_text())["test_acc"]
        if abs(scores[d]["test"]["clean"] - reported) > 0.05:
            print(f"WARNING {d.name}: clean test {scores[d]['test']['clean']:.2f} ≠ reported {reported:.2f}")
    print(summarise(args.runs_root, levels))


if __name__ == "__main__":
    main()
