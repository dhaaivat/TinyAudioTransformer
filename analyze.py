# analyze.py
# Summarises a sweep: mean ± std over seeds, and each config's change vs. its "none" baseline,
# paired by seed (seed s of a config is compared with seed s of the baseline).
#   python analyze.py runs_phase2            → prints a table and writes runs_phase2/summary.md
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from metrics import mcnemar_p


def load_runs(root):
    runs = defaultdict(dict)
    for f in sorted(Path(root).glob("*/results.json")):
        m = re.fullmatch(r"(.+)_s(\d+)", f.parent.name)
        if m:
            runs[m.group(1)][int(m.group(2))] = json.loads(f.read_text())
    return runs


def test_logits(root, name, seed):
    return torch.load(Path(root) / f"{name}_s{seed}" / "test_logits.pt", weights_only=False)


def correct_vector(root, name, seed):
    d = test_logits(root, name, seed)
    return (d["logits"].argmax(1) == d["y"]).numpy()


def overconfidence(root, name, seed):
    # mean confidence − accuracy, in points: > 0 overconfident, < 0 underconfident (ECE alone has no sign)
    d = test_logits(root, name, seed)
    conf, pred = d["logits"].float().softmax(1).max(1)
    return (conf.mean() - (pred == d["y"]).float().mean()).item() * 100


def fmt(xs):
    return f"{np.mean(xs):.2f} ± {np.std(xs, ddof=1) if len(xs) > 1 else 0:.2f}"


def parent_of(name, names):
    # The longest other config whose name this one extends: tat_shift_specaug_mixup → tat_shift_specaug.
    # Falls back to <model>_none.
    prefixes = [n for n in names if name.startswith(n + "_")]
    return max(prefixes, key=len) if prefixes else name.split("_")[0] + "_none"


def compare(root, runs, name, ref):
    # Paired by seed: Δ test acc (mean, min, max), seeds better, per-seed McNemar p
    common = [s for s in sorted(runs[name]) if s in runs.get(ref, {})]
    if name == ref or not common:
        return ["—", "—", "—"]
    deltas = [runs[name][s]["test_acc"] - runs[ref][s]["test_acc"] for s in common]
    ps = [mcnemar_p(correct_vector(root, name, s), correct_vector(root, ref, s)) for s in common]
    return [f"{np.mean(deltas):+.2f} (min {min(deltas):+.2f}, max {max(deltas):+.2f})",
            f"{sum(d > 0 for d in deltas)}/{len(deltas)}", ", ".join(f"{p:.2g}" for p in ps)]


def summarise(root):
    runs = load_runs(root)
    header = ("| config | seeds | params | val acc | test acc | test ECE | conf − acc | Δ test vs none (paired) "
              "| seeds better | McNemar p (per seed) | parent | Δ vs parent | seeds better | McNemar p |")
    lines = [header, "|" + "---|" * 14]
    for name, by_seed in runs.items():
        seeds = sorted(by_seed)
        r0 = by_seed[seeds[0]]
        row = [name, str(len(seeds)), f"{r0['params'] / 1e6:.2f}M",
               fmt([by_seed[s]["best_val_acc"] for s in seeds]),
               fmt([by_seed[s]["test_acc"] for s in seeds]),
               fmt([by_seed[s]["test_ece"] for s in seeds]),
               f"{np.mean([overconfidence(root, name, s) for s in seeds]):+.1f}"]
        base, parent = name.split("_")[0] + "_none", parent_of(name, runs)
        row += compare(root, runs, name, base)
        row += [parent, *compare(root, runs, name, parent)] if parent != base else ["—", "—", "—", "—"]
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    note = ("\nΔ is paired by seed. A real effect should be positive for every seed and clearly larger than "
            "the seed-to-seed std of the baseline. McNemar p < 0.05 means that seed's two models disagree "
            "in a way unlikely to be chance. conf − acc > 0 is overconfident, < 0 underconfident. "
            "\"parent\" is the config this one extends (e.g. shift_specaug for shift_specaug_mixup).\n")
    Path(root, "summary.md").write_text(table + "\n" + note)
    return table + "\n" + note


if __name__ == "__main__":
    print(summarise(sys.argv[1] if len(sys.argv) > 1 else "runs_phase2"))
