# sweep.py
# Runs a named set of experiments × 3 seeds, scheduled over all GPUs.
#   phase2:  each augmentation alone vs. none vs. all
#   phase2b: combine the two winners (shift + SpecAugment), then add the others back one at a time
# Finished runs (results.json present) are skipped, so re-running resumes an interrupted sweep.
#   python sweep.py --sweep phase2b --cache-dir /tmp/sc_cache --out-root runs_phase2b
import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

EXPERIMENTS = {
    "tat_none":    "--model tat --aug none",
    "tat_shift":   "--model tat --aug shift",
    "tat_noise":   "--model tat --aug noise",
    "tat_mixup":   "--model tat --aug mixup",
    "tat_specaug": "--model tat --aug specaug",
    "tat_ls":      "--model tat --aug ls",
    "tat_all":     "--model tat --aug all",
    "cnn_none":    "--model cnn --aug none",
    "cnn_all":     "--model cnn --aug all",
    "tat_shift_specaug":       "--model tat --aug shift,specaug",
    "tat_shift_specaug_mixup": "--model tat --aug shift,specaug,mixup",
    "tat_shift_specaug_ls":    "--model tat --aug shift,specaug,ls",
    "tat_shift_specaug_noise": "--model tat --aug shift,specaug,noise",
}

SWEEPS = {
    "phase2": ["tat_none", "tat_shift", "tat_noise", "tat_mixup", "tat_specaug", "tat_ls", "tat_all",
               "cnn_none", "cnn_all"],
    # none and noise are re-run in the same session so 2b is self-contained (and as a reproducibility check)
    "phase2b": ["tat_none", "tat_noise", "tat_shift_specaug", "tat_shift_specaug_mixup",
                "tat_shift_specaug_ls", "tat_shift_specaug_noise"],
}


def last_line(log):
    try:
        lines = log.read_text().strip().splitlines()
        return lines[-1] if lines else "starting"
    except FileNotFoundError:
        return "starting"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--sweep", choices=SWEEPS, default="phase2b")
    p.add_argument("--out-root", default="runs_phase2b")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--only", nargs="+", default=None, help="subset of experiment names")
    p.add_argument("--per-gpu", type=int, default=2, help="concurrent runs per GPU (tiny models leave a T4 idle)")
    p.add_argument("--report-every", type=int, default=120, help="seconds between progress prints")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="passed to train.py after --, e.g. -- --epochs 1")
    args = p.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    names = args.only or SWEEPS[args.sweep]
    queue = []
    for name, seed in itertools.product(names, args.seeds):
        out = Path(args.out_root) / f"{name}_s{seed}"
        if (out / "results.json").exists():
            print(f"skip {out.name} (done)")
            continue
        queue.append((out, EXPERIMENTS[name].split() + ["--seed", str(seed)]))

    n_gpu = torch.cuda.device_count()
    slots = [str(g) for g in range(n_gpu) for _ in range(args.per_gpu)] if n_gpu else [None]
    running, failed = {}, []
    print(f"{len(queue)} runs, {len(slots)} slots ({n_gpu} GPUs × {args.per_gpu})", flush=True)

    t0, last_report = time.time(), 0.0
    while queue or running:
        for slot in slots:
            if slot not in running and queue:
                out, cmd = queue.pop(0)
                out.mkdir(parents=True, exist_ok=True)
                env = {**os.environ, **({"CUDA_VISIBLE_DEVICES": slot} if slot is not None else {})}
                full = [sys.executable, "train.py", "--cache-dir", args.cache_dir, "--out-dir", str(out), *cmd, *extra]
                running[slot] = (out, subprocess.Popen(full, stdout=open(out / "train.log", "w"),
                                                       stderr=subprocess.STDOUT, env=env))
                print(f"[{(time.time() - t0) / 60:5.1f} min] start {out.name} on GPU {slot}", flush=True)
        for slot, (out, proc) in list(running.items()):
            if proc.poll() is not None:
                del running[slot]
                status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                if proc.returncode != 0:
                    failed.append(out.name)
                print(f"[{(time.time() - t0) / 60:5.1f} min] {status} {out.name}: {last_line(out / 'train.log')}",
                      flush=True)
        if time.time() - last_report > args.report_every and running:
            last_report = time.time()
            for out, _ in running.values():
                print(f"    {out.name}: {last_line(out / 'train.log')}", flush=True)
        time.sleep(5)

    print(f"\nsweep finished in {(time.time() - t0) / 60:.1f} min; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
