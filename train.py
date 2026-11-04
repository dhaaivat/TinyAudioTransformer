# train.py
# Trains on cached 1 s waveforms. Per batch, on GPU:
#   pcm → [time shift → mixup → background noise] → kaldi fbank → [SpecAugment] → model
# Augmentations are switched on by name with --aug, so each one can be measured on its own.
# Picks the checkpoint on val, then evaluates on test exactly once.
#   python train.py --model tat --aug none                 # Phase-1-style baseline
#   python train.py --model tat --aug all                  # every augmentation
#   python train.py --model tat --overfit-batch 64         # sanity check: must reach 100% on one batch
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import augment
from baselines import CNNBaseline, CNNConfig
from dataset import load_cache, pcm_to_float
from features import fbank, num_frames
from metrics import accuracy, confusion_matrix, expected_calibration_error
from model import TATConfig, TinyAudioTransformer

AUGS = ("shift", "mixup", "noise", "specaug", "ls")
N_MELS, N_FRAMES = 128, 128


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["tat", "cnn"], default="tat")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out-dir", default="runs/tat")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="use only N clips per split (smoke tests)")
    p.add_argument("--overfit-batch", type=int, default=0, help="train on one batch of N clips until 100%%")
    # Augmentation: comma-separated subset of AUGS, or "none" / "all"
    p.add_argument("--aug", default="none")
    p.add_argument("--shift-ms", type=float, default=100)
    p.add_argument("--mixup-prob", type=float, default=0.6)
    p.add_argument("--mixup-alpha", type=float, default=10.0)
    p.add_argument("--noise-prob", type=float, default=0.8)
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--snr-max", type=float, default=20.0)
    p.add_argument("--freqm", type=int, default=24)
    p.add_argument("--timem", type=int, default=24)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    # TinyAudioTransformer size
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--patch-stride", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)
    args = p.parse_args(argv)

    args.aug = set(AUGS) if args.aug == "all" else set() if args.aug == "none" else set(args.aug.split(","))
    unknown = args.aug - set(AUGS)
    if unknown:
        p.error(f"unknown --aug {sorted(unknown)}; choose from {AUGS}")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, num_classes):
    if args.model == "cnn":
        return CNNBaseline(CNNConfig(num_classes=num_classes))
    return TinyAudioTransformer(TATConfig(
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.heads, mlp_ratio=args.mlp_ratio,
        patch_stride=args.patch_stride, dropout=args.dropout, num_classes=num_classes))


def load_split(cache_dir, split, device, limit=0):
    d = load_cache(cache_dir, split)
    if limit:
        idx = torch.randperm(len(d["y"]), generator=torch.Generator().manual_seed(0))[:limit]
        d["pcm"], d["y"] = d["pcm"][idx], d["y"][idx]
        d["files"] = [d["files"][i] for i in idx.tolist()]
    # Whole split fits on a T4 as int16 (train ≈ 2.7 GB), so batching is just indexing
    d["pcm"], d["y"] = d["pcm"].to(device), d["y"].to(device)
    return d


def features(pcm):
    # fbank stays in fp32 (outside autocast): power spectra span many orders of magnitude
    return fbank(pcm_to_float(pcm), N_MELS, N_FRAMES)


@torch.no_grad()
def all_features(pcm, batch_size=1024):
    return torch.cat([features(pcm[i:i + batch_size]).half() for i in range(0, len(pcm), batch_size)])


@torch.no_grad()
def feature_stats(pcm, batch_size=1024):
    # Mean/std of clean train features, padding included (the model sees padding too); streamed to save memory
    s = s2 = n = 0.0
    for i in range(0, len(pcm), batch_size):
        x = features(pcm[i:i + batch_size]).double()
        s, s2, n = s + x.sum().item(), s2 + x.pow(2).sum().item(), n + x.numel()
    mean = s / n
    return mean, (s2 / n - mean ** 2) ** 0.5


@torch.no_grad()
def predict(model, x, batch_size, amp):
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=amp):
            out.append(model(x[i:i + batch_size].float()).float())
    return torch.cat(out)


def augmented_batch(pcm, y, args, noise_bank, mean, num_classes):
    wav = pcm_to_float(pcm)
    targets = nn.functional.one_hot(y, num_classes).float()
    if "shift" in args.aug:
        wav = augment.time_shift(wav, int(args.shift_ms * 16))
    if "mixup" in args.aug:
        wav, targets = augment.mixup(wav, targets, args.mixup_alpha, args.mixup_prob)
    if "noise" in args.aug:
        wav = augment.add_background_noise(wav, noise_bank, (args.snr_min, args.snr_max), args.noise_prob)
    x = fbank(wav, N_MELS, N_FRAMES)
    if "specaug" in args.aug:
        x = augment.spec_augment(x, args.freqm, args.timem, value=mean, valid_frames=num_frames(pcm.shape[1]))
    if "ls" in args.aug:
        targets = augment.smooth_labels(targets, args.label_smoothing)
    return x, targets


def overfit_one_batch(model, train, args, device, amp, max_steps=1000):
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(train["y"]), generator=g)[:args.overfit_batch].to(device)
    x, y = features(train["pcm"][idx]), train["y"][idx]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    for step in range(1, max_steps + 1):
        model.train()
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            acc = accuracy(predict(model, x, len(x), amp), y)
            print(f"step {step:4d}  loss {loss.item():.4f}  acc {acc:.1f}%", flush=True)
            if acc == 100.0:
                print(f"PASS: memorised {len(y)} clips in {step} steps")
                return True
    print(f"FAIL: did not reach 100% on {len(y)} clips in {max_steps} steps")
    return False


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(args.cache_dir, "train", device, args.limit)
    classes = train["classes"]
    model = build_model(args, len(classes)).to(device)
    mean, std = feature_stats(train["pcm"])
    model.norm_in.set_stats(mean, std)  # clean train-split stats only
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params / 1e6:.3f}M params, {len(train['y'])} train clips, device {device}, "
          f"aug {sorted(args.aug) or 'none'}, feature mean {mean:.3f} std {std:.3f}")

    if args.overfit_batch:
        sys.exit(0 if overfit_one_batch(model, train, args, device, amp) else 1)

    noise_bank = pcm_to_float(torch.load(Path(args.cache_dir) / "noise.pt").to(device)) if "noise" in args.aug else None
    val = load_split(args.cache_dir, "val", device, args.limit)
    val_x = all_features(val["pcm"])  # clean eval features, computed once

    # No weight decay on biases, norms, tokens and position embeddings
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "pos_embed" not in n and "prefix" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "pos_embed" not in n and "prefix" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    steps_per_epoch = math.ceil(len(train["y"]) / args.batch_size)
    total, warmup = args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warmup) if s < warmup
                                              else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    g = torch.Generator(device=device).manual_seed(args.seed)

    history, best_val, best_epoch = [], -1.0, -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train["y"]), device=device, generator=g)
        loss_sum, correct, seen = 0.0, 0, 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            y = train["y"][idx]
            x, targets = augmented_batch(train["pcm"][idx], y, args, noise_bank, mean, len(classes))
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = augment.soft_cross_entropy(logits.float(), targets)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()  # vs the dominant (unmixed) label
            seen += len(y)

        val_logits = predict(model, val_x, 1024, amp)
        val_acc = accuracy(val_logits, val["y"])
        val_nll = nn.functional.cross_entropy(val_logits, val["y"]).item()
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": 100 * correct / seen,
                        "val_acc": val_acc, "val_nll": val_nll, "lr": sched.get_last_lr()[0],
                        "time_s": time.time() - t0})
        print(f"epoch {epoch:3d}  loss {loss_sum / seen:.4f}  train {100 * correct / seen:5.2f}%  "
              f"val {val_acc:5.2f}%  val_nll {val_nll:.3f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s",
              flush=True)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            torch.save({"model": args.model, "config": model.config_dict(), "state_dict": model.state_dict(),
                        "classes": classes, "epoch": epoch, "val_acc": val_acc}, out_dir / "best.pt")

    # Test once, with the val-selected checkpoint
    del train
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["state_dict"])
    val_logits = predict(model, val_x, 1024, amp)
    test = load_split(args.cache_dir, "test", device, args.limit)
    logits = predict(model, all_features(test["pcm"]), 1024, amp)
    preds, y = logits.argmax(1).cpu(), test["y"].cpu()
    cm = confusion_matrix(preds, y, len(classes))
    per_class = (cm.diag().float() / cm.sum(1).clamp(min=1) * 100).tolist()

    results = {
        "model": args.model, "config": model.config_dict(), "params": n_params, "seed": args.seed,
        "aug": sorted(args.aug), "best_epoch": best_epoch, "best_val_acc": best_val,
        "val_ece": expected_calibration_error(val_logits, val["y"]),
        "test_acc": accuracy(logits, test["y"]), "test_ece": expected_calibration_error(logits, test["y"]),
        "test_nll": nn.functional.cross_entropy(logits, test["y"]).item(),
        "per_class_test_acc": dict(zip(classes, per_class)), "confusion_matrix": cm.tolist(),
        "train_time_s": time.time() - t0, "args": {**vars(args), "aug": sorted(args.aug)},
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({"logits": logits.cpu(), "y": y, "files": test["files"], "classes": classes}, out_dir / "test_logits.pt")
    print(f"\nbest val {best_val:.2f}% (epoch {best_epoch})  →  TEST {results['test_acc']:.2f}%  "
          f"ECE {results['test_ece']:.2f}%")


if __name__ == "__main__":
    main()
