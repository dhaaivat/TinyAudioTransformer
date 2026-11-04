# prepare_data.py
# Phase 0: download Speech Commands v2, check the official split, cache 1 s int16 waveforms + noise bank.
import argparse
from collections import Counter

from dataset import EXPECTED_SIZES, build_cache, download_speech_commands, official_splits, speaker_id


def check_splits(root):
    splits, classes = official_splits(root)
    assert len(classes) == 35, f"expected 35 words, found {len(classes)}: {classes}"

    speakers = {s: {speaker_id(r) for r, _ in items} for s, items in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = speakers[a] & speakers[b]
        assert not shared, f"{len(shared)} speakers in both {a} and {b}"

    for s, items in splits.items():
        per_class = Counter(label for _, label in items)
        print(f"{s:>5}: {len(items):6d} clips  {len(speakers[s]):5d} speakers  "
              f"{len(per_class)} classes  (min/max per class {min(per_class.values())}/{max(per_class.values())})")
        if len(items) != EXPECTED_SIZES[s]:
            print(f"  WARNING: expected {EXPECTED_SIZES[s]} clips for {s}")
    print("Splits are speaker-disjoint.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/speech_commands_v2")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    download_speech_commands(args.root)
    check_splits(args.root)
    build_cache(args.root, args.cache_dir, num_workers=args.num_workers)
