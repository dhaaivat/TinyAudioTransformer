# dataset.py
# Speech Commands v2, 35 words, official split (validation_list.txt / testing_list.txt).
# The official lists are speaker-disjoint: the part of the filename before "_nohash_" is the speaker id.
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from features import SAMPLE_RATE

SC_V2_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
EXPECTED_SIZES = {"train": 84843, "val": 9981, "test": 11005}
CLIP_SAMPLES = SAMPLE_RATE  # every clip is zero-padded / cropped to 1 s


def download_speech_commands(root):
    root = Path(root)
    if (root / "testing_list.txt").exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "speech_commands_v0.02.tar.gz"
    if not archive.exists():
        print(f"Downloading {SC_V2_URL} (~2.3 GB)")
        urllib.request.urlretrieve(SC_V2_URL, archive)
    print("Extracting")
    with tarfile.open(archive) as tar:
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(root, **kwargs)
    archive.unlink()
    return root


def speaker_id(rel_path):
    return Path(rel_path).name.split("_nohash_")[0]


def class_names(root):
    return sorted(d.name for d in Path(root).iterdir() if d.is_dir() and not d.name.startswith("_"))


def official_splits(root):
    """Returns {split: [(relative_path, label_idx), ...]}, classes."""
    root = Path(root)
    classes = class_names(root)
    idx = {c: i for i, c in enumerate(classes)}
    val = set(root.joinpath("validation_list.txt").read_text().split())
    test = set(root.joinpath("testing_list.txt").read_text().split())

    splits = {"train": [], "val": [], "test": []}
    for c in classes:
        for f in sorted((root / c).glob("*.wav")):
            rel = f"{c}/{f.name}"
            split = "val" if rel in val else "test" if rel in test else "train"
            splits[split].append((rel, idx[c]))
    return splits, classes


def load_pcm(path):
    # Speech Commands is 16-bit mono PCM at 16 kHz
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def load_wav(path):
    # float in [-1, 1], like torchaudio.load
    return torch.from_numpy(load_pcm(path).astype(np.float32) / 32768.0)


def pcm_to_float(pcm):
    return pcm.float() / 32768.0


class SpeechCommandsPCM(Dataset):
    def __init__(self, root, items):
        self.root, self.items = Path(root), items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        pcm = torch.from_numpy(load_pcm(self.root / rel)[:CLIP_SAMPLES].copy())
        return torch.nn.functional.pad(pcm, (0, CLIP_SAMPLES - len(pcm))), label


def background_noise_bank(root, part="train", holdout=0.2, hop=CLIP_SAMPLES // 2):
    # Cuts the long _background_noise_ recordings into overlapping 1 s segments: [S, 16000] int16.
    # Each recording is split in time: the first 80% feeds training augmentation ("train"), the last
    # 20% is kept for the noisy evaluation ("test"), so evaluation never replays noise seen in training.
    segs = []
    for f in sorted(Path(root, "_background_noise_").glob("*.wav")):
        pcm = torch.from_numpy(load_pcm(f).copy())
        cut = int(len(pcm) * (1 - holdout))
        pcm = pcm[:cut] if part == "train" else pcm[cut:]
        segs += [pcm[s:s + CLIP_SAMPLES] for s in range(0, len(pcm) - CLIP_SAMPLES + 1, hop)]
    return torch.stack(segs)


def build_cache(root, cache_dir, num_workers=4):
    """Stores each split as int16 waveforms [N, 16000] (lossless, ~2.7 GB for train), plus the noise bank.
    Features are computed on the GPU per batch, after waveform augmentation."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits, classes = official_splits(root)

    for split, items in splits.items():
        out = cache_dir / f"{split}.pt"
        if out.exists():
            continue
        loader = DataLoader(SpeechCommandsPCM(root, items), batch_size=1024, num_workers=num_workers)
        pcm = torch.empty(len(items), CLIP_SAMPLES, dtype=torch.int16)
        y = torch.empty(len(items), dtype=torch.long)
        n = 0
        for p, labels in loader:
            pcm[n:n + len(p)], y[n:n + len(p)] = p, labels
            n += len(p)
        torch.save({"pcm": pcm, "y": y, "files": [r for r, _ in items], "classes": classes}, out)
        print(f"{split}: {tuple(pcm.shape)} int16")

    for part, name in (("train", "noise.pt"), ("test", "noise_test.pt")):
        noise_out = cache_dir / name
        if not noise_out.exists():
            bank = background_noise_bank(root, part)
            torch.save(bank, noise_out)
            print(f"noise bank ({part}): {tuple(bank.shape)} segments")


def load_cache(cache_dir, split):
    return torch.load(Path(cache_dir) / f"{split}.pt", weights_only=False)
