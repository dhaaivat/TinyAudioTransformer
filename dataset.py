# dataset.py
import os
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

class MiniSpeechCommandsMelDataset(Dataset):
    def __init__(self, root_dir=DATASET_DIR, split="train", sample_rate=16000, mel_bins=128):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=mel_bins,
            n_fft=1024,
            hop_length=256
        )
        self.db_transform = AmplitudeToDB()

        self.all_files = []
        self.labels = []

        # Build label set
        self.label_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}

        # Load files
        for label in self.label_names:
            files = list((self.root_dir / label).glob("*.wav"))
            for f in files:
                self.all_files.append(f)
                self.labels.append(self.label_to_idx[label])

        # Split
        total = len(self.all_files)
        if split == "train":
            self.all_files = self.all_files[:int(0.8 * total)]
            self.labels = self.labels[:int(0.8 * total)]
        elif split == "val":
            self.all_files = self.all_files[int(0.8 * total):int(0.9 * total)]
            self.labels = self.labels[int(0.8 * total):int(0.9 * total)]
        elif split == "test":
            self.all_files = self.all_files[int(0.9 * total):]
            self.labels = self.labels[int(0.9 * total):]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        mel_spec = self.mel_transform(waveform)      # → [1, mel_bins, time]
        mel_spec = self.db_transform(mel_spec)        # → log scale
        mel_spec = mel_spec[:, :128, :64]             # Ensure consistent shape

        return mel_spec, label
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32):
    train_set = MiniSpeechCommandsMelDataset(split="train")
    val_set = MiniSpeechCommandsMelDataset(split="val")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    return train_loader, val_loader

