# baselines.py
# A small CNN on the same features. If the transformer cannot beat this, the problem is the transformer
# (or its training), not the data pipeline.
from dataclasses import dataclass, asdict

import torch.nn as nn

from model import FeatureNorm


@dataclass
class CNNConfig:
    channels: tuple = (32, 64, 128, 128)
    num_classes: int = 35
    dropout: float = 0.1


class CNNBaseline(nn.Module):
    def __init__(self, cfg: CNNConfig = CNNConfig()):
        super().__init__()
        self.cfg = cfg
        self.norm_in = FeatureNorm()
        layers, c_in = [], 1
        for c in cfg.channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(cfg.dropout),
                                  nn.Linear(c_in, cfg.num_classes))

    def forward(self, x):
        # x: raw log-mel fbank [B, T, F]
        x = self.norm_in(x).unsqueeze(1)  # → [B, 1, T, F]
        return self.head(self.features(x))

    def config_dict(self):
        return asdict(self.cfg)
