# model.py
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from patch_embed import PatchEmbed
from transformer_block import TransformerEncoderBlock


@dataclass
class TATConfig:
    n_mels: int = 128
    n_frames: int = 128          # 1 s clip = 98 kaldi frames, zero-padded to 128 (AST speech-commands recipe)
    patch_size: int = 16
    patch_stride: int = 10       # < patch_size → overlapping patches (AST); set to 16 for ViT-style
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    num_classes: int = 35
    dropout: float = 0.1
    attn_dropout: float = 0.0
    num_prefix_tokens: int = 1   # 1 = [CLS]; 2 = [CLS]+[DIST] (AST/DeiT), pooled as their mean
    head_norm: bool = False      # extra LayerNorm inside the head (AST's ASTMLPHead has one)
    ln_eps: float = 1e-6


class FeatureNorm(nn.Module):
    """(x - mean) / (2 * std), as in AST. Stats live in the model so they ship with the checkpoint."""

    def __init__(self, mean=0.0, std=0.5):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def set_stats(self, mean, std):
        self.mean.fill_(float(mean))
        self.std.fill_(float(std))

    def forward(self, x):
        return (x - self.mean) / (2 * self.std)


class TinyAudioTransformer(nn.Module):
    def __init__(self, cfg: TATConfig = TATConfig()):
        super().__init__()
        self.cfg = cfg
        self.norm_in = FeatureNorm()

        self.patch_embed = PatchEmbed(cfg.n_mels, cfg.n_frames, cfg.patch_size, cfg.patch_stride, cfg.embed_dim)
        n_tokens = self.patch_embed.num_patches + cfg.num_prefix_tokens

        self.prefix_tokens = nn.Parameter(torch.zeros(1, cfg.num_prefix_tokens, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio,
                                    cfg.dropout, cfg.attn_dropout, cfg.ln_eps)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.ln_eps)
        self.head_norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.ln_eps) if cfg.head_norm else nn.Identity()
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.prefix_tokens, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(m):
        # ViT/DeiT init: small truncated normal for linears, identity LayerNorms
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            nn.init.trunc_normal_(m.in_proj_weight, std=0.02)
            nn.init.zeros_(m.in_proj_bias)

    def forward(self, x):
        # x: raw log-mel fbank [B, T, F]
        B = x.shape[0]
        x = self.norm_in(x)
        x = self.patch_embed(x)                                 # → [B, N, D]

        prefix = self.prefix_tokens.expand(B, -1, -1)           # → [B, P, D]
        x = torch.cat((prefix, x), dim=1) + self.pos_embed      # → [B, P+N, D]
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        pooled = x[:, :self.cfg.num_prefix_tokens].mean(dim=1)  # [CLS] (or mean of [CLS],[DIST])
        return self.head(self.head_norm(pooled))

    def config_dict(self):
        return asdict(self.cfg)
