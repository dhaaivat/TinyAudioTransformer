# transformer_block.py
import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    """Pre-norm ViT block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0, ln_eps=1e-6):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim, eps=ln_eps)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim, eps=ln_eps)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)  # normalise once, use as q, k and v
        x = x + self.drop1(self.attn(h, h, h, need_weights=False)[0])
        x = x + self.mlp(self.norm2(x))
        return x
