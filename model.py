# model.py
import torch
import torch.nn as nn
from patch_embed import PatchEmbed
from transformer_block import TransformerEncoderBlock

class TinyAudioTransformer(nn.Module):
    def __init__(self, 
                 input_shape=(1, 128, 64),  # C, H, W
                 patch_size=(16, 16),
                 embed_dim=128,
                 num_blocks=4,
                 num_heads=4,
                 num_classes=8,
                 dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbed(
            in_channels=input_shape[0],
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        num_patches = (input_shape[1] // patch_size[0]) * (input_shape[2] // patch_size[1])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [B, 1, H, W]
        B = x.shape[0]
        x = self.patch_embed(x)  # → [B, N_patches, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # prepend cls token

        x = x + self.pos_embed  # add positional encoding

        x = self.blocks(x)  # transformer blocks
        x = self.norm(x)

        cls_output = x[:, 0]  # use CLS token output
        logits = self.head(cls_output)
        return logits