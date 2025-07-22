
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=1, patch_size=(16, 16), embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )  # [B, embed_dim, H/Ph, W/Pw]

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # → [B, embed_dim, H', W']
        x = x.flatten(2)  # → [B, embed_dim, N_patches]
        x = x.transpose(1, 2)  # → [B, N_patches, embed_dim]
        return x