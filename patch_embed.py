# patch_embed.py
import torch
import torch.nn as nn


def num_patches_1d(size, patch, stride):
    # Number of conv windows along one axis (no padding), same formula as AST's get_shape
    return (size - patch) // stride + 1


class PatchEmbed(nn.Module):
    """Spectrogram -> sequence of patch embeddings.

    Input follows the AST convention: [B, T, F] (time frames, mel bins), which is also
    what kaldi fbank returns. With stride < patch the patches overlap (AST: 16x16, stride 10).
    """

    def __init__(self, n_mels=128, n_frames=128, patch_size=16, patch_stride=10, embed_dim=128):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.grid = (num_patches_1d(n_mels, patch_size, patch_stride),
                     num_patches_1d(n_frames, patch_size, patch_stride))
        self.num_patches = self.grid[0] * self.grid[1]

        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_stride)

    def forward(self, x):
        # x: [B, T, F]
        if x.shape[1:] != (self.n_frames, self.n_mels):
            raise ValueError(f"expected input [B, {self.n_frames}, {self.n_mels}], got {list(x.shape)}")
        x = x.unsqueeze(1).transpose(2, 3)  # → [B, 1, F, T]
        x = self.proj(x)                    # → [B, embed_dim, F', T']
        x = x.flatten(2).transpose(1, 2)    # → [B, F'*T', embed_dim]  (frequency-major order, as in AST)
        return x
