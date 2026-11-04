import pytest
import torch
import torch.nn as nn

from baselines import CNNBaseline, CNNConfig
from model import TATConfig, TinyAudioTransformer


@pytest.mark.parametrize("stride,grid", [(10, (12, 12)), (16, (8, 8))])
def test_patch_count_matches_conv_output(stride, grid):
    m = TinyAudioTransformer(TATConfig(patch_stride=stride))
    assert m.patch_embed.grid == grid
    with torch.no_grad():
        conv_out = m.patch_embed.proj(torch.zeros(1, 1, 128, 128))
    assert tuple(conv_out.shape[2:]) == grid
    assert m.pos_embed.shape[1] == grid[0] * grid[1] + m.cfg.num_prefix_tokens


@pytest.mark.parametrize("prefix,head_norm", [(1, False), (2, True)])
@pytest.mark.parametrize("batch", [1, 3])
def test_forward_shape(prefix, head_norm, batch):
    m = TinyAudioTransformer(TATConfig(num_prefix_tokens=prefix, head_norm=head_norm)).eval()
    assert m(torch.randn(batch, 128, 128)).shape == (batch, 35)


@pytest.mark.parametrize("shape", [(2, 63, 128), (2, 1, 128, 64), (2, 128, 64)])
def test_rejects_wrong_input_shape(shape):
    # The old pipeline produced 63 frames for a 1 s clip and crashed inside the pos_embed add
    with pytest.raises(ValueError, match="expected input"):
        TinyAudioTransformer()(torch.randn(*shape))


def test_eval_is_deterministic():
    m = TinyAudioTransformer().eval()
    x = torch.randn(2, 128, 128)
    torch.testing.assert_close(m(x), m(x))


def test_init_scale():
    m = TinyAudioTransformer()
    assert m.pos_embed.std().item() == pytest.approx(0.02, rel=0.2)
    assert m.prefix_tokens.abs().max().item() < 0.1  # trunc_normal_ bounds are absolute (±2), so this is ~4σ


@pytest.mark.parametrize("model", ["tat", "cnn"])
def test_overfits_one_batch(model):
    # A model that cannot memorise 32 examples has a bug (wiring, masking, gradient flow)
    torch.manual_seed(0)
    if model == "tat":
        m = TinyAudioTransformer(TATConfig(embed_dim=64, depth=2, num_heads=2, patch_stride=16, dropout=0.0))
    else:
        m = CNNBaseline(CNNConfig(dropout=0.0))  # 2-layer version is too small to memorise pure noise through avg-pool
    m.norm_in.set_stats(0.0, 1.0)
    x, y = torch.randn(32, 128, 128), torch.randint(0, 35, (32,))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(300):
        m.train()
        loss = nn.functional.cross_entropy(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        assert (m(x).argmax(1) == y).all()
