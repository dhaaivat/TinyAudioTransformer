# ast_reference.py
# Loads the authors' fine-tuned AST (MIT/ast-finetuned-speech-commands-v2, via HF transformers) into
# TinyAudioTransformer. If our modules compute the same function as AST, logits match to float precision;
# any architectural misunderstanding (patch order, token pooling, LayerNorm placement, attention layout)
# shows up as a mismatch. Used by tests/test_parity_ast.py.
import torch

from model import TATConfig, TinyAudioTransformer

HF_AST_SC = "MIT/ast-finetuned-speech-commands-v2"


def config_from_hf(hf_cfg):
    assert hf_cfg.frequency_stride == hf_cfg.time_stride, "TAT uses one stride for both axes"
    return TATConfig(
        n_mels=hf_cfg.num_mel_bins, n_frames=hf_cfg.max_length,
        patch_size=hf_cfg.patch_size, patch_stride=hf_cfg.time_stride,
        embed_dim=hf_cfg.hidden_size, depth=hf_cfg.num_hidden_layers, num_heads=hf_cfg.num_attention_heads,
        mlp_ratio=hf_cfg.intermediate_size / hf_cfg.hidden_size, num_classes=hf_cfg.num_labels,
        dropout=0.0, num_prefix_tokens=2, head_norm=True, ln_eps=hf_cfg.layer_norm_eps,
    )


# transformers renamed AST's modules over versions; rewrite older spellings to the 5.x ones.
# Order matters: attention.output.dense must be renamed before the MLP's output.dense.
_RENAMES = [
    (".encoder.layer.", ".layers."),
    (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."),
    (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."),
    (".intermediate.dense.", ".mlp.fc1."),
    (".output.dense.", ".mlp.fc2."),
]


class _TrackedDict(dict):
    # Records which tensors were read, so a mapping that silently skips weights fails loudly
    def __init__(self, *a):
        super().__init__(*a)
        self.used = set()

    def __getitem__(self, k):
        self.used.add(k)
        return super().__getitem__(k)


def _to_current_names(sd):
    out = _TrackedDict()
    for k, v in sd.items():
        for old, new in _RENAMES:
            k = k.replace(old, new)
        out[k] = v
    return out


def map_hf_state_dict(sd, depth):
    sd = _to_current_names(sd)
    e = "audio_spectrogram_transformer.embeddings."
    out = {
        "prefix_tokens": torch.cat([sd[e + "cls_token"], sd[e + "distillation_token"]], dim=1),
        "pos_embed": sd[e + "position_embeddings"],
        "patch_embed.proj.weight": sd[e + "patch_embeddings.projection.weight"],
        "patch_embed.proj.bias": sd[e + "patch_embeddings.projection.bias"],
        "norm.weight": sd["audio_spectrogram_transformer.layernorm.weight"],
        "norm.bias": sd["audio_spectrogram_transformer.layernorm.bias"],
        "head_norm.weight": sd["classifier.layernorm.weight"],
        "head_norm.bias": sd["classifier.layernorm.bias"],
        "head.weight": sd["classifier.dense.weight"],
        "head.bias": sd["classifier.dense.bias"],
    }
    for i in range(depth):
        h, o = f"audio_spectrogram_transformer.layers.{i}.", f"blocks.{i}."
        for p in ("weight", "bias"):
            qkv = [sd[h + f"attention.{n}_proj.{p}"] for n in "qkv"]
            out[o + f"attn.in_proj_{p}"] = torch.cat(qkv, dim=0)
            out[o + f"attn.out_proj.{p}"] = sd[h + f"attention.o_proj.{p}"]
            out[o + f"norm1.{p}"] = sd[h + f"layernorm_before.{p}"]
            out[o + f"norm2.{p}"] = sd[h + f"layernorm_after.{p}"]
            out[o + f"mlp.0.{p}"] = sd[h + f"mlp.fc1.{p}"]
            out[o + f"mlp.3.{p}"] = sd[h + f"mlp.fc2.{p}"]
    unused = set(sd) - sd.used
    assert not unused, f"AST tensors not mapped: {sorted(unused)[:5]}"
    return out


def load_ast_into_tat(name=HF_AST_SC):
    """Returns (our model with AST weights, the HF model, the HF feature extractor)."""
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    hf = ASTForAudioClassification.from_pretrained(name).eval()
    fe = ASTFeatureExtractor.from_pretrained(name)
    cfg = config_from_hf(hf.config)

    ours = TinyAudioTransformer(cfg).eval()
    missing, unexpected = ours.load_state_dict(map_hf_state_dict(hf.state_dict(), cfg.depth), strict=False)
    assert not unexpected, unexpected
    assert set(missing) == {"norm_in.mean", "norm_in.std"}, missing
    ours.norm_in.set_stats(fe.mean, fe.std)
    return ours, hf, fe
