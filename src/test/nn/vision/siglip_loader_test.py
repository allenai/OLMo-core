import torch
from safetensors.torch import save_file

from olmo_core.config import DType
from olmo_core.nn.vision import (
    VisionEncoderConfig,
    VisionEncoderType,
    VisionTransformer,
    load_siglip_hf_vision_state_dict,
    siglip_hf_state_dict_to_vision,
    vision_state_fingerprint,
)


def _tiny_siglip_config() -> VisionEncoderConfig:
    return VisionEncoderConfig(
        name=VisionEncoderType.siglip2,
        use_cls_token=False,
        patch_embedding_bias=True,
        use_pre_ln=False,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_emb_dim=16,
        image_num_heads=2,
        image_num_key_value_heads=2,
        image_num_layers=2,
        image_head_dim=8,
        image_mlp_dim=32,
        image_num_pos=4,
        dtype=DType.float32,
    )


def _native_to_hf(native: dict[str, torch.Tensor], cfg: VisionEncoderConfig):
    hf = {
        "embeddings.patch_embedding.weight": native["patch_embedding.weight"].reshape(
            cfg.image_emb_dim, 3, cfg.image_patch_size, cfg.image_patch_size
        ),
        "embeddings.patch_embedding.bias": native["patch_embedding.bias"],
        "embeddings.position_embedding.weight": native["positional_embedding"],
    }
    for layer_idx in range(cfg.image_num_layers):
        source = f"blocks.{layer_idx}"
        target = f"encoder.layers.{layer_idx}"
        for native_name, hf_name in (
            ("attn_norm", "layer_norm1"),
            ("ffn_norm", "layer_norm2"),
        ):
            for suffix in ("weight", "bias"):
                hf[f"{target}.{hf_name}.{suffix}"] = native[f"{source}.{native_name}.{suffix}"]
        for native_name, hf_name in (
            ("wq", "q_proj"),
            ("wk", "k_proj"),
            ("wv", "v_proj"),
            ("wo", "out_proj"),
        ):
            for suffix in ("weight", "bias"):
                hf[f"{target}.self_attn.{hf_name}.{suffix}"] = native[
                    f"{source}.attn.{native_name}.{suffix}"
                ]
        for native_name, hf_name in (("w1", "fc1"), ("w2", "fc2")):
            for suffix in ("weight", "bias"):
                hf[f"{target}.mlp.{hf_name}.{suffix}"] = native[
                    f"{source}.ffn.{native_name}.{suffix}"
                ]
    return hf


def test_siglip_converter_strictly_reconstructs_native_vision_state():
    cfg = _tiny_siglip_config()
    torch.manual_seed(17)
    model = VisionTransformer(cfg, init_device="cpu")
    expected = model.state_dict()
    converted = siglip_hf_state_dict_to_vision(_native_to_hf(expected, cfg), cfg)

    assert converted.keys() == expected.keys()
    for name, tensor in converted.items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    model.load_state_dict(converted, strict=True)


def test_siglip_loader_filters_and_strips_vision_namespace(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(
        {
            "text_model.embeddings.weight": torch.ones(2, 2),
            "vision_model.embeddings.patch_embedding.bias": torch.arange(3.0),
            "vision_model.encoder.layers.0.layer_norm1.weight": torch.arange(4.0),
            "vision_model.post_layernorm.weight": torch.ones(4),
            "vision_model.head.probe": torch.ones(1, 4),
        },
        checkpoint_path,
    )

    def _download(**kwargs):
        assert kwargs["repo_id"] == "test/siglip"
        assert kwargs["revision"] == "immutable-revision"
        assert kwargs["filename"] == "model.safetensors"
        return str(checkpoint_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)
    loaded = load_siglip_hf_vision_state_dict(
        "test/siglip", revision="immutable-revision", local_files_only=True
    )

    assert set(loaded) == {
        "embeddings.patch_embedding.bias",
        "encoder.layers.0.layer_norm1.weight",
    }


def test_vision_state_fingerprint_is_deterministic_and_value_sensitive():
    state = {
        "b": torch.arange(20.0).reshape(4, 5),
        "a": torch.tensor([1.0, 2.0]),
    }
    fingerprint = vision_state_fingerprint(state, samples_per_tensor=8)
    assert fingerprint == vision_state_fingerprint(
        dict(reversed(list(state.items()))), samples_per_tensor=8
    )

    changed = {name: tensor.clone() for name, tensor in state.items()}
    changed["a"][0] += 1
    assert vision_state_fingerprint(changed, samples_per_tensor=8) != fingerprint
