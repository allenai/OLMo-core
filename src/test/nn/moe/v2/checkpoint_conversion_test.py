import torch

from olmo_core.nn.moe.v2.checkpoint_conversion import (
    BLOCK_CONFIG_CLASS,
    MODEL_CONFIG_CLASS,
    DenseLayerSpec,
    convert_legacy_config,
    convert_legacy_model_state,
    get_legacy_dense_layer_specs,
)


def _legacy_dense_block(d_model: int, hidden_size: int) -> dict:
    norm = {"name": "rms", "eps": 1e-6, "bias": False, "dtype": "float32"}
    return {
        "sequence_mixer": {"dtype": "float32"},
        "attention_norm": norm,
        "feed_forward_norm": norm,
        "feed_forward": {
            "hidden_size": hidden_size,
            "bias": False,
            "activation": "silu",
        },
        "name": "peri_norm",
    }


def _legacy_moe_block() -> dict:
    return {
        "name": "moe_fused_v2",
        "routed_experts": {"num_experts": 8},
        "shared_experts": {"num_experts": 1},
        "ep_no_sync": True,
        "ep_no_sync_use_2d_all_to_all": False,
        "ep_no_sync_use_rowwise_all_to_all": True,
        "ep_no_sync_rowwise_nblocks": 64,
        "ep_no_sync_capacity_factor": 1.5,
        "ep_no_sync_shared_slots": 2,
        "checkpoint_combined_ep_tbo": True,
    }


def test_convert_legacy_config() -> None:
    config = {
        "model": {
            "d_model": 4,
            "n_layers": 2,
            "two_batch_overlap": False,
            "block": _legacy_moe_block(),
            "block_overrides": {"0": _legacy_dense_block(4, 6)},
        },
        "train_module": {"optim": {}},
    }

    converted = convert_legacy_config(config)
    model = converted["model"]
    assert model["_CLASS_"] == MODEL_CONFIG_CLASS
    assert model["block"]["_CLASS_"] == BLOCK_CONFIG_CLASS
    assert model["block"]["ep"]["path"] == "rowwise_nvshmem"
    assert model["block"]["ep"]["rowwise_nblocks"] == 64
    assert model["block"]["ep"]["capacity_factor"] == 1.5
    assert model["block"]["ep"]["checkpoint_tbo"] is True
    assert not any(key.startswith("ep_no_sync") for key in model["block"])

    dense = model["block_overrides"]["0"]
    assert dense["_CLASS_"] == BLOCK_CONFIG_CLASS
    assert dense["use_peri_norm"] is True
    assert dense["shared_experts"]["hidden_size"] == 6
    assert dense["shared_experts"]["num_experts"] == 1
    assert "feed_forward" not in dense

    # The converter must not mutate the recorded source config.
    assert "ep_no_sync" in config["model"]["block"]
    assert "feed_forward" in config["model"]["block_overrides"]["0"]


def test_convert_all_moe_config_and_state_without_dense_layers() -> None:
    config = {
        "model": {
            "d_model": 4,
            "n_layers": 2,
            "two_batch_overlap": False,
            "block": _legacy_moe_block(),
        },
        "train_module": {"optim": {}},
    }
    state = {
        "module.blocks.0.routed_experts.w_down.main": torch.randn(9),
        "module.blocks.1.routed_experts.w_down.main": torch.randn(9),
    }

    dense_layers = get_legacy_dense_layer_specs(config)
    assert dense_layers == []

    converted_config = convert_legacy_config(config)
    converted_block = converted_config["model"]["block"]
    assert converted_block["_CLASS_"] == BLOCK_CONFIG_CLASS
    assert converted_block["ep"]["path"] == "rowwise_nvshmem"

    converted_state = convert_legacy_model_state(state, dense_layers)
    assert converted_state == state
    assert all(converted_state[key] is tensor for key, tensor in state.items())


def test_convert_dense_state_preserves_swiglu_math_and_norm_roles() -> None:
    torch.manual_seed(7)
    d_model, hidden_size = 4, 6
    prefix = "module.blocks.0"
    w1 = torch.randn(hidden_size, d_model)
    w2 = torch.randn(d_model, hidden_size)
    w3 = torch.randn(hidden_size, d_model)
    norms = {
        "attention_norm": torch.randn(d_model),
        "post_attention_norm": torch.randn(d_model),
        "feed_forward_norm": torch.randn(d_model),
        "post_feed_forward_norm": torch.randn(d_model),
    }
    untouched = torch.randn(9)
    state = {
        f"{prefix}.feed_forward.w1.weight.main": w1.flatten(),
        f"{prefix}.feed_forward.w2.weight.main": w2.flatten(),
        f"{prefix}.feed_forward.w3.weight.main": w3.flatten(),
        **{
            f"{prefix}.{name}.weight.main": tensor
            for name, tensor in norms.items()
        },
        "module.blocks.1.routed_experts.w_down.main": untouched,
    }

    converted = convert_legacy_model_state(
        state, [DenseLayerSpec(layer_idx=0, d_model=d_model, hidden_size=hidden_size)]
    )

    x = torch.randn(3, d_model)
    old_output = (torch.nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T
    up_gate = converted[f"{prefix}.shared_experts.w_up_gate.main"].view(
        d_model, 2 * hidden_size
    )
    up, gate = (x @ up_gate).split(hidden_size, dim=-1)
    w_down = converted[f"{prefix}.shared_experts.w_down.main"].view(
        1, hidden_size, d_model
    )
    new_output = (up * torch.nn.functional.silu(gate)) @ w_down[0]
    torch.testing.assert_close(new_output, old_output)

    assert converted[f"{prefix}.attention_input_norm.weight.main"] is norms[
        "attention_norm"
    ]
    assert converted[f"{prefix}.attention_norm.weight.main"] is norms[
        "post_attention_norm"
    ]
    assert converted[f"{prefix}.feed_forward_input_norm.weight.main"] is norms[
        "feed_forward_norm"
    ]
    assert converted[f"{prefix}.feed_forward_norm.weight.main"] is norms[
        "post_feed_forward_norm"
    ]
    assert converted["module.blocks.1.routed_experts.w_down.main"] is untouched
