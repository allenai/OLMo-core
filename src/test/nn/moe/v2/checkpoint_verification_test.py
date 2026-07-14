import pytest
import torch

from olmo_core.nn.moe.v2.checkpoint_conversion import DenseLayerSpec
from olmo_core.nn.moe.v2.checkpoint_verification import verify_converted_state_exact


def _states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], DenseLayerSpec]:
    d_model, hidden_size = 3, 5
    prefix = "module.blocks.0"
    w1 = torch.arange(hidden_size * d_model, dtype=torch.float32)
    w2 = torch.arange(d_model * hidden_size, dtype=torch.float32) + 100
    w3 = torch.arange(hidden_size * d_model, dtype=torch.float32) + 200
    source = {
        f"{prefix}.feed_forward.w1.weight.main": w1,
        f"{prefix}.feed_forward.w2.weight.main": w2,
        f"{prefix}.feed_forward.w3.weight.main": w3,
        f"{prefix}.attention_norm.weight.main": torch.arange(d_model, dtype=torch.float32),
        f"{prefix}.post_attention_norm.weight.main": torch.arange(d_model, dtype=torch.float32)
        + 10,
        f"{prefix}.feed_forward_norm.weight.main": torch.arange(d_model, dtype=torch.float32) + 20,
        f"{prefix}.post_feed_forward_norm.weight.main": torch.arange(d_model, dtype=torch.float32)
        + 30,
        "module.embeddings.weight.main": torch.arange(7, dtype=torch.float32),
    }
    target = {
        f"{prefix}.shared_experts.w_up_gate.main": torch.cat(
            (w3.view(hidden_size, d_model).T, w1.view(hidden_size, d_model).T), dim=1
        )
        .contiguous()
        .view(-1),
        f"{prefix}.shared_experts.w_down.main": w2.view(d_model, hidden_size)
        .T.contiguous()
        .view(-1),
        f"{prefix}.attention_input_norm.weight.main": source[
            f"{prefix}.attention_norm.weight.main"
        ].clone(),
        f"{prefix}.attention_norm.weight.main": source[
            f"{prefix}.post_attention_norm.weight.main"
        ].clone(),
        f"{prefix}.feed_forward_input_norm.weight.main": source[
            f"{prefix}.feed_forward_norm.weight.main"
        ].clone(),
        f"{prefix}.feed_forward_norm.weight.main": source[
            f"{prefix}.post_feed_forward_norm.weight.main"
        ].clone(),
        "module.embeddings.weight.main": source["module.embeddings.weight.main"].clone(),
    }
    return source, target, DenseLayerSpec(0, d_model, hidden_size)


def test_strict_verifier_accepts_exact_independent_mapping() -> None:
    source, target, spec = _states()
    report = verify_converted_state_exact(source, target, [spec])
    assert report["status"] == "STRICT_TENSOR_MATCH"
    assert report["bitwise_equal"] is True
    assert report["unchanged_tensor_count"] == 1


def test_strict_verifier_accepts_bitwise_unchanged_all_moe_state() -> None:
    source = {
        "module.blocks.0.routed_experts.w_down.main": torch.tensor(
            [0.0, -0.0, 1.0], dtype=torch.float32
        )
    }
    target = {key: tensor.clone() for key, tensor in source.items()}

    report = verify_converted_state_exact(source, target, [])

    assert report["status"] == "STRICT_TENSOR_MATCH"
    assert report["bitwise_equal"] is True
    assert report["transformed_target_tensor_count"] == 0
    assert report["unchanged_tensor_count"] == 1


@pytest.mark.parametrize(
    "key",
    [
        "module.embeddings.weight.main",
        "module.blocks.0.shared_experts.w_up_gate.main",
        "module.blocks.0.attention_input_norm.weight.main",
    ],
)
def test_strict_verifier_rejects_one_value_corruption(key: str) -> None:
    source, target, spec = _states()
    target[key][0] += 1
    with pytest.raises(ValueError, match="not bitwise equal"):
        verify_converted_state_exact(source, target, [spec])


def test_strict_verifier_distinguishes_signed_zero_bits() -> None:
    source, target, spec = _states()
    source["module.embeddings.weight.main"][0] = 0.0
    target["module.embeddings.weight.main"][0] = -0.0
    with pytest.raises(ValueError, match="not bitwise equal"):
        verify_converted_state_exact(source, target, [spec])
