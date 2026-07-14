"""Independent exact-value checks for legacy-to-OLMoDDP checkpoint conversion."""

from __future__ import annotations

from typing import Iterable, Mapping, Protocol

import torch


class DenseLayerLike(Protocol):
    layer_idx: int
    d_model: int
    hidden_size: int


def _take(state: dict[str, torch.Tensor], key: str, expected_numel: int) -> torch.Tensor:
    try:
        tensor = state.pop(key)
    except KeyError as exc:
        raise ValueError(f"Missing tensor {key!r}") from exc
    if tensor.numel() != expected_numel:
        raise ValueError(f"{key}: expected {expected_numel:,} elements, got {tensor.numel():,}")
    return tensor


def _require_bitwise_equal(expected: torch.Tensor, actual: torch.Tensor, *, key: str) -> None:
    if expected.shape != actual.shape:
        raise ValueError(
            f"{key}: shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
        )
    if expected.dtype != actual.dtype:
        raise ValueError(f"{key}: dtype mismatch: expected {expected.dtype}, got {actual.dtype}")
    expected_bytes = expected.contiguous().view(torch.uint8)
    actual_bytes = actual.contiguous().view(torch.uint8)
    if not torch.equal(expected_bytes, actual_bytes):
        unequal = expected_bytes != actual_bytes
        flat_unequal = unequal.reshape(-1)
        first_byte = -1
        chunk_size = 1024 * 1024
        for start in range(0, flat_unequal.numel(), chunk_size):
            indices = torch.nonzero(flat_unequal[start : start + chunk_size], as_tuple=False)
            if indices.numel():
                first_byte = start + int(indices[0].item())
                break
        assert first_byte >= 0
        raise ValueError(
            f"{key}: values are not bitwise equal; first differing byte={first_byte}, "
            f"differing_bytes={int(unequal.sum().item()):,}"
        )


def verify_converted_state_exact(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    dense_layers: Iterable[DenseLayerLike],
) -> dict[str, int | str | bool]:
    """Verify all values without calling the converter's state mapping."""

    source = dict(source_state)
    target = dict(target_state)
    transformed_target_count = 0
    transformed_target_numel = 0

    norm_mapping = {
        "attention_norm": "attention_input_norm",
        "post_attention_norm": "attention_norm",
        "feed_forward_norm": "feed_forward_input_norm",
        "post_feed_forward_norm": "feed_forward_norm",
    }

    for spec in dense_layers:
        prefix = f"module.blocks.{spec.layer_idx}"
        d_model = spec.d_model
        hidden_size = spec.hidden_size

        w1 = _take(
            source,
            f"{prefix}.feed_forward.w1.weight.main",
            hidden_size * d_model,
        ).reshape(hidden_size, d_model)
        w2 = _take(
            source,
            f"{prefix}.feed_forward.w2.weight.main",
            d_model * hidden_size,
        ).reshape(d_model, hidden_size)
        w3 = _take(
            source,
            f"{prefix}.feed_forward.w3.weight.main",
            hidden_size * d_model,
        ).reshape(hidden_size, d_model)

        # Re-derive the layout here instead of importing conversion code.
        expected_up_gate = torch.cat((w3.transpose(0, 1), w1.transpose(0, 1)), dim=1)
        expected_up_gate = expected_up_gate.contiguous().reshape(-1)
        expected_down = w2.transpose(0, 1).contiguous().unsqueeze(0).reshape(-1)
        for name, expected in (
            ("shared_experts.w_up_gate.main", expected_up_gate),
            ("shared_experts.w_down.main", expected_down),
        ):
            key = f"{prefix}.{name}"
            actual = _take(target, key, expected.numel())
            _require_bitwise_equal(expected, actual, key=key)
            transformed_target_count += 1
            transformed_target_numel += expected.numel()

        for source_name, target_name in norm_mapping.items():
            source_key = f"{prefix}.{source_name}.weight.main"
            target_key = f"{prefix}.{target_name}.weight.main"
            expected = _take(source, source_key, d_model)
            actual = _take(target, target_key, d_model)
            _require_bitwise_equal(expected, actual, key=target_key)
            transformed_target_count += 1
            transformed_target_numel += expected.numel()

    source_keys = set(source)
    target_keys = set(target)
    if source_keys != target_keys:
        raise ValueError(
            "Unchanged tensor key mismatch: "
            f"missing_from_target={sorted(source_keys - target_keys)[:20]}, "
            f"unexpected_in_target={sorted(target_keys - source_keys)[:20]}"
        )

    unchanged_numel = 0
    for key in sorted(source):
        _require_bitwise_equal(source[key], target[key], key=key)
        unchanged_numel += source[key].numel()

    return {
        "status": "STRICT_TENSOR_MATCH",
        "bitwise_equal": True,
        "source_tensor_count": len(source_state),
        "target_tensor_count": len(target_state),
        "source_numel": sum(tensor.numel() for tensor in source_state.values()),
        "target_numel": sum(tensor.numel() for tensor in target_state.values()),
        "transformed_target_tensor_count": transformed_target_count,
        "transformed_target_numel": transformed_target_numel,
        "unchanged_tensor_count": len(source),
        "unchanged_numel": unchanged_numel,
    }
