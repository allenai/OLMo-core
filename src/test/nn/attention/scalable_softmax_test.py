from typing import Optional

import pytest
import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.testing import FLASH_2_MARKS, FLASH_3_MARKS, FLASH_4_MARKS, requires_gpu


@pytest.mark.parametrize(
    "cu_doc_lens,expected_lengths",
    [
        pytest.param(None, [[1, 2, 3, 4], [1, 2, 3, 4]], id="absolute-prefix"),
        pytest.param(
            torch.tensor([0, 2, 4, 7, 8], dtype=torch.int32),
            [[1, 2, 1, 2], [1, 2, 3, 1]],
            id="packed-documents",
        ),
    ],
)
def test_scalable_softmax_query_scaling_and_gradients(
    cu_doc_lens: Optional[torch.Tensor], expected_lengths: list[list[int]]
) -> None:
    attention = Attention(
        d_model=8,
        n_heads=2,
        head_dim=4,
        bias=False,
        scalable_softmax=True,
    )
    assert attention.ssmax_scale is not None
    with torch.no_grad():
        attention.ssmax_scale.copy_(torch.tensor([0.5, 2.0]))

    q = torch.ones(2, 4, 2, 4, requires_grad=True)
    scaled_q = attention._apply_scalable_softmax(q, cu_doc_lens)
    expected = torch.tensor(expected_lengths, dtype=q.dtype).log()
    expected = expected[:, :, None, None] * torch.tensor([0.5, 2.0])[None, None, :, None]

    torch.testing.assert_close(scaled_q, expected.expand_as(q))
    scaled_q.sum().backward()
    assert attention.ssmax_scale.grad is not None
    assert torch.isfinite(attention.ssmax_scale.grad).all()
    assert torch.all(attention.ssmax_scale.grad > 0)
    assert q.grad is not None
    torch.testing.assert_close(q.grad, expected.expand_as(q))


def test_scalable_softmax_disabled_is_identity_and_has_no_state() -> None:
    attention = Attention(d_model=8, n_heads=2, head_dim=4, bias=False)
    q = torch.randn(2, 4, 2, 4)

    assert attention.ssmax_scale is None
    assert attention._apply_scalable_softmax(q, None) is q
    assert "ssmax_scale" not in attention.state_dict()


def test_scalable_softmax_config_param_count_and_state_round_trip() -> None:
    d_model = 16
    n_heads = 4
    base_config = AttentionConfig(n_heads=n_heads, bias=False)
    ssmax_config = AttentionConfig(n_heads=n_heads, bias=False, scalable_softmax=True)

    base_attention = base_config.build(d_model, layer_idx=0, n_layers=1)
    attention = ssmax_config.build(d_model, layer_idx=0, n_layers=1)

    assert ssmax_config.num_params(d_model) == base_config.num_params(d_model) + n_heads
    assert ssmax_config.num_params(d_model) == sum(p.numel() for p in attention.parameters())
    assert sum(p.numel() for p in attention.parameters()) == (
        sum(p.numel() for p in base_attention.parameters()) + n_heads
    )
    assert set(attention.state_dict()) - set(base_attention.state_dict()) == {"ssmax_scale"}

    assert isinstance(attention, Attention)
    assert attention.ssmax_scale is not None
    with torch.no_grad():
        attention.ssmax_scale.copy_(torch.tensor([0.25, 0.5, 1.0, 2.0]))
    restored = ssmax_config.build(d_model, layer_idx=0, n_layers=1)
    restored.load_state_dict(attention.state_dict(), strict=True)
    assert isinstance(restored, Attention)
    assert restored.ssmax_scale is not None
    torch.testing.assert_close(restored.ssmax_scale, attention.ssmax_scale)


def test_scalable_softmax_scale_is_initialized_to_one() -> None:
    attention = Attention(
        d_model=8,
        n_heads=2,
        head_dim=4,
        bias=False,
        scalable_softmax=True,
    )
    assert attention.ssmax_scale is not None
    with torch.no_grad():
        attention.ssmax_scale.fill_(float("nan"))

    attention.init_weights(
        init_method=InitMethod.normal,
        d_model=8,
        block_idx=0,
        num_blocks=1,
    )

    torch.testing.assert_close(attention.ssmax_scale, torch.ones(2))


def test_scalable_softmax_is_applied_after_head_qk_norm_and_only_to_queries() -> None:
    class ConstantNorm(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, 3.0)

    attention = Attention(
        d_model=8,
        n_heads=2,
        head_dim=4,
        bias=False,
        qk_norm=LayerNormConfig(),
        use_head_qk_norm=True,
        scalable_softmax=True,
    )
    attention.q_norm = ConstantNorm()  # type: ignore[assignment]
    attention.k_norm = ConstantNorm()  # type: ignore[assignment]
    assert attention.ssmax_scale is not None
    with torch.no_grad():
        attention.ssmax_scale.fill_(2.0)

    captured_q: Optional[torch.Tensor] = None
    captured_k: Optional[torch.Tensor] = None

    def capture_sdpa(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **_: object
    ) -> torch.Tensor:
        nonlocal captured_q, captured_k
        captured_q, captured_k = q, k
        return torch.zeros_like(q)

    attention.sdpa = capture_sdpa  # type: ignore[method-assign]
    attention(torch.randn(1, 3, 8))

    assert captured_q is not None
    assert captured_k is not None
    expected_q = 6.0 * torch.arange(1, 4, dtype=captured_q.dtype).log()
    expected_q = expected_q.view(1, 3, 1, 1).expand_as(captured_q)
    torch.testing.assert_close(captured_q, expected_q)
    torch.testing.assert_close(captured_k, torch.full_like(captured_k, 3.0))


def test_scalable_softmax_absolute_lengths_do_not_follow_masks_or_position_ids() -> None:
    attention = Attention(
        d_model=8,
        n_heads=2,
        head_dim=4,
        bias=False,
        scalable_softmax=True,
    )
    assert attention.ssmax_scale is not None
    with torch.no_grad():
        attention.ssmax_scale.fill_(1.0)

    captured_q: Optional[torch.Tensor] = None

    def capture_sdpa(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **_: object
    ) -> torch.Tensor:
        nonlocal captured_q
        captured_q = q
        return torch.zeros_like(q)

    attention.sdpa = capture_sdpa  # type: ignore[method-assign]
    x = torch.randn(1, 4, 8)
    # These deliberately permit non-causal image attention and use non-monotonic RoPE positions.
    # Scalable-Softmax still follows the source checkpoint's absolute [1, 2, 3, 4] convention.
    attention(
        x,
        or_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool),
        position_ids=torch.tensor([[9, 9, 2, 2]]),
    )

    assert captured_q is not None
    raw_q = attention.w_q(x).view(1, 4, 2, 4)
    expected_scale = torch.arange(1, 5, dtype=raw_q.dtype).log().view(1, 4, 1, 1)
    torch.testing.assert_close(captured_q, raw_q * expected_scale)


@pytest.mark.parametrize("backend", [AttentionBackendName.torch, AttentionBackendName.flex])
def test_scalable_softmax_current_cpu_backends_match_torch(backend: AttentionBackendName) -> None:
    torch.manual_seed(17)
    kwargs = dict(
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        bias=False,
        scalable_softmax=True,
    )
    reference = Attention(backend=AttentionBackendName.torch, **kwargs)
    candidate = Attention(backend=backend, **kwargs)
    candidate.load_state_dict(reference.state_dict(), strict=True)

    x = torch.randn(1, 8, 16)
    is_image = torch.zeros(1, 8, dtype=torch.bool)
    is_image[:, 2:6] = True
    or_mask = (is_image[:, :, None] & is_image[:, None, :]).unsqueeze(1)
    # FlexAttention has no CPU backward implementation; this test exercises forward semantics.
    with torch.no_grad():
        ref = reference(x, or_mask=or_mask)
        actual = candidate(x, or_mask=or_mask)

    torch.testing.assert_close(actual, ref, atol=1e-4, rtol=1e-4)


@requires_gpu
def test_scalable_softmax_flex_cuda_forward_and_backward_match_torch() -> None:
    torch.manual_seed(19)
    kwargs = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        bias=False,
        scalable_softmax=True,
        init_device="cuda",
        dtype=torch.float32,
    )
    reference = Attention(backend=AttentionBackendName.torch, **kwargs)
    candidate = Attention(backend=AttentionBackendName.flex, **kwargs)
    candidate.load_state_dict(reference.state_dict(), strict=True)

    x_ref = torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_actual = x_ref.detach().clone().requires_grad_(True)
    is_image = torch.zeros(1, 16, dtype=torch.bool, device="cuda")
    is_image[:, 3:11] = True
    or_mask = (is_image[:, :, None] & is_image[:, None, :]).unsqueeze(1)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = reference(x_ref, or_mask=or_mask)
        actual = candidate(x_actual, or_mask=or_mask)
    torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)

    ref.float().square().mean().backward()
    actual.float().square().mean().backward()
    assert reference.ssmax_scale is not None and candidate.ssmax_scale is not None
    assert reference.ssmax_scale.grad is not None and candidate.ssmax_scale.grad is not None
    assert x_ref.grad is not None and x_actual.grad is not None
    assert torch.isfinite(candidate.ssmax_scale.grad).all()
    assert torch.isfinite(x_actual.grad).all()
    torch.testing.assert_close(
        candidate.ssmax_scale.grad, reference.ssmax_scale.grad, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(x_actual.grad, x_ref.grad, atol=2e-2, rtol=2e-2)


@requires_gpu
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(AttentionBackendName.flash_2, marks=FLASH_2_MARKS),
        pytest.param(AttentionBackendName.flash_3, marks=FLASH_3_MARKS),
        pytest.param(AttentionBackendName.flash_4, marks=FLASH_4_MARKS),
    ],
)
def test_scalable_softmax_flash_backends_match_torch(backend: AttentionBackendName) -> None:
    torch.manual_seed(23)
    kwargs = dict(
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        bias=False,
        scalable_softmax=True,
        init_device="cuda",
        dtype=torch.float32,
    )
    reference = Attention(backend=AttentionBackendName.torch, **kwargs)
    candidate = Attention(backend=backend, **kwargs)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = reference(x)
        actual = candidate(x)

    torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)
    actual.float().sum().backward()
    assert candidate.ssmax_scale is not None
    assert candidate.ssmax_scale.grad is not None
    assert torch.isfinite(candidate.ssmax_scale.grad).all()


@pytest.mark.parametrize("unsupported_mode", ["kv-cache", "context-parallel"])
def test_scalable_softmax_rejects_unsupported_runtime_modes(unsupported_mode: str) -> None:
    attention = Attention(
        d_model=8,
        n_heads=2,
        head_dim=4,
        bias=False,
        scalable_softmax=True,
    )
    if unsupported_mode == "kv-cache":
        attention.kv_cache_manager = nn.Identity()  # type: ignore[assignment]
        match = "KV caching"
    else:
        attention.backend.cp_enabled = True
        match = "context parallelism"

    with pytest.raises(NotImplementedError, match=match):
        attention._apply_scalable_softmax(torch.ones(1, 2, 2, 4), None)


def test_scalable_softmax_rejects_sliding_window_attention() -> None:
    config = AttentionConfig(
        n_heads=2,
        scalable_softmax=True,
        sliding_window=SlidingWindowAttentionConfig(
            pattern=[128],
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=False,
        ),
    )

    with pytest.raises(OLMoConfigurationError, match="scalable_softmax"):
        config.build(d_model=8, layer_idx=0, n_layers=2)
    with pytest.raises(OLMoConfigurationError, match="scalable_softmax"):
        Attention(
            d_model=8,
            n_heads=2,
            scalable_softmax=True,
            window_size=128,
        )


def test_scalable_softmax_allows_global_layer_in_sliding_window_config() -> None:
    config = AttentionConfig(
        n_heads=2,
        scalable_softmax=True,
        sliding_window=SlidingWindowAttentionConfig(
            pattern=[128, -1],
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=False,
        ),
    )

    attention = config.build(d_model=8, layer_idx=1, n_layers=2)
    assert isinstance(attention, Attention)
    assert attention.window_size is None
    assert attention.ssmax_scale is not None


@pytest.mark.parametrize(
    "attention_type",
    [AttentionType.fused, AttentionType.fused_v2, AttentionType.normalized],
)
def test_scalable_softmax_rejects_other_attention_implementations(
    attention_type: AttentionType,
) -> None:
    config = AttentionConfig(
        name=attention_type,
        n_heads=2,
        scalable_softmax=True,
        backend=AttentionBackendName.torch,
    )

    with pytest.raises(OLMoConfigurationError, match="only supported by default attention"):
        config.build(d_model=8, layer_idx=0, n_layers=1)


@pytest.mark.parametrize("attention_type", [AttentionType.fused_v2, AttentionType.normalized])
def test_disabled_scalable_softmax_preserves_other_attention_implementations(
    attention_type: AttentionType,
) -> None:
    config = AttentionConfig(
        name=attention_type,
        n_heads=2,
        scalable_softmax=False,
        backend=AttentionBackendName.torch,
    )

    attention = config.build(d_model=8, layer_idx=0, n_layers=1)
    assert "ssmax_scale" not in attention.state_dict()
    assert attention(torch.randn(1, 4, 8)).shape == (1, 4, 8)
