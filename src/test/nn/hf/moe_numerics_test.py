import pytest
import torch

from olmo_core.testing.utils import requires_gpu, requires_te, requires_triton


def test_core_combine_rounds_fused_updates_in_routing_order():
    from olmo_core.nn.moe.v2.hf.moe_numerics import core_combine

    torch.manual_seed(73)
    values = torch.randn(5, 16, 32).bfloat16()
    probabilities = torch.rand(5, 16)
    # FP64 supplies an independent arithmetic oracle before each BF16 rounding.
    expected = torch.zeros(5, 32, dtype=torch.bfloat16)
    for slot in range(16):
        expected = (
            expected.double()
            + values[:, slot].double() * probabilities[:, slot, None].bfloat16().double()
        ).bfloat16()
    actual = core_combine(values, probabilities)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    fp32_sum = (values.float() * probabilities[..., None]).sum(1).bfloat16()
    assert not torch.equal(actual, fp32_sum)


@requires_gpu
@requires_te
@requires_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_hf_core_experts_match_actual_core_and_te(dtype):
    from olmo_core.config import DType
    from olmo_core.nn.moe.utils import moe_permute_no_compile, moe_unpermute_no_compile
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
        Olmo3MoeCoreExperts,
        Olmo3MoeExpert,
    )
    from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig

    torch.manual_seed(83)
    N, D, inner_dim, E, K = 7, 32, 64, 8, 3
    core = RoutedExpertsConfig(
        d_model=D,
        hidden_size=inner_dim,
        num_experts=E,
        bias=False,
        dtype=DType.bfloat16 if dtype == torch.bfloat16 else DType.float32,
    ).build(init_device="cuda")
    hf = Olmo3MoeCoreExperts(Olmo3MoeExpert(D, inner_dim, "silu") for _ in range(E)).to(
        device="cuda", dtype=dtype
    )
    with torch.no_grad():
        for param in core.parameters():
            param.normal_(std=0.1)
        for index, expert in enumerate(hf):
            expert.up_proj.weight.copy_(core.w_up_gate[index, :inner_dim])
            expert.gate_proj.weight.copy_(core.w_up_gate[index, inner_dim:])
            expert.down_proj.weight.copy_(core.w_down[index].t())
        x = torch.randn(N, D, device="cuda", dtype=dtype)
        weights, indices = torch.randn(N, E, device="cuda").softmax(-1).topk(K)
        counts = torch.bincount(indices.flatten(), minlength=E).to(torch.int32)
        permuted, row_map = moe_permute_no_compile(
            inp=x, routing_map=indices.int(), num_out_tokens=N * K, map_type="index"
        )
        expected = moe_unpermute_no_compile(
            inp=core(permuted, counts),
            row_id_map=row_map,
            restore_shape=x.shape,
            merging_probs=weights,
            map_type="index",
        )
        actual = hf._forward_grouped_mm(x, indices, weights)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_core_numerics_is_serialized_and_old_configs_keep_original_mode():
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
        Olmo3MoeCoreExperts,
        Olmo3MoeSparseMLP,
    )

    assert not Olmo3MoeConfig().moe_use_core_numerics
    config = Olmo3MoeConfig(
        hidden_size=32,
        moe_intermediate_size=64,
        n_routed_experts=4,
        shared_expert_intermediate_size=None,
        moe_use_core_numerics=True,
    )
    config = Olmo3MoeConfig.from_dict(config.to_dict())
    assert isinstance(Olmo3MoeSparseMLP(config).experts, Olmo3MoeCoreExperts)
    with pytest.raises(ValueError, match="requires hidden_act"):
        Olmo3MoeConfig(moe_use_core_numerics=True, hidden_act="relu")
