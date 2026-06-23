import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from olmo_core.config import DType
from olmo_core.distributed.utils import get_world_size
from olmo_core.nn.moe.router import MoERouterGatingFunction
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.testing import BACKENDS, run_distributed_test


def _build(*, top_k=2, num_experts=8, d_model=16, **kwargs):
    return MoERouterConfigV2(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        dtype=DType.float32,
        **kwargs,
    ).build(init_device="cpu")


def test_router_config_num_params():
    cfg = MoERouterConfigV2(d_model=16, num_experts=8, top_k=2)
    # The router weight is (num_experts * d_model).
    assert cfg.num_params() == 16 * 8


def test_router_forward_shapes_and_invariants():
    torch.manual_seed(0)
    B, S, D, E, K = 2, 4, 16, 8, 2
    router = _build(top_k=K, num_experts=E, d_model=D)

    x = torch.randn(B, S, D)
    weights, indices, batch_size_per_expert, aux = router(x, False)

    assert weights.shape == (B, S, K)
    assert indices.shape == (B, S, K)
    assert batch_size_per_expert.shape == (E,)
    # Every selected expert index is valid.
    assert int(indices.min()) >= 0 and int(indices.max()) < E
    # Each of B*S tokens is routed to exactly top_k experts.
    assert int(batch_size_per_expert.sum()) == B * S * K
    # Forward returns the auxiliary-loss inputs (not the reduced losses).
    assert aux is not None


def test_router_scores_only_short_circuits():
    B, S, D, E = 2, 4, 16, 8
    router = _build(num_experts=E, d_model=D)

    scores, indices, batch_size_per_expert, aux = router(torch.randn(B, S, D), True)

    assert scores.shape == (B, S, E)
    assert indices is None and batch_size_per_expert is None and aux is None


def test_router_top1_and_sigmoid_gating():
    router = _build(top_k=1, gating_function=MoERouterGatingFunction.sigmoid)

    weights, indices, _, _ = router(torch.randn(2, 4, 16), False)

    assert weights.shape == (2, 4, 1)
    assert indices.shape == (2, 4, 1)


def test_router_uniform_expert_assignment_balances_experts():
    B, S, D, E, K = 2, 8, 16, 4, 2
    router = _build(top_k=K, num_experts=E, d_model=D, uniform_expert_assignment=True)

    _, indices, batch_size_per_expert, _ = router(torch.randn(B, S, D), False)

    # Uniform assignment spreads the B*S*K routing slots evenly across experts.
    assert int(indices.min()) >= 0 and int(indices.max()) < E
    expected = B * S * K // E
    torch.testing.assert_close(
        batch_size_per_expert,
        torch.full_like(batch_size_per_expert, expected),
    )


def test_router_random_expert_assignment_runs():
    torch.manual_seed(0)
    B, S, D, E, K = 2, 4, 16, 8, 2
    router = _build(top_k=K, num_experts=E, d_model=D, random_expert_assignment=True)

    _, indices, batch_size_per_expert, _ = router(torch.randn(B, S, D), False)

    assert indices.shape == (B, S, K)
    assert int(indices.min()) >= 0 and int(indices.max()) < E
    assert int(batch_size_per_expert.sum()) == B * S * K


def test_router_restore_weight_scale_multiplies_by_top_k():
    torch.manual_seed(0)
    router = _build(top_k=4)
    x = torch.randn(2, 4, 16)

    base_weights, _, _, _ = router(x, False)
    router.restore_weight_scale = True
    scaled_weights, _, _, _ = router(x, False)

    torch.testing.assert_close(scaled_weights, base_weights * router.top_k)


def test_router_original_top_k_rescales_weights():
    torch.manual_seed(0)
    router = _build(top_k=2, original_top_k=8)
    x = torch.randn(2, 4, 16)

    scaled_weights, _, _, _ = router(x, False)
    router.original_top_k = None
    base_weights, _, _, _ = router(x, False)

    torch.testing.assert_close(scaled_weights, base_weights * (8 / 2) ** 0.5)


def test_router_normalize_expert_weights_unit_norm():
    router = _build(num_experts=8, normalize_expert_weights=2.0)

    # The scores-only path normalizes the score vector to unit Lp norm per token.
    scores, _, _, _ = router(torch.randn(2, 4, 16), True)

    norms = scores.norm(p=2.0, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_router_bias_gamma_creates_buffer_and_biases_routing():
    router = _build(top_k=1, num_experts=4, bias_gamma=0.01)
    assert router.score_bias is not None
    assert tuple(router.score_bias.shape) == (4,)

    # Strongly bias expert 3; with top_k=1 every token must select it (selection uses
    # scores + score_bias, while the returned weights still come from the raw scores).
    with torch.no_grad():
        router.score_bias.copy_(torch.tensor([-10.0, -10.0, -10.0, 10.0]))
    _, indices, _, _ = router(torch.randn(2, 4, 16), False)
    assert torch.equal(indices, torch.full_like(indices, 3))

    # No bias buffer when bias_gamma is unset.
    assert _build(bias_gamma=None).score_bias is None


def test_router_use_quant_scores_matches_plain_for_separated_scores():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16)
    router = _build(top_k=2, use_quant_scores=True)

    weights, indices, _, _ = router(x, False)
    assert weights.shape == (2, 4, 2) and indices.shape == (2, 4, 2)

    # Quantizing well-separated softmax scores (q=2**14) does not change which experts
    # are selected; weights come from the original (un-quantized) scores either way.
    router.use_quant_scores = False
    _, indices_ref, _, _ = router(x, False)
    assert torch.equal(indices, indices_ref)


def test_router_grad_flows_to_weight_and_input():
    router = _build(top_k=2)
    x = torch.randn(2, 4, 16, requires_grad=True)

    weights, _, _, _ = router(x, False)
    weights.sum().backward()

    assert router.weight.grad is not None
    assert x.grad is not None


# NOTE: ``use_recompute_fp32_cast`` is exercised end-to-end on GPU. It routes the fp32 cast
# through OutputDiscardCheckpoint, whose storage sharing relies on a C++ extension (covered
# by the OutputDiscardCheckpoint test suite); the Python fallback cannot recompute through
# autograd on a plain CPU host, so it is not unit-tested at the router level here.


def test_router_compute_metrics_reports_scaled_losses():
    router = _build(top_k=2, num_experts=4, lb_loss_weight=0.1, z_loss_weight=0.01)

    # The losses are populated by the consumer; set them directly to test the metric surface.
    router.load_balancing_loss = torch.tensor(2.0)
    router.z_loss = torch.tensor(3.0)
    router.batch_size_per_expert = torch.tensor([1.0, 1.0, 1.0, 1.0])

    metrics = router.compute_metrics(reset=False)

    assert "load imbalance" in metrics
    torch.testing.assert_close(metrics["load balancing loss"][0], torch.tensor(0.2))
    torch.testing.assert_close(metrics["load balancing loss unscaled"][0], torch.tensor(2.0))
    torch.testing.assert_close(metrics["router Z loss"][0], torch.tensor(0.03))


def _run_router_tp(device: torch.device):
    tp_mesh = dist.init_device_mesh(device.type, (get_world_size(),), mesh_dim_names=("tp",))

    router = MoERouterConfigV2(d_model=16, num_experts=4, top_k=2, dtype=DType.float32).build(
        init_device=device.type
    )
    router.apply_tp(tp_mesh)

    # apply_tp replicates the router weight across the TP mesh.
    assert isinstance(router.weight, DTensor)
    assert router.weight.placements == (Replicate(),)

    # PrepareModuleInput shards the input on the sequence dim; forward sees the local shard.
    B, S, D, K = 2, 4 * get_world_size(), 16, 2
    x = torch.randn(B, S, D, device=device)
    local_x = distribute_tensor(x, tp_mesh, [Shard(1)]).to_local()

    # scores_only must be passed by keyword: PrepareModuleInput maps positional inputs to
    # input_layouts, so the lone sharded positional arg is the activation tensor.
    weights, indices, _, _ = router(local_x, scores_only=False)

    assert weights.shape == (B, S // get_world_size(), K)
    assert indices.shape == (B, S // get_world_size(), K)


@pytest.mark.parametrize("backend", BACKENDS)
def test_router_tp_replicates_weight_and_runs(backend: str):
    device = torch.device("cuda") if "nccl" in backend else torch.device("cpu")
    run_distributed_test(
        _run_router_tp,
        world_size=2,
        backend=backend,
        func_args=(device,),
    )
