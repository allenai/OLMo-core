"""Correctness gates for optional early no-EP global load-balancing reductions."""

import copy

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from olmo_core.config import DType
from olmo_core.distributed.utils import backend_supports_cuda
from olmo_core.nn.moe.emo import EmoRouterConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.testing import BACKENDS, run_distributed_test


def _run_count_overlap_parity(compiled: bool, emo: bool):
    device = torch.device(f"cuda:{dist.get_rank()}" if backend_supports_cuda() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(315)
    config = MoERouterConfigV2(
        d_model=16,
        num_experts=8,
        top_k=2,
        dtype=DType.float32,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        global_load_balancing=True,
        emo=(
            EmoRouterConfig(eos_token_id=0, min_document_expert_pool=2, max_document_expert_pool=8)
            if emo
            else None
        ),
    )
    reference = config.build(init_device=device)
    with torch.no_grad():
        reference.weight.normal_(0, 0.1)
    candidate = copy.deepcopy(reference)
    for router in (reference, candidate):
        router.set_load_balancing_process_group(dist.group.WORLD)
        # Allocate metric storage before Dynamo capture, as production setup does.
        assert router.load_balancing_loss is not None
        assert router.z_loss is not None
        assert router.global_batch_size_per_expert is not None
    optimizers = [torch.optim.AdamW(r.parameters(), lr=1e-3) for r in (reference, candidate)]
    segment_ids = torch.arange(32, device=device).div(8, rounding_mode="floor").expand(2, -1)

    def make_forward(router, overlap):
        def forward(x):
            weights, indices, counts, aux = router(
                x,
                False,
                loss_div_factor=64.0,
                **({"segment_ids": segment_ids} if emo else {}),
            )
            assert aux is not None and counts is not None
            pending = router.start_global_count_reduce(counts) if overlap else None
            # Independent work between initiation and consumption, like expert compute.
            independent = x.square().mean()
            loss = router.compute_aux_loss(*aux, pending, accumulate_metrics=True)
            assert loss is not None
            return loss + weights.square().mean() + independent, indices, counts

        return torch.compile(forward) if compiled else forward

    forwards = [make_forward(reference, False), make_forward(candidate, True)]
    for step in range(3):
        torch.manual_seed(510 + step + 10 * dist.get_rank())
        source = torch.randn(2, 32, 16, device=device)
        results = []
        for router, optimizer, forward in zip((reference, candidate), optimizers, forwards):
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(700 + step + 10 * dist.get_rank())
            x = source.clone().requires_grad_()
            loss, indices, counts = forward(x)
            original_counts = counts.clone()
            loss.backward()
            grads = []
            for parameter in router.parameters():
                assert parameter.grad is not None
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(dist.get_world_size())
                grads.append(parameter.grad.clone())
            assert x.grad is not None
            optimizer.step()
            torch.testing.assert_close(counts, original_counts, rtol=0, atol=0)
            results.append((loss.detach(), indices, counts, x.grad.clone(), grads))

        expected, actual = results
        for index in (1, 2):
            torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
        for index in (0, 3):
            torch.testing.assert_close(actual[index], expected[index], rtol=2e-5, atol=1e-7)
        for actual_grad, expected_grad in zip(actual[4], expected[4]):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=1e-7)
        for ref_parameter, new_parameter in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(new_parameter, ref_parameter, rtol=2e-5, atol=1e-7)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizers[1].state[new_parameter][key],
                    optimizers[0].state[ref_parameter][key],
                    rtol=2e-5,
                    atol=1e-8,
                )
        torch.testing.assert_close(
            candidate.global_batch_size_per_expert,
            reference.global_batch_size_per_expert,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            candidate.load_balancing_loss, reference.load_balancing_loss, rtol=2e-5, atol=1e-7
        )
        for router in (reference, candidate):
            router.reset_metrics()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("emo", [False, True])
def test_early_global_count_reduction_parity(backend: str, compiled: bool, emo: bool):
    """Check local routing, global counts, loss, gradients and Adam state over three updates."""
    run_distributed_test(
        _run_count_overlap_parity,
        world_size=2,
        backend=backend,
        start_method="spawn",
        func_kwargs={"compiled": compiled, "emo": emo},
    )


def _run_explicit_wait_graph():
    router = MoERouterConfigV2(
        d_model=16, num_experts=8, top_k=2, global_load_balancing=True
    ).build(init_device="cpu")
    router.set_load_balancing_process_group(dist.group.WORLD)
    graphs = []

    def forward(counts, matrix):
        pending = router.start_global_count_reduce(counts)
        independent = matrix @ matrix
        reduced = funcol.wait_tensor(pending)
        return reduced, independent

    def capture(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    counts = torch.arange(8) + dist.get_rank()
    matrix = torch.eye(16)
    reduced, independent = torch.compile(forward, backend=capture, fullgraph=True)(counts, matrix)
    torch.testing.assert_close(reduced, 2 * torch.arange(8).float() + 1, rtol=0, atol=0)
    torch.testing.assert_close(independent, matrix, rtol=0, atol=0)
    assert len(graphs) == 1
    targets = [str(node.target) for node in graphs[0].graph.nodes if node.op == "call_function"]
    launches = [i for i, name in enumerate(targets) if "all_reduce" in name]
    waits = [i for i, name in enumerate(targets) if "wait_tensor" in name]
    compute = [i for i, name in enumerate(targets) if "matmul" in name]
    assert len(launches) == len(waits) == len(compute) == 1, targets
    assert launches[0] < compute[0] < waits[0], targets
    print("EXPLICIT_WAIT_CAPTURE", graphs[0].code, flush=True)


def test_captured_count_launch_does_not_insert_an_early_wait():
    """Guard FX launch/compute/wait order; runtime overlap still needs a GPU trace."""
    run_distributed_test(
        _run_explicit_wait_graph, world_size=2, backend="gloo", start_method="spawn"
    )
