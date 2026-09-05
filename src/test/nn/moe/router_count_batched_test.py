"""Numerical gates for packing independent layer counts, without changing LB math."""

import copy

import pytest
import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.utils import backend_supports_cuda
from olmo_core.nn.moe.emo import EmoRouterConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.ops import attach_auxiliary_loss
from olmo_core.ops.batched_router_aux import finish_batched_router_aux
from olmo_core.testing import BACKENDS, run_distributed_test


def _run_batched_parity(compiled):
    device = torch.device(f"cuda:{dist.get_rank()}" if backend_supports_cuda() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(333)
    config = MoERouterConfigV2(
        d_model=16,
        num_experts=8,
        top_k=2,
        dtype=DType.float32,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        global_load_balancing=True,
        emo=EmoRouterConfig(eos_token_id=0, min_document_expert_pool=2, max_document_expert_pool=8),
    )
    original = torch.nn.ModuleList([config.build(init_device=device) for _ in range(3)])
    candidate = copy.deepcopy(original)
    for routers in (original, candidate):
        for router in routers:
            router.set_load_balancing_process_group(dist.group.WORLD)
            assert router.load_balancing_loss is not None
            assert router.z_loss is not None
            assert router.global_batch_size_per_expert is not None
    segments = torch.arange(32, device=device).div(8, rounding_mode="floor").expand(2, -1)

    def make_forward(routers, batched):
        def forward(h):
            records, routing = [], []
            for router in routers:
                weights, indices, counts, aux = router(
                    h, False, loss_div_factor=64.0, segment_ids=segments
                )
                h = h + (weights * (indices + 1)).sum(-1, keepdim=True) * 0.01
                routing.append((indices, counts))
                if batched:
                    records.append((router, aux))
                else:
                    h = attach_auxiliary_loss(h, router.compute_aux_loss(*aux))
            if batched:
                h = finish_batched_router_aux(h, records)
            return h.square().mean(), routing

        return torch.compile(forward) if compiled else forward

    forwards = [make_forward(original, False), make_forward(candidate, True)]
    opts = [torch.optim.AdamW(r.parameters(), lr=1e-3) for r in (original, candidate)]
    for step in range(3):
        torch.manual_seed(512 + step + 10 * dist.get_rank())
        source = torch.randn(2, 32, 16, device=device)
        outputs = []
        for routers, optimizer, forward in zip((original, candidate), opts, forwards):
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(713 + step + 10 * dist.get_rank())
            x = source.clone().requires_grad_()
            loss, routing = forward(x)
            loss.backward()
            grads = []
            for param in routers.parameters():
                dist.all_reduce(param.grad)
                param.grad.div_(dist.get_world_size())
                grads.append(param.grad.clone())
            outputs.append((loss.detach(), routing, x.grad.clone(), grads))
            optimizer.step()
        ref, new = outputs
        torch.testing.assert_close(new[1], ref[1], rtol=0, atol=0)
        for field in (0, 2, 3):
            torch.testing.assert_close(new[field], ref[field], rtol=2e-5, atol=1e-7)
        for old_param, new_param in zip(original.parameters(), candidate.parameters()):
            torch.testing.assert_close(new_param, old_param, rtol=2e-5, atol=1e-7)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    opts[1].state[new_param][key],
                    opts[0].state[old_param][key],
                    rtol=2e-5,
                    atol=1e-8,
                )
        for old_router, new_router in zip(original, candidate):
            torch.testing.assert_close(
                new_router.global_batch_size_per_expert,
                old_router.global_batch_size_per_expert,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                new_router.load_balancing_loss, old_router.load_balancing_loss, rtol=2e-5, atol=1e-7
            )
            old_router.reset_metrics()
            new_router.reset_metrics()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("compiled", [False, True])
def test_batched_layer_counts_parity(backend, compiled):
    """Compare chained router gradients, unchanged counts, metrics and three Adam updates."""
    run_distributed_test(
        _run_batched_parity,
        world_size=2,
        backend=backend,
        start_method="spawn",
        func_args=(compiled,),
    )
