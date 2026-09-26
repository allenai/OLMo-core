"""Mask exactness and compiled EMO router numerical qualification."""

import json
import os
from pathlib import Path

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.moe.emo import EmoRouterConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.ops.moe import doc_sum_scatter, pool_keep_mask_inverse_scatter

document_pool_keep_mask = pytest.importorskip(
    "olmo_core.ops.emo_document_pool"
).document_pool_keep_mask


@pytest.mark.gpu
@pytest.mark.parametrize("length", [65535, 65536])
def test_document_masks_at_cuda_grid_boundary(length):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(918)
    boundaries = torch.rand(2, length, device="cuda") < 0.01
    boundaries[:, 0] = False
    boundaries[1, 1:] = True  # Include the last live document at the grid boundary.
    segments = boundaries.long().cumsum(1)
    pools = torch.randint(16, 513, (2, length), device="cuda").gather(1, segments)
    # Exact integer scatter sums, with ties, isolate launch/mask semantics from
    # floating point scatter order. This checks all tokens in both batch rows.
    scores = torch.randint(-2, 3, (2, length, 512), device="cuda").float()
    reference = torch.compile(
        lambda x, s, p: pool_keep_mask_inverse_scatter(doc_sum_scatter(x, s), p),
        fullgraph=True,
    )
    candidate = torch.compile(document_pool_keep_mask, fullgraph=True)
    torch.testing.assert_close(
        reference(scores, segments, pools), candidate(scores, segments, pools), rtol=0, atol=0
    )


@pytest.mark.gpu
def test_mixed_document_masks():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(81)
    for length in (257, 8192):
        # Ragged documents, consecutive boundaries, one long and one all-singleton row.
        boundaries = torch.rand(4, length, device="cuda") < 0.08
        boundaries[:, 0] = False
        boundaries[0] = False
        boundaries[1, 1:] = True
        segments = boundaries.long().cumsum(1)
        pools = torch.randint(16, 513, (4, length), device="cuda").gather(1, segments)
        for tied in (False, True):
            scores = (
                torch.randint(-2, 3, (4, length, 512), device="cuda").float()
                if tied
                # Multiples of 2^-10 have exact FP32 sums here: at most 8192
                # terms with integer numerators <= 1023 stay below 2^24.
                # Unrestricted floats make even reference/reference masks
                # sensitive to atomic scatter order near the pool cutoff.
                else torch.randint(0, 1024, (4, length, 512), device="cuda").float() / 1024
            )
            old = torch.compile(
                lambda x, s, p: pool_keep_mask_inverse_scatter(doc_sum_scatter(x, s), p),
                fullgraph=True,
            )
            new = torch.compile(document_pool_keep_mask, fullgraph=True)
            expected = old(scores, segments, pools)
            torch.testing.assert_close(expected, old(scores, segments, pools), rtol=0, atol=0)
            torch.testing.assert_close(expected, new(scores, segments, pools), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("track_trajectory", (False, True))
def test_compiled_router_gradients_and_adam(monkeypatch, track_trajectory):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import olmo_core.ops.moe as moe_ops

    monkeypatch.setattr(moe_ops, "pool_keep_mask", pool_keep_mask_inverse_scatter)
    routers, optimizers = [], []
    config = MoERouterConfigV2(
        d_model=1024,
        num_experts=512,
        top_k=16,
        dtype=DType.float32,
        normalize_expert_weights=1.0,
        emo=EmoRouterConfig(
            eos_token_id=0, min_document_expert_pool=16, max_document_expert_pool=512
        ),
    )
    # Include an independent reference/reference control. Mask/selected-index
    # equality is strict; changing the graph may change fused FP32 arithmetic.
    settings = [(False, False), (False, False), (True, False)]
    if torch.__version__.startswith(("2.11.", "2.13.")):
        # Exercise combined optimizations on every qualified Torch version.
        settings += [(False, True), (True, True)]
    for document_pool, top16 in settings:
        torch.manual_seed(754)
        router = config.build(init_device="cuda")
        router._profile_document_pool = document_pool
        router._profile_top16 = top16
        with torch.no_grad():
            router.weight.normal_(std=0.02)
        routers.append(torch.compile(router, dynamic=False))
        optimizers.append(torch.optim.AdamW(router.parameters(), lr=0.00185))
    torch.manual_seed(117)
    boundaries = torch.rand(2, 257, device="cuda") < 0.07
    boundaries[:, 0] = False
    segments = boundaries.long().cumsum(1)
    # The expert's contribution follows its identity, not its top-k slot. A
    # slot-specific coefficient invents a loss discontinuity when near ties swap.
    coefficient = torch.randn(2, 257, 512, device="cuda")
    report = []
    mode = "trajectory" if track_trajectory else "same-weights"
    output = Path(os.environ.get("RESULTS_DIR", "/results")) / f"document-router-{mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    def compare(left, right, label, *, exact=False, relative_limit=2e-4, scale=None):
        delta = left.float() - right.float()
        normalizer = left.float().norm() if scale is None else scale
        relative_l2 = float(delta.norm() / normalizer.clamp_min(1e-20))
        report.append(
            {
                "label": label,
                "max_abs": float(delta.abs().max()),
                "relative_l2": relative_l2,
                "mismatch_count": int((left != right).sum()),
                "normalization": "tensor_l2" if scale is None else "loss_terms_l1",
                "normalizer": float(normalizer),
            }
        )
        output.write_text(json.dumps(report, indent=2))
        assert bool(torch.isfinite(right).all()), label
        if exact:
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=label)
        else:
            assert relative_l2 <= relative_limit, (label, report[-1])

    for update in range(3):
        if not track_trajectory:
            with torch.no_grad():
                for router in routers[1:]:
                    for left, right in zip(routers[0].parameters(), router.parameters()):
                        right.copy_(left)
        inputs = [torch.randn(2, 257, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(8)]
        outputs: list[list[tuple[torch.Tensor, ...]]] = [[] for _ in settings]
        input_grads: list[list[torch.Tensor]] = [[] for _ in settings]
        loss_scales = []
        for arm, router in enumerate(routers):
            for microbatch, original in enumerate(inputs):
                x = original.detach().clone().requires_grad_(True)
                torch.manual_seed(1000 + update * 8 + microbatch)
                weights, indices, counts, aux = router(x, False, segment_ids=segments)
                # Both routing weights and all-expert auxiliary scores affect backward.
                routing_terms = weights * coefficient.gather(-1, indices)
                aux_term = aux[0].square().sum() * 0.01
                loss = (routing_terms.sum() + aux_term) / 8
                if arm == 0:
                    # The signed reduction can nearly cancel even when every
                    # term agrees closely. Normalize scalar loss error by its
                    # reference term magnitudes; tensor/gradient/Adam checks
                    # below keep their original norms and error limits.
                    loss_scales.append(
                        (routing_terms.detach().double().abs().sum() + aux_term.detach().abs()) / 8
                    )
                loss.backward()
                outputs[arm].append(
                    (weights.detach(), indices, counts, loss.detach(), aux[0].detach())
                )
                assert x.grad is not None
                input_grads[arm].append(x.grad)
        for arm in range(1, len(settings)):
            prefix = f"update{update}/arm{arm}"
            for mb, (reference, candidate) in enumerate(zip(outputs[0], outputs[arm])):
                for field, (left, right) in enumerate(zip(reference, candidate)):
                    if field == 1 and track_trajectory and update > 0:
                        # The independent optimizer trajectories need not preserve
                        # rank order for near ties. Still require the SAME selected
                        # experts. Primitive and same-weight checks require order too.
                        left, right = left.sort(-1).values, right.sort(-1).values
                    compare(
                        left,
                        right,
                        f"{prefix}/mb{mb}/output{field}",
                        exact=field in (1, 2),
                        scale=loss_scales[mb] if field == 3 else None,
                    )
                compare(input_grads[0][mb], input_grads[arm][mb], f"{prefix}/mb{mb}/dx")
            for index, (left, right) in enumerate(
                zip(routers[0].parameters(), routers[arm].parameters())
            ):
                compare(left.grad, right.grad, f"{prefix}/parameter{index}/grad")
        for optim in optimizers:
            optim.step()
            optim.zero_grad(set_to_none=True)
        for arm in range(1, len(settings)):
            for index, (left, right) in enumerate(
                zip(routers[0].parameters(), routers[arm].parameters())
            ):
                prefix = f"update{update}/arm{arm}/parameter{index}"
                compare(left, right, f"{prefix}/weight")
                for key, value in optimizers[0].state[left].items():
                    compare(value, optimizers[arm].state[right][key], f"{prefix}/{key}")
