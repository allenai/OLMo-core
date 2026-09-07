"""Compiled EMO A/A/B diagnostic with production-sized documents and RNG tracking."""

import hashlib
import json
import os
from pathlib import Path

import torch
from olmoe3_medium_kda_parity import error

import olmo_core.ops.moe as moe_ops
from olmo_core.config import DType
from olmo_core.nn.moe.emo import EmoRouterConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2


def rng():
    """Fingerprint CUDA RNG without consuming random draws."""
    return hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()


def main():
    """Measure masks, expert weights, and gradients; do not update model parameters."""
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    root = Path(os.environ.get("RESULTS_DIR", "/results"))
    root.mkdir(parents=True, exist_ok=True)
    report = []
    original_mask = moe_ops.pool_keep_mask
    for batch, width, label in [(2, 1536, "medium-mb2"), (4, 1024, "small-mb4")]:
        torch.manual_seed(83173)
        config = MoERouterConfigV2(
            d_model=width,
            num_experts=512,
            top_k=16,
            dtype=DType.float32,
            normalize_expert_weights=1.0,
            emo=EmoRouterConfig(
                eos_token_id=0, min_document_expert_pool=16, max_document_expert_pool=512
            ),
        )
        weight = torch.randn(512, width, device="cuda") * 0.02
        # Mix short documents with a complete 8192-token document. Boundary
        # identities remain fixed across all arms and microbatches.
        boundaries = torch.rand(batch, 8192, device="cuda") < 0.001
        boundaries[:, 0] = False
        boundaries[0] = False
        segments = boundaries.long().cumsum(1)
        inputs = [
            torch.randn(batch, 8192, width, device="cuda", dtype=torch.bfloat16) for _ in range(4)
        ]
        coefficient = torch.randn(batch, 8192, 512, device="cuda")
        reference = None
        for arm in ("reference", "reference-aa", "optimized"):
            optimized = arm == "optimized"
            moe_ops.pool_keep_mask = (
                moe_ops.pool_keep_mask_inverse_scatter if optimized else original_mask
            )
            router = config.build(init_device="cuda")
            router._profile_document_pool = optimized
            router._profile_top16 = optimized
            with torch.no_grad():
                router.weight.copy_(weight.reshape_as(router.weight))
            compiled = torch.compile(router, dynamic=False)

            def execute(source):
                x = source.detach().clone().requires_grad_()
                weights, indices, counts, aux = compiled(x, False, segment_ids=segments)
                loss = (
                    (weights * coefficient.gather(-1, indices)).sum() + aux[0].square().sum() * 0.01
                ) / len(inputs)
                loss.backward()
                return {
                    "weights": weights.detach().cpu(),
                    "indices": indices.cpu(),
                    "counts": counts.cpu(),
                    "dx": x.grad.cpu(),
                    "loss": loss.detach().cpu(),
                    "rng": rng(),
                }

            execute(inputs[0])  # Match the trainer's compile/dry-run before seed matching.
            router.zero_grad(set_to_none=True)
            torch.manual_seed(17711)
            outputs = [execute(x) for x in inputs]
            grad = router.weight.grad.detach().cpu()
            if reference is None:
                reference = (outputs, grad)
            else:
                rows = []
                for index, (old, new) in enumerate(zip(reference[0], outputs)):
                    rows.append(
                        {
                            "microbatch": index,
                            "rng_equal": old["rng"] == new["rng"],
                            "index_mismatches": int((old["indices"] != new["indices"]).sum()),
                            "selected_set_mismatches": int(
                                (
                                    old["indices"].sort(-1).values != new["indices"].sort(-1).values
                                ).sum()
                            ),
                            "count_mismatches": int((old["counts"] != new["counts"]).sum()),
                            "weights": error(old["weights"], new["weights"]),
                            "dx": error(old["dx"], new["dx"]),
                            "loss": error(old["loss"], new["loss"]),
                        }
                    )
                result = {
                    "shape": label,
                    "arm": arm,
                    "microbatches": rows,
                    "weight_gradient": error(reference[1], grad),
                }
                report.append(result)
                print("MEDIUM_ROUTER_PARITY", json.dumps(result), flush=True)
                (root / "medium-router-parity.json").write_text(json.dumps(report, indent=2))
            del router, compiled, outputs, grad
            torch.cuda.empty_cache()
    moe_ops.pool_keep_mask = original_mask


if __name__ == "__main__":
    main()
