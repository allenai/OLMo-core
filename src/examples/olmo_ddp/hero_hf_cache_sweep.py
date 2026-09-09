"""Measure cache arithmetic across all short qualification cases; never publish a model."""

import argparse
import json
import os

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from hero_hf_convert import cases, statistics
from hero_hf_download import SCRATCH, TARGETS, prepare_scratch, write_json
from olmo_core.nn.hf.config import _register_olmo3moe_auto_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("emo", "non-emo"), required=True)
    parser.add_argument("--step", type=int, choices=tuple(TARGETS.values()), required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--recurrent-prefill", action="store_true")
    args = parser.parse_args()
    prepare_scratch()
    root = SCRATCH / args.arm / f"step{args.step}"
    if not (root / "download-success.json").is_file() or root.resolve() != root:
        raise RuntimeError("Require a verified, owned scratch source")
    _register_olmo3moe_auto_classes()
    os.environ.pop("OLMO_HF_MOE_CORE_REFERENCE", None)
    os.environ["OLMO_HF_MOE_REFERENCE_LOOP"] = "1"
    torch.set_float32_matmul_precision("highest")
    if args.recurrent_prefill:
        # Diagnostic only: remove the chunk-vs-recurrent algorithm change while
        # leaving HF's independent cache lifecycle and full/cached input paths intact.
        import fla.ops.kda as kda_ops
        from fla.modules.l2norm import l2norm_fwd

        def recurrent_prefill(q, k, v, **kwargs):
            if kwargs.pop("use_qk_l2norm_in_kernel", False):
                q, _ = l2norm_fwd(q)
                k, _ = l2norm_fwd(k)
            return kda_ops.fused_recurrent_kda(q=q, k=k, v=v, **kwargs)

        kda_ops.chunk_kda = recurrent_prefill
    model = (
        AutoModelForCausalLM.from_pretrained(
            root / "hf.partial", dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(root / "hf.partial")
    rows = []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for name, ids in cases(tokenizer, model.config.vocab_size, full=True):
            if ids.shape[1] > 1024:
                continue
            print("CACHE_SWEEP_START", args.arm, args.dtype, name, flush=True)
            ids = ids.cuda()
            split = ids.shape[1] - 4
            complete = model(ids, use_cache=False).logits[:, split:].cpu()
            prefix = model(ids[:, :split], use_cache=True)
            cache = prefix.past_key_values
            pieces = []
            for index in range(split, ids.shape[1]):
                part = model(ids[:, index : index + 1], past_key_values=cache, use_cache=True)
                cache = part.past_key_values
                pieces.append(part.logits.cpu())
            actual = torch.cat(pieces, dim=1)
            expected_lp = complete.float().log_softmax(-1)
            actual_lp = actual.float().log_softmax(-1)
            error = actual_lp - expected_lp
            row = dict(case=name, **statistics(actual, complete))
            row.update(
                logprob_mean=error.abs().mean().item(),
                logprob_max=error.abs().max().item(),
                mean_kl=(expected_lp.exp() * -error).sum(-1).mean().item(),
            )
            row["meets_original_cache_limits"] = bool(
                row["finite"]
                and row["relative_l2"] <= 0.005
                and row["logprob_mean"] <= 0.01
                and row["logprob_max"] <= 0.25
            )
            rows.append(row)
            print("CACHE_SWEEP_RESULT", json.dumps(row), flush=True)
    result = dict(
        arm=args.arm,
        step=args.step,
        dtype=args.dtype,
        checks=rows,
        recurrent_prefill=args.recurrent_prefill,
        diagnostic_only=True,
    )
    suffix = "-recurrent" if args.recurrent_prefill else ""
    write_json(root / f"cache-sweep-{args.dtype}{suffix}.json", result)


if __name__ == "__main__":
    main()
