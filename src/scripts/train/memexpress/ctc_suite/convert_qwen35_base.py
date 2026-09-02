"""Convert an already-downloaded local Qwen3.5 HF base -> olmo-core model-only distcp.

Thin wrapper around `corpus_reasoning.train.convert_hf_to_olmo`'s hybrid Qwen3.5 path
(`_convert_qwen3_5`, which drives `convert_qwen3_5_state_from_hf` — same converter used for
`q35-08b-base-modelonly`). The upstream CLI unconditionally calls `snapshot_download(base_model)`
before dispatching, which only works for a bare HF repo id; our 4B/9B bases already sit on
cubbins node-local `/data/prasann/hf_models/Qwen3.5-{4B,9B}-Base` as flat HF directories (not an
HF-cache snapshot), so this script skips straight to the local-dir conversion instead of
re-downloading. Also runs the marker-embedding audit (cos + norm of the box_start/box_end rows)
on the converted checkpoint so Stage 0 of `records/ctc-suite-scaling-plan.md` §4 can be recorded
in one shot.

Usage:
    python convert_qwen35_base.py --base-dir /data/prasann/hf_models/Qwen3.5-4B-Base \
        --out /data/prasann/ctc_suite/bases/q35-4b-base-modelonly
"""

import argparse
import json
import os
import sys

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        required=True,
        help="local flat HF checkpoint dir (has config.json + *.safetensors)",
    )
    ap.add_argument("--out", required=True, help="output olmo-core model-only distcp dir")
    ap.add_argument("--skip-audit", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from corpus_reasoning.lib.olmo_models import (
        build_transformer_config,
        resolve_olmo_model,
    )
    from corpus_reasoning.train.convert_hf_to_olmo import _convert_qwen3_5
    from olmo_core.data.tokenizer import TokenizerConfig

    spec = resolve_olmo_model(args.base_dir)
    print(
        f"[convert] {args.base_dir} -> builder={spec.builder} vocab={spec.vocab_size} "
        f"is_hybrid={spec.is_hybrid}",
        flush=True,
    )
    if not spec.is_hybrid:
        sys.exit(
            f"[convert] {args.base_dir} resolved to a non-hybrid builder ({spec.builder}); "
            "this script is for Qwen3.5 hybrid bases only."
        )

    tok = AutoTokenizer.from_pretrained(args.base_dir, trust_remote_code=True)
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    bos = tok.bos_token_id

    model_cfg = build_transformer_config(spec)
    tok_cfg = TokenizerConfig(
        vocab_size=spec.vocab_size,
        eos_token_id=eos,
        pad_token_id=pad,
        bos_token_id=bos,
        identifier=args.base_dir,
    )
    print(f"[convert] eos={eos} pad={pad} bos={bos}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    _convert_qwen3_5(args.base_dir, args.out, model_cfg, tok_cfg)

    model_and_optim = os.path.join(args.out, "model_and_optim")
    metadata_path = os.path.join(model_and_optim, ".metadata")
    if not os.path.exists(metadata_path):
        sys.exit(
            f"[convert] FAILED: {metadata_path} missing after save_model_and_optim_state "
            "-- conversion did not complete cleanly."
        )
    print(f"[convert] verified {metadata_path} exists", flush=True)

    if args.skip_audit:
        return

    # --- marker-embedding audit (records/ctc-suite-scaling-plan.md Stage 0 gate) ---
    from olmo_core.distributed.checkpoint import load_keys

    (emb,) = load_keys(model_and_optim, ["model.embeddings.weight"])
    emb = emb.float()
    box_start, box_end = 248049, 248050
    e0, e1 = emb[box_start], emb[box_end]
    cos = torch.nn.functional.cosine_similarity(e0.unsqueeze(0), e1.unsqueeze(0)).item()
    n0, n1 = e0.norm().item(), e1.norm().item()
    trained_median = emb[:150_000].norm(dim=-1).median().item()

    result = {
        "base_dir": args.base_dir,
        "out": args.out,
        "box_start_id": box_start,
        "box_end_id": box_end,
        "cos_box_start_end": cos,
        "norm_box_start": n0,
        "norm_box_end": n1,
        "trained_row_median_norm_first150k": trained_median,
        "norm_ratio_start": n0 / trained_median,
        "norm_ratio_end": n1 / trained_median,
        "verdict": (
            "PASS"
            if (
                abs(cos) < 0.9
                and 0.5 < (n0 / trained_median) < 2.0
                and 0.5 < (n1 / trained_median) < 2.0
            )
            else "FAIL-CHECK"
        ),
    }
    print("[audit] " + json.dumps(result, indent=2), flush=True)
    with open(os.path.join(args.out, "marker_audit.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
