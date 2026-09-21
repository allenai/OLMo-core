#!/usr/bin/env python
"""Measure what a CTC-suite cell (task x rung) actually COSTS on one GPU, via vLLM.

The 22-task suite to 256k is 2.52B prompt tokens at the declared eval sizes. Whether any
subsetting of it fits an 8-GPU-hour budget is an empirical question about prefill and decode
throughput, not an arithmetic one -- decode in particular is unbounded for a model that never
emits EOS, and olmo-eval deliberately passes NO decode-time stop strings (its stop rules run
post-hoc in CTCScorer, because a decode-time stop cannot honour the "not inside <think>" rule).

So this measures, per (task, rung): wall-clock, realized prompt tokens, generated tokens, and --
because a timing number from a broken pipeline is worthless -- the task's own metric and parse
rate through the vendored ctc spec, the same three calls olmo-eval's CTCScorer makes.

Prompts are built with `spec.build_prompt(example, query_position="both")`, identical to
olmo-eval's `format_request`, and the generation budget is olmo-eval's
`max(stop.max_new_tokens, spec.max_new_tokens)`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="vLLM serving copy (see qwen35-4b recipe)")
    p.add_argument(
        "--data-root",
        required=True,
        help="a local <subset>/rung_<tokens>.jsonl tree, or the literal 'hf' to read the "
        "public PrasannSinghal/ctc-suite-eval dataset (what olmo-eval itself reads)",
    )
    p.add_argument(
        "--ctc-vendor",
        default=os.path.expanduser(
            "~/projects/olmo-eval/src/olmo_eval/evals/tasks/ctc_suite/_vendor"
        ),
        help="dir holding the vendored `ctc` package (pure python, stdlib only)",
    )
    p.add_argument(
        "--cells",
        required=True,
        help="comma-separated subset:spec:rung triples. With a local tree the rung is the file's "
        "token budget (fiqa:retrieval:2048); with --data-root hf it is the split label "
        "(fiqa:retrieval:r2k).",
    )
    p.add_argument("--limit", type=int, default=100, help="examples per cell")
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument(
        "--model-family",
        default="qwen3_5",
        choices=["qwen3_5", "other"],
        help="qwen3_5 adds the VL-wrapper architecture override + limit_mm_per_prompt=0",
    )
    p.add_argument("--stop-strings", action="store_true", help="probe decode-time early stop")
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_specs(vendor_dir: str):
    """Import the vendored ctc registry and add plain `grouping`, exactly as olmo-eval does."""
    sys.path.insert(0, vendor_dir)
    from ctc.format import registry
    from ctc.format.prompts import GROUPING_INSTRUCTION
    from ctc.tasks import load_all
    from ctc.tasks._grouping import make_grouping_spec

    load_all()
    if "grouping" not in registry.names():
        registry.register(
            make_grouping_spec(
                name="grouping",
                description="Partition abstracts into their (unnamed) field clusters.",
                instruction=GROUPING_INSTRUCTION,
                rungs=("2k", "4k", "8k", "16k", "32k"),
                query_builder=lambda ex: (
                    f"{GROUPING_INSTRUCTION}\n\n{ex['queries'][0]}"
                    if ex.get("queries")
                    else GROUPING_INSTRUCTION
                ),
                sources=("openalex",),
            )
        )
    return registry


HF_DATASET = "PrasannSinghal/ctc-suite-eval"

#: Rung label -> token budget, as the olmo-eval roster declares it.
RUNG_TOKENS = {
    "r2k": 2048, "r4k": 4096, "r8k": 8192, "r16k": 16384, "r32k": 32768,
    "r64k": 65536, "r128k": 131072, "r256k": 262144, "r512k": 524288, "r1m": 1048576,
}


def _as_int(rung) -> int:
    try:
        return int(rung)
    except (TypeError, ValueError):
        return 0


def read_jsonl(path: str, limit: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def read_hf(subset: str, rung: str, limit: int) -> list[dict]:
    """Pull one rung from the public dataset, pinning the single parquet olmo-eval pins.

    Without `data_files` the loader builds every split in the config first, so reading 100 rows
    of r2k downloads the whole 2k-to-1M ladder.
    """
    from datasets import load_dataset

    ds = load_dataset(
        HF_DATASET,
        subset,
        split=rung,
        data_files={rung: f"data/{subset}/{rung}.parquet"},
    )
    return [dict(ds[i]) for i in range(min(limit, len(ds)))]


def main() -> int:
    args = parse_args()
    registry = load_specs(args.ctc_vendor)
    from ctc.eval.stopping import STOP_PRESETS
    from ctc.eval.stopping import apply as apply_stop

    cells = []
    for raw in args.cells.split(","):
        subset, spec_name, rung = raw.split(":")
        if args.data_root == "hf":
            cells.append((subset, spec_name, rung, None))
            continue
        path = os.path.join(args.data_root, subset, f"rung_{rung}.jsonl")
        if not os.path.exists(path):
            print(f"[bench] SKIP {raw}: no {path}", flush=True)
            continue
        cells.append((subset, spec_name, rung, path))
    if not cells:
        print("[bench] nothing to run", file=sys.stderr)
        return 2

    # Build every prompt up front so a data problem fails before the GPU is touched, and so the
    # reported wall-clock is generation only.
    print(f"[bench] building prompts for {len(cells)} cells", flush=True)
    built = []
    for subset, spec_name, rung, path in cells:
        spec = registry.get(spec_name)
        try:
            rows = read_hf(subset, rung, args.limit) if path is None else read_jsonl(path, args.limit)
        except Exception as exc:  # a missing rung must not lose the cells already built
            print(f"[bench] SKIP {subset}:{rung}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not rows:
            print(f"[bench] SKIP {subset}:{rung}: no rows", flush=True)
            continue
        prompts = [spec.build_prompt(ex, query_position="both") for ex in rows]
        built.append((subset, spec_name, rung, spec, rows, prompts))
        print(
            f"    {subset}:{rung} n={len(rows)} "
            f"chars_p50={sorted(len(p) for p in prompts)[len(prompts) // 2]}",
            flush=True,
        )

    from vllm import LLM, SamplingParams

    extra = {}
    if args.model_family == "qwen3_5":
        # The export is text-only but vLLM resolves Qwen3_5* to the multimodal class; these two
        # are what make it load as text. See records: qwen35-4b-vllm-load-recipe.
        extra.update(
            hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
    t0 = time.time()
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        **extra,
    )
    load_s = time.time() - t0
    print(f"[bench] LLM up in {load_s:.1f}s", flush=True)
    tok = llm.get_tokenizer()

    results = []
    for subset, spec_name, rung, spec, rows, prompts in built:
        stop = STOP_PRESETS[spec.stop]
        max_new = max(stop.max_new_tokens, spec.max_new_tokens)
        sp = SamplingParams(max_tokens=max_new, temperature=0)
        if args.stop_strings and getattr(stop, "text_stops", None):
            sp = SamplingParams(max_tokens=max_new, temperature=0, stop=list(stop.text_stops))
        t1 = time.time()
        outs = llm.generate(prompts, sp)
        wall = time.time() - t1

        prompt_toks = sum(len(o.prompt_token_ids) for o in outs)
        gen_toks = sum(len(o.outputs[0].token_ids) for o in outs)
        lens = sorted(len(o.prompt_token_ids) for o in outs)
        scores, parsed_ok = [], 0
        for ex, o in zip(rows, outs):
            cleaned = apply_stop(o.outputs[0].text or "", stop)
            parsed = spec.parse(cleaned, len(ex["documents"]))
            parsed_ok += parsed is not None
            if spec.extra.get("score_takes_example"):
                gold = ex
            else:
                gold = ex.get(spec.extra.get("gold_field", "gold_doc_indices")) or ex
            scores.append(float(spec.score(parsed, gold).get(spec.primary_metric, 0.0)))

        rec = {
            "subset": subset,
            "spec": spec_name,
            "rung": rung,
            "rung_label_tokens": RUNG_TOKENS.get(str(rung), _as_int(rung)),
            "eval_size": len(rows),
            "max_new_tokens": max_new,
            "wall_s": round(wall, 2),
            "prompt_tokens": prompt_toks,
            "gen_tokens": gen_toks,
            "prompt_tok_p50": lens[len(lens) // 2],
            "prompt_tok_p95": lens[int(len(lens) * 0.95)],
            "label_ratio": round(lens[len(lens) // 2] / max(1, RUNG_TOKENS.get(str(rung), _as_int(rung))), 3),
            "prompt_tok_per_s": round(prompt_toks / wall, 1),
            "gen_tok_per_example": round(gen_toks / max(1, len(rows)), 1),
            "s_per_example": round(wall / max(1, len(rows)), 3),
            "parse_rate": round(parsed_ok / max(1, len(rows)), 4),
            spec.primary_metric: round(sum(scores) / max(1, len(scores)), 4),
        }
        results.append(rec)
        print(f"[bench] {json.dumps(rec)}", flush=True)
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "load_s": round(load_s, 1), "cells": results}, f, indent=2)

    print(f"[bench] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
