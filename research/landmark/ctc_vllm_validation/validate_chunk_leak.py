"""Validate that the document-chunked mask has NO free-token leakage between chunks.

The chunked arm isolates each document: at inference, vllm_chunked_patch._build_chunk_ids_for_batch
scans the token ids for <|doc_start|>..<|doc_end|> pairs and labels every token FREE (-1, attends
globally) unless it falls inside a matched doc span (chunk id >= 0). The design invariant
(document_chunk_landmark._wrap_documents docstring) is strict: the ONLY free tokens are the
instruction/question PREFIX (before the first doc) and the query/answer SUFFIX (after the last doc).
Every id / title / label AND every inter-document separator must live inside a chunk.

A LEAK = any FREE token strictly BETWEEN two consecutive wrapped chunks (end of chunk k .. start of
chunk k+1). Such a token attends to everything and everything attends to it -> a bridge across the
supposedly-isolated chunks. This inflates the chunked arm and confounds dense-vs-chunked.

This driver builds each task's REAL eval prefill with the SAME build_eval_prefill the native
evaluators call (no reimplementation), replicates the exact chunk-id assignment, and reports per task:
  - inter_chunk_free   : total FREE tokens between chunks (MUST be 0)
  - ex_with_leak        : how many sampled examples have any inter-chunk free token
  - wrap coverage       : n_chunks vs expected document count (docs silently unwrapped = leak too)
  - decoded leak samples: what text is leaking, to diagnose the cause

CPU-only; corpus-reasoning-olmo env. Usage:
  python validate_chunk_leak.py --eval-root /scratch/users/prasann/ctc_suite_staged/eval_rungs \
     --rung 2048 --max-samples 100 [--tasks niah outlier ...]
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "research", "common"))  # corpus_reasoning.*

DOC_START, DOC_END = 248049, 248050

# eval_rungs dir name -> run_rung_eval --task value (mirrors gpu_eval_task.sh's alias corrections).
TASK_ALIAS = {
    "niah": "niah_contradiction",
    "scifact": "beir_scifact",
    "fiqa": "beir_fiqa",
    "outlier_amzn": "outlier_amazon",
    "absence_gutenberg": "absence",
    "xabsence": "absence",
    "qdmatch_hpqa": "qdmatch",
    "qdmatch_nq": "qdmatch",
    "obliq_retrieval": "retrieval",
}


def analyze_prefill(prefill):
    """Return (n_chunks, prefix_free, suffix_free, inter_chunk_free, gaps) for a token list.

    Replicates vllm_chunked_patch._build_chunk_ids_for_batch's matching: for each doc_start, pair it
    with the next doc_end after it. `gaps` is a list of (gap_start_idx, gap_end_idx) for inter-chunk
    free spans (tokens strictly between one chunk's end and the next chunk's start)."""
    starts = [i for i, t in enumerate(prefill) if t == DOC_START]
    ends = [i for i, t in enumerate(prefill) if t == DOC_END]
    spans = []
    ei = 0
    for s in starts:
        while ei < len(ends) and ends[ei] < s:
            ei += 1
        if ei >= len(ends):
            break
        spans.append((s, ends[ei]))
        ei += 1
    if not spans:
        return 0, len(prefill), 0, 0, []
    prefix_free = spans[0][0]  # tokens before first doc_start
    suffix_free = len(prefill) - (spans[-1][1] + 1)  # tokens after last doc_end
    inter = 0
    gaps = []
    for k in range(len(spans) - 1):
        gap_lo = spans[k][1] + 1  # first token after chunk k's doc_end
        gap_hi = spans[k + 1][0]  # chunk k+1's doc_start (exclusive)
        if gap_hi > gap_lo:
            inter += gap_hi - gap_lo
            gaps.append((gap_lo, gap_hi))
    return len(spans), prefix_free, suffix_free, inter, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="/scratch/users/prasann/ctc_suite_staged/eval_rungs")
    ap.add_argument("--rung", default="2048")
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--tasks", nargs="*", default=None, help="eval_rungs dir names; default = all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import importlib.util

    from transformers import AutoTokenizer

    from corpus_reasoning.eval import eval_lc_native_docchunk as gen_eval
    from corpus_reasoning.eval import eval_lc_native_docchunk_contra as contra_eval
    from corpus_reasoning.eval.evaluate import load_unified_examples

    # TODO(port): depends on src/scripts/eval/ctc_suite/run_rung_eval.py, not ported
    rre_path = os.path.join(REPO_ROOT, "src", "scripts", "eval", "ctc_suite", "run_rung_eval.py")
    spec = importlib.util.spec_from_file_location("_rre", rre_path)
    rre = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(rre_path))
    spec.loader.exec_module(rre)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.tasks:
        task_dirs = args.tasks
    else:
        task_dirs = sorted(
            d for d in os.listdir(args.eval_root) if os.path.isdir(os.path.join(args.eval_root, d))
        )

    results = []
    for td in task_dirs:
        rung_file = os.path.join(args.eval_root, td, f"rung_{args.rung}.jsonl")
        if not os.path.exists(rung_file):
            # fall back to any available rung
            cand = (
                sorted(
                    f
                    for f in os.listdir(os.path.join(args.eval_root, td))
                    if f.startswith("rung_") and f.endswith(".jsonl")
                )
                if os.path.isdir(os.path.join(args.eval_root, td))
                else []
            )
            if not cand:
                results.append({"task": td, "status": "no rung data"})
                continue
            rung_file = os.path.join(args.eval_root, td, cand[0])

        eval_task = TASK_ALIAS.get(td, td)
        canonical = rre.TASK_ALIASES.get(eval_task, eval_task)
        canonical = gen_eval.TASK_ALIASES.get(canonical, canonical)
        if canonical not in gen_eval.TASK_CFG:
            results.append({"task": td, "status": f"unresolved task ({eval_task}->{canonical})"})
            continue
        cfg = gen_eval.TASK_CFG[canonical]
        cot_mode = cfg["cot"]
        is_contra = canonical == "contradiction"

        try:
            examples = load_unified_examples(
                rung_file,
                args.max_samples,
                task=canonical,
                query_position="both",
                use_alpaca=True,
            )
        except Exception as e:
            results.append({"task": td, "status": f"load error: {e}"})
            continue

        tot_inter = 0
        ex_with_leak = 0
        wrap_short = 0  # examples where n_chunks < expected doc count
        n_ex = 0
        leak_samples = []
        chunks_seen = []
        for ex in examples:
            raw = ex.get("ex", ex)
            try:
                if is_contra:
                    prefill = contra_eval.build_eval_prefill(
                        tok,
                        raw,
                        variant="dense",
                        cot_mode=cot_mode,
                        doc_start_id=DOC_START,
                        doc_end_id=DOC_END,
                        mem_freq=63,
                    )
                else:
                    prefill = gen_eval.build_eval_prefill(
                        tok,
                        raw,
                        canonical,
                        variant="dense",
                        cot_mode=cot_mode,
                        doc_start_id=DOC_START,
                        doc_end_id=DOC_END,
                        mem_freq=63,
                    )
            except Exception as e:
                results.append({"task": td, "status": f"prefill error: {e}"})
                prefill = None
                break
            n_ex += 1
            n_chunks, pf, sf, inter, gaps = analyze_prefill(prefill)
            chunks_seen.append(n_chunks)
            tot_inter += inter
            if inter > 0:
                ex_with_leak += 1
                if len(leak_samples) < 4:
                    lo, hi = gaps[0]
                    leaked_txt = tok.decode(prefill[lo:hi])
                    leak_samples.append(
                        {
                            "gap_tokens": hi - lo,
                            "n_gaps": len(gaps),
                            "leaked_text": leaked_txt[:200],
                        }
                    )
            # wrap coverage (document tasks only: we know the expected count)
            if cfg["chunk_by"] == "document":
                exp = len(raw.get("documents", []))
                if exp and n_chunks < exp:
                    wrap_short += 1
        else:
            pass
        if n_ex == 0:
            if not any(r["task"] == td for r in results):
                results.append({"task": td, "status": "no examples"})
            continue
        results.append(
            {
                "task": td,
                "canonical": canonical,
                "chunk_by": cfg["chunk_by"],
                "n_ex": n_ex,
                "inter_chunk_free_total": tot_inter,
                "ex_with_leak": ex_with_leak,
                "wrap_short_ex": wrap_short,
                "chunks_min_max": [min(chunks_seen), max(chunks_seen)] if chunks_seen else None,
                "leak_samples": leak_samples,
                "status": "CLEAN" if (tot_inter == 0 and wrap_short == 0) else "LEAK",
            }
        )

    # report
    print(
        f"\n{'='*92}\nCHUNK-LEAK VALIDATION  rung={args.rung}  max_samples={args.max_samples}\n{'='*92}"
    )
    print(
        f"{'task':<22}{'status':<8}{'n_ex':>5}{'inter_free':>12}{'leak_ex':>9}{'wrap_short':>11}  chunks"
    )
    for r in results:
        if r.get("status") in ("CLEAN", "LEAK"):
            print(
                f"{r['task']:<22}{r['status']:<8}{r['n_ex']:>5}{r['inter_chunk_free_total']:>12}"
                f"{r['ex_with_leak']:>9}{r['wrap_short_ex']:>11}  {r['chunks_min_max']}"
            )
        else:
            print(f"{r['task']:<22}{'--':<8}   {r.get('status')}")
    print("\n--- leak detail (decoded inter-chunk free spans) ---")
    for r in results:
        if r.get("status") == "LEAK":
            print(
                f"\n[{r['task']}] {r['ex_with_leak']}/{r['n_ex']} ex leak, "
                f"{r['inter_chunk_free_total']} total inter-chunk free toks, "
                f"wrap_short={r['wrap_short_ex']}"
            )
            for s in r["leak_samples"]:
                print(
                    f"    gap={s['gap_tokens']}tok n_gaps={s['n_gaps']}  leaked={s['leaked_text']!r}"
                )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
