"""Faithful HELMET-style summarization metric: LLM-judge atomic-claim F1.

Given an eval details file (JSON list of {"prediction", "reference"}), uses a
local vLLM-served model (default Qwen2.5-14B-Instruct) to:
  1. decompose each REFERENCE summary into atomic claims,
  2. judge RECALL  = fraction of reference claims supported by the prediction,
  3. judge PRECISION = fraction of prediction sentences supported by the reference,
  4. F1 = harmonic mean (× fluency is omitted; predictions are model text).

This is the model-based metric HELMET uses (with GPT-4o); we run it locally for
free (see [[project_local_qwen_gen]]). Run inside a serve+consume job that sets
LOCAL_LLM_BASE_URL (mirror jobs/serve_qwen_validate.sh).

Usage:
    python scripts/eval/score_summarization_claims.py \\
        --details <eval_details.json> --base-url http://127.0.0.1:8765/v1 \\
        --model Qwen/Qwen2.5-14B-Instruct
"""

import argparse
import json
import re

DECOMPOSE = """Break the following summary into a list of atomic factual claims, one per line, each a short standalone sentence. Output only the claims.

Summary: {text}

Claims:"""

SUPPORTED = """Reference text:
{ref}

Statement: {claim}

Is the statement supported by (entailed by) the reference text? Answer with exactly one word: YES or NO."""


def _lines(text):
    return [re.sub(r'^[\-\*\d\.\)\s]+', '', l).strip()
            for l in (text or "").splitlines() if l.strip()]


def _sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text or "") if s.strip()]


def main():
    from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--max-concurrent", type=int, default=32)
    args = ap.parse_args()

    rows = json.load(open(args.details))
    client = ParallelResponsesClient(max_concurrent=args.max_concurrent,
                                     use_cache=True, local_base_url=args.base_url)

    # 1) decompose references into claims
    dec = client.run(model=args.model,
                     prompts=[DECOMPOSE.format(text=r["reference"]) for r in rows],
                     temperature=0.0, max_output_tokens=512)
    ref_claims = [_lines(d.get("response")) for d in dec]

    # 2) recall: each ref claim supported by the prediction?
    rec_prompts, rec_map = [], []
    for i, r in enumerate(rows):
        for c in ref_claims[i]:
            rec_map.append(i)
            rec_prompts.append(SUPPORTED.format(ref=r["prediction"], claim=c))
    rec = client.run(model=args.model, prompts=rec_prompts, temperature=0.0,
                     max_output_tokens=6)

    # 3) precision: each prediction sentence supported by the reference?
    pre_prompts, pre_map = [], []
    pred_sents = [_sentences(r["prediction"]) for r in rows]
    for i, r in enumerate(rows):
        for s in pred_sents[i]:
            pre_map.append(i)
            pre_prompts.append(SUPPORTED.format(ref=r["reference"], claim=s))
    pre = client.run(model=args.model, prompts=pre_prompts, temperature=0.0,
                     max_output_tokens=6)

    def yes(r):
        return (r.get("response") or "").strip().upper().startswith("YES")

    rec_hit = [0] * len(rows); rec_tot = [0] * len(rows)
    for i, r in zip(rec_map, rec):
        rec_tot[i] += 1; rec_hit[i] += yes(r)
    pre_hit = [0] * len(rows); pre_tot = [0] * len(rows)
    for i, r in zip(pre_map, pre):
        pre_tot[i] += 1; pre_hit[i] += yes(r)

    f1s = []
    for i in range(len(rows)):
        rr = rec_hit[i] / rec_tot[i] if rec_tot[i] else 0.0
        pp = pre_hit[i] / pre_tot[i] if pre_tot[i] else 0.0
        f1s.append(2 * pp * rr / (pp + rr) if (pp + rr) else 0.0)
    print(json.dumps({"eval_size": len(rows), "claim_f1": sum(f1s) / max(1, len(f1s)),
                      "mean_recall": sum(rec_hit) / max(1, sum(rec_tot)),
                      "mean_precision": sum(pre_hit) / max(1, sum(pre_tot))},
                     indent=2))


if __name__ == "__main__":
    main()
