"""Validate a local Qwen vs Gemini for contradiction-perturbation generation.

Runs the same SIMPLE/SUBTLE perturbation prompts through (a) a local vLLM-served
Qwen and (b) gemini-2.5-flash, then judges BOTH with the same Gemini validity
judge. Reports validity pass-rate and mean word-overlap so we can decide whether
the local model is at parity before adopting it for data generation.

Run inside jobs/serve_qwen_validate.sh, which starts the vLLM server first and
sets LOCAL_LLM_BASE_URL + QWEN_MODEL.
"""

import os
import random

from corpus_reasoning.lib.llm_request_client import ParallelResponsesClient
from corpus_reasoning.data.generate_pubmed_contradiction_data import (
    load_pubmed_pool, SIMPLE_PROMPT, SUBTLE_PROMPT, JUDGE_PROMPT, clean_response,
)

N_DEV = int(os.environ.get("N_DEV", "40"))
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-14B-Instruct")
BASE_URL = os.environ["LOCAL_LLM_BASE_URL"]


def word_overlap(a, b):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / max(1, len(wa | wb))


def gen(client, model, sents, modes):
    prompts = [(SIMPLE_PROMPT if m == "simple" else SUBTLE_PROMPT).format(sentence=s)
               for s, m in zip(sents, modes)]
    resp = client.run(model=model, prompts=prompts, temperature=0.7,
                      max_output_tokens=200)
    return [clean_response(r.get("response") or "") for r in resp]


def judge(judge_client, sents, perts):
    idx = [i for i, p in enumerate(perts) if p]
    prompts = [JUDGE_PROMPT.format(a=sents[i], b=perts[i]) for i in idx]
    resp = judge_client.run(model="gemini-2.5-flash", prompts=prompts,
                            temperature=0.0, max_output_tokens=10)
    ok = sum(1 for r in resp if (r.get("response") or "").strip().upper().startswith("YES"))
    return ok, len(idx)


def main():
    rng = random.Random(0)
    claim_pool, _ = load_pubmed_pool(3000, seed=0)
    rng.shuffle(claim_pool)
    sents = [s for s, _ in claim_pool[:N_DEV]]
    modes = (["simple", "subtle"] * (N_DEV // 2 + 1))[:N_DEV]

    qwen = ParallelResponsesClient(max_concurrent=16, use_cache=False,
                                   local_base_url=BASE_URL)
    gem = ParallelResponsesClient(max_concurrent=16, use_cache=False)

    print(f"\n=== generating {N_DEV} perturbations with each model ===")
    q_perts = gen(qwen, QWEN_MODEL, sents, modes)
    g_perts = gen(gem, "gemini-2.5-flash", sents, modes)

    print("=== judging both with gemini-2.5-flash ===")
    for name, perts in [("QWEN", q_perts), ("GEMINI", g_perts)]:
        usable = [p for p in perts if p]
        ov = [word_overlap(s, p) for s, p in zip(sents, perts) if p]
        ov_sub = [word_overlap(s, p) for s, p, m in zip(sents, perts, modes)
                  if p and m == "subtle"]
        ok, judged = judge(gem, sents, perts)
        print(f"\n----- {name} ({QWEN_MODEL if name=='QWEN' else 'gemini-2.5-flash'}) -----")
        print(f"  usable outputs:      {len(usable)}/{N_DEV}")
        print(f"  valid contradictions:{ok}/{judged}  ({ok/max(1,judged):.0%})")
        print(f"  mean word-overlap:   {sum(ov)/max(1,len(ov)):.3f}  "
              f"(subtle only: {sum(ov_sub)/max(1,len(ov_sub)):.3f})")

    print("\n=== sample (sentence | qwen | gemini) ===")
    for i in range(min(4, N_DEV)):
        print(f"\n[{modes[i]}] S : {sents[i]}")
        print(f"        Q : {q_perts[i]}")
        print(f"        G : {g_perts[i]}")


if __name__ == "__main__":
    main()
