"""
Expected decode length per (task, rung): the rendered training target of real eval rows, in Qwen3.5
tokens. With the FLA autotune fix, a landmark checkpoint's eval cost is prefill + ~30 ms per decoded
token at bs=1, so answer length -- not context length -- dominates the long-answer tasks (grouping
lists every document; rerank writes the whole ranking). Feeds launch_ctc_evals.py's cost model.

    python debug/ctc_eval_speed/measure_answer_tokens.py --out debug/ctc_eval_speed/answer_tokens.json
"""

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "debug", "ctc_sft_setA"))
sys.path.insert(0, os.path.join(REPO, "src"))

import audit_train_eval_iid as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--rungs", default="r2k,r4k,r8k,r16k,r32k")
    ap.add_argument("--out", default=os.path.join(HERE, "answer_tokens.json"))
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from olmo_core.data.corpus_reasoning_prompts import build_prompt

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    R = json.load(open(os.path.join(REPO, "debug", "ctc_sft_setA", "olmo_eval_roster.json")))
    sys.path.insert(0, os.path.join(REPO, "src", "scripts", "data", "ctc_sft"))
    import build_ctc_sft as B

    spec = {t.name: t.spec for t in B.SET_A}
    out = {}
    for task, key in A.TASK_TO_ROW.items():
        row = R["roster"][key]
        out[key] = {}
        for rung in args.rungs.split(","):
            if rung not in row["rungs"]:
                continue
            rows = A._eval_rows(R["hf_dataset"], row["subset"], rung, args.sample) or []
            lens = []
            for ex in rows:
                _, ans = build_prompt(ex, task=spec[task], query_position="both",
                                      use_alpaca=False, cot_mode="none", use_titles=False)
                lens.append(len(tok(ans, add_special_tokens=False)["input_ids"]) + 2)  # + im_end, eos
            if lens:
                out[key][rung] = int(statistics.median(lens))
        print(key, out[key], flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
