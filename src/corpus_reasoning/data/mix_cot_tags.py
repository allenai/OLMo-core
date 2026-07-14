"""Tag a unified-JSONL dataset for MIXED CoT / non-CoT training, purely at the
data level (no trainer/tokenizer changes).

The olmo tokenizer (scripts/data/tokenize_unified_for_olmo.py) and the axolotl
converter (scripts/train/train.py) both auto-detect a "multitask" file by the
presence of a `_task` tag on each row, and then dispatch prompt/target building
per-row using `_task` and `_cot_mode`. So to train on a mix of CoT and plain
targets we don't touch any training code — we just emit a copy of the dataset
where each row carries `_task` and a per-row `_cot_mode` chosen by a coin flip:

    `_cot_mode = cot_on`  with probability `--cot-prob`   (e.g. "template")
    `_cot_mode = cot_off` otherwise                       (e.g. "label" = plain answer)

The split is deterministic given `--seed`, and the realized fraction is printed.

Usage:
    python scripts/data/mix_cot_tags.py \\
        --input  data/mathmatch_n100_k3_x4_r-500to500_numsonly_train.jsonl \\
        --output data/mathmatch_n100_k3_x4_r-500to500_numsonly_cotmix50_train.jsonl \\
        --task mathmatch --cot-prob 0.5 --cot-on template --cot-off label --seed 42
"""

import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="unified JSONL to tag")
    ap.add_argument("--output", required=True)
    ap.add_argument("--task", required=True,
                    help="value written to each row's `_task` tag (e.g. mathmatch)")
    ap.add_argument("--cot-prob", type=float, default=0.5,
                    help="probability a row gets the CoT target (cot-on mode)")
    ap.add_argument("--cot-on", default="template",
                    help="cot_mode for CoT rows (default: template)")
    ap.add_argument("--cot-off", default="label",
                    help="cot_mode for plain rows (default: label = bare answer)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_cot = 0
    n_total = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ex["_task"] = args.task
            if rng.random() < args.cot_prob:
                ex["_cot_mode"] = args.cot_on
                n_cot += 1
            else:
                ex["_cot_mode"] = args.cot_off
            n_total += 1
            fout.write(json.dumps(ex) + "\n")

    frac = n_cot / n_total if n_total else 0.0
    print(f"Wrote {n_total} rows -> {args.output}")
    print(f"  CoT ({args.cot_on}):   {n_cot} ({frac:.1%})")
    print(f"  plain ({args.cot_off}): {n_total - n_cot} ({1 - frac:.1%})")


if __name__ == "__main__":
    main()
