"""
Where does a layer pattern's error land? Split the per-answer-token CE stored by
``layer_soft_probe.py`` (``per_row.<pattern>.tok_ce`` / ``tok_ids``) into the DOC-ID digits and
everything else.

Qwen3.5 token ids 15..24 are the single characters ``0``..``9``, so the digits of the answer's
document id are identifiable without loading a tokenizer.

    python research/token_routing/pooled_kv/layer_probe/analyze_tokens.py /net/sneetches/data/prasann/layer_probe/I_*.json
"""

import glob
import json
import sys

import numpy as np

DIGITS = set(range(15, 25))


def main():
    paths = []
    for a in sys.argv[1:]:
        paths += sorted(glob.glob(a))
    for path in paths:
        d = json.load(open(path))
        print(
            f"== {path.split('/')[-1]}  task={d['task']} rung={d['rung']} "
            f"eval_size={d['eval_size']} slot_source={d['slot_source']} keep={d['keep']}"
            + ("   ** eval_size < 500 **" if d["eval_size"] < 500 else "")
        )
        print(
            f"{'pattern':16} {'CE all':>14} {'CE doc-id digits':>18} {'CE other':>14} {'digit share of CE':>18}"
        )
        for name, pr in d["per_row"].items():
            if "tok_ce" not in pr or not pr["tok_ce"]:
                continue
            ce = np.concatenate([np.asarray(r) for r in pr["tok_ce"]])
            ids = np.concatenate([np.asarray(r) for r in pr["tok_ids"]])
            m = np.isin(ids, list(DIGITS))
            if m.sum() == 0:
                print(f"{name:16} {ce.mean():14.4f} {'(no digit tokens)':>18}")
                continue

            def ms(v):
                return f"{v.mean():8.4f}+-{v.std(ddof=1) / np.sqrt(len(v)):<5.4f}"

            share = ce[m].sum() / max(1e-9, ce.sum())
            print(
                f"{name:16} {ce.mean():14.4f} {ms(ce[m]):>18} {ms(ce[~m]):>14} "
                f"{share:11.3f} ({m.mean():.2f} of tokens)"
            )
        print()


if __name__ == "__main__":
    main()
