"""Per-point table of a layer-skip run JSON (``runs/<task>/<name>.json``): test dCE vs full and vs
the token router alone (paired SEs), FLOP ratio vs full and vs token-only, train/val/test dCE, the
per-layer skip pattern, and the baselines.

    python debug/learned_router/layerskip/summarize_layerskip.py debug/learned_router/layerskip/runs/nq/ls.json [...]
"""
import json
import sys

import numpy as np

ATTN = set(range(3, 32, 4))


def main():
    for path in sys.argv[1:]:
        r = json.load(open(path))
        ref = r["ref"]["test"]
        full, tok = np.array(ref["full"]), np.array(ref["tok"])
        n = len(full)
        tok_fl = float(np.mean(ref["tok_flops"]))
        d = tok - full
        print(f"== {path}  (test eval_size {n}; token router only: dCE {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(n):.4f}, "
              f"flash {np.mean(ref['tok_flash']) - full.mean():+.4f}, FLOPs x{tok_fl:.4f})")
        print(f"{'point':34s} {'keep':>5s} {'dCE vs full':>17s} {'dCE vs tok':>17s} {'FLOPs/full':>10s} {'/tok':>6s} "
              f"{'train':>7s} {'val':>7s} | skip attn gdn early late")
        rows = [(k, v, "pt") for k, v in r["points"].items()] + [(k, v["test"], "bl") for k, v in r["baselines"].items()]
        for name, v, kind in rows:
            t = v["test"] if kind == "pt" else v
            extra = ""
            if kind == "pt":
                sr = np.array(t["skip_rate_per_layer"])
                a_ = [i for i in range(len(sr)) if i in ATTN]
                g_ = [i for i in range(len(sr)) if i not in ATTN]
                extra = f"{v['train']['dce']:+7.4f} {v['val']['dce']:+7.4f} | {sr[a_].mean():.2f} {sr[g_].mean():.2f} {sr[:16].mean():.2f} {sr[16:].mean():.2f}"
            else:
                extra = f"{'':7s} {r['baselines'][name]['val']['dce']:+7.4f} |"
            print(f"{name:34s} {t['keep_pairs']:5.2f} {t['dce']:+8.4f} +- {t['dce_se']:.4f} {t['dce_vs_tok']:+8.4f} +- {t['dce_vs_tok_se']:.4f} "
                  f"{t['flops']:10.4f} {t['flops'] / tok_fl:6.3f} {extra}")
        for name, v in r["points"].items():
            print(f"{name} per-layer skip rate: " + " ".join(f"{x:.2f}" for x in v["test"]["skip_rate_per_layer"]))


if __name__ == "__main__":
    main()
