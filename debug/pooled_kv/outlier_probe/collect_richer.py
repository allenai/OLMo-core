"""Print the richer-slot probe's final table as markdown, from one or more result JSONs.

    python debug/pooled_kv/outlier_probe/collect_richer.py a.json b.json
"""

import json
import sys


def main():
    for path in sys.argv[1:]:
        d = json.load(open(path))
        print(f"\n### {path}")
        print(f"ckpt {d.get('ckpt_name')} rung {d.get('rung')} eval_size {d.get('eval_size')} "
              f"gen_rows {d.get('gen_rows')} n_layers {d.get('n_layers')}")
        print("| condition | CE | CE(digits) | genF1 | R@gold_pooled (SE, n) | R@gold_real | "
              "pred_pooled | \\|slot\\| | compaction | extra FLOPs (dense / compacted) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        base = {}
        for name, r in d["conditions"].items():
            hp = r.get("hit_gold_pooled", float("nan"))
            se = r.get("hit_gold_pooled_se", float("nan"))
            n = r.get("hit_gold_pooled_count", 0)
            print(
                f"| `{name}` | {r['ce']:.3f} | {r['ce_digit']:.3f} | {r['gen_f1']:.3f} | "
                f"{hp:.3f} ± {se:.3f} ({n}) | {r.get('hit_gold_real', float('nan')):.3f} | "
                f"{r.get('pred_is_pooled', float('nan')):.3f} | {r.get('slot_norm', float('nan')):.3f} | "
                f"{r['compaction']:.3f} | {r.get('enc_frac_dense', 0):.3f} / "
                f"{r.get('enc_frac_compact', 0):.3f} |"
            )
            base[name] = (hp, se, n)
        # swap deltas
        print("\nswap controls (a real slot read must DROP under the swap):")
        for name in list(base):
            if name.endswith("_swap"):
                continue
            sw = f"{name}_swap"
            if sw in base:
                a, sa, na = base[name]
                b, sb, nb = base[sw]
                print(f"  {name:24} {a:.3f} -> {b:.3f}   delta {a - b:+.3f} "
                      f"(SE_diff ~{(sa ** 2 + sb ** 2) ** 0.5:.3f}, n {na}/{nb})")


if __name__ == "__main__":
    main()
