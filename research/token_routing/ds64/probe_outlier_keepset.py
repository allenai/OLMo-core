"""
Why outlier collapses: quantify the "gold is always real" shortcut in the training keep set.

CPU only, no data and no GPU -- it exercises the real selection code
(:func:`olmo_core.nn.attention.gold_grad_mask.select_keep_docs`) on outlier-shaped rows and reports
how often the gold document survives as REAL under each keep mode.

At inference every document is real, so any cue of the form "the answer is one of the few real
documents" is worth exactly nothing at eval. The size of that cue during training is the number this
prints: P(gold real) together with the candidate-set shrink it buys.

    python research/token_routing/ds64/probe_outlier_keepset.py
"""

import random
import sys

sys.path.insert(0, "src")

from olmo_core.nn.attention.gold_grad_mask import select_keep_docs  # noqa: E402

# outlier eval rungs, by document count (rung_2048 .. the 64k xlong file at n448)
RUNGS = {"2k": 15, "8k": 55, "16k": 110, "32k": 220, "64k": 448}
KEEP = 1 / 3
TRIALS = 4000


def main():
    print(f"outlier keep-set probe — keep {KEEP:.3f}, {TRIALS} rows per cell\n")
    hdr = f'{"rung":>6}{"docs":>7}{"kept":>7}' + "".join(
        f"{m:>22}" for m in ("gold_plus_random", "gold_subsample", "gold-blind(keep_prob)")
    )
    print(hdr)
    for rung, n in RUNGS.items():
        present = list(range(n))
        gold = {0}
        n_rand = max(1, round(KEEP * (n - 1)))
        row = f"{rung:>6}{n:>7}{n_rand + 1:>7}"
        for mode in ("gold_plus_random", "gold_subsample", "blind"):
            hits = 0
            for t in range(TRIALS):
                rng = random.Random(t)
                if mode == "blind":
                    # the fallback path: each doc kept independently with prob KEEP
                    keep = {d for d in present if rng.random() < KEEP}
                else:
                    # gold_subsample keeps n_gold RANDOMLY CHOSEN gold docs; with one gold doc and
                    # n_gold=1 it degenerates to gold_plus_random, so the base-rate-preserving mode
                    # has nothing to offer a single-gold task like outlier. n_gold=1 shown.
                    keep = select_keep_docs(
                        present, gold, n_random=n_rand, mode=mode, rng=rng,
                        n_gold=1 if mode == "gold_subsample" else 0,
                    )
                hits += int(bool(gold & keep))
            p = hits / TRIALS
            # what the cue buys: candidates the model must weigh, real-only vs all
            shrink = n / (n_rand + 1) if p > 0.99 else 1.0
            row += f"{p:>13.3f}{('  (' + f'{shrink:.1f}' + 'x)') if shrink > 1 else '       ':>9}"
        # the cue's RATIO is constant, but the absolute distractors it hides grows with context --
        # that, not the ratio, is why the collapse deepens rung by rung.
        row += f"{n - (n_rand + 1):>8} extra at eval"
        print(row)
    print(
        "\nP(gold real) = how often the answer document is one of the few REAL documents in training.\n"
        "At eval every document is real, so a cue worth (Nx) during training is worth 1x at eval:\n"
        "that gap is the train/eval shortcut. gold_plus_random pins it at 1.000 by construction."
    )


if __name__ == "__main__":
    main()
