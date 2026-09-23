"""Extract Gutenberg passages from the SHIPPED absence_gutenberg files into disjoint
train/eval pool inputs for the exact-copy xabsence builder.

Provenance (the whole point of this script -- do NOT swap in another Gutenberg dump):
the passages are the ``documents[].text`` items of

    /data/prasann/ctc_suite_data/absence_gutenberg/absence_{train,eval}_gutenberg_n32_k3.jsonl

on cubbins, which are the CTC-suite ``absence_gutenberg`` task files themselves, generated
2026-07-19 by ``gen_absence_gutenberg_pilot_fix.sbatch`` from
``generate_absence_data.py --gutenberg`` over the HF dataset ``sedthh/gutenberg_english``
(seed 43, 3383 books scanned). Every ``source`` field in those rows reads ``absence_gutenberg``.
So an xabsence example built from this pool contains literally the same prose items a model
sees in the absence_gutenberg task -- the corpus is held fixed by construction.

Disjointness: the train pool is drawn ONLY from the absence train file and the eval pool ONLY
from the absence eval file (different examples, different sampled books), and we additionally
subtract the train text set from the eval candidates so the overlap is provably 0.

Output rows are ``{"text": <passage>}``, which is what
``generate_xabsence_data.py --build-pool --from-abstracts`` consumes.

    python build_gutenberg_pools.py --src-dir <dir> --out-dir <dir>
"""

import argparse
import json
import random
import statistics
from pathlib import Path


def iter_passages(path):
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            assert ex["source"] == "absence_gutenberg", ex["source"]
            for d in ex["documents"]:
                t = (d.get("text") or "").strip()
                if t:
                    yield t


def unique_shuffled(path, rng):
    """Unique passages in a seeded random order (file order would draw the whole pool
    from the first handful of books)."""
    seen, out = set(), []
    for t in iter_passages(path):
        if t not in seen:
            seen.add(t)
            out.append(t)
    rng.shuffle(out)
    return out


def stats(texts, label):
    w = [len(t.split()) for t in texts]
    w_sorted = sorted(w)

    def pct(q):
        return w_sorted[min(len(w_sorted) - 1, int(q / 100 * len(w_sorted)))]

    print(
        f"{label}: {len(texts)} passages | words mean={statistics.mean(w):.1f} "
        f"median={statistics.median(w)} min={min(w)} p90={pct(90)} p99={pct(99)} max={max(w)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="/data/prasann/ctc_suite_data/absence_gutenberg")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-size", type=int, default=5498)
    ap.add_argument("--eval-size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    tr_all = unique_shuffled(src / "absence_train_gutenberg_n32_k3.jsonl", rng)
    ev_all = unique_shuffled(src / "absence_eval_gutenberg_n32_k3.jsonl", rng)
    print(f"unique passages available: train-file={len(tr_all)} eval-file={len(ev_all)}")

    train = tr_all[: args.train_size]
    train_set = set(train)
    tr_file_set = set(tr_all)
    # Guarantee 0 overlap against the ENTIRE train file, not just the sampled pool.
    ev_cand = [t for t in ev_all if t not in tr_file_set]
    print(
        f"eval candidates after removing every train-FILE text: {len(ev_cand)} "
        f"(dropped {len(ev_all) - len(ev_cand)})"
    )
    evl = ev_cand[: args.eval_size]

    assert len(train) == args.train_size, len(train)
    assert len(evl) == args.eval_size, len(evl)
    overlap = train_set & set(evl)
    print(f"OVERLAP train_pool_input ^ eval_pool_input = {len(overlap)}")
    assert not overlap

    stats(train, "train")
    stats(evl, "eval")

    for name, texts in [
        ("gutenberg_passages_train.jsonl", train),
        ("gutenberg_passages_eval.jsonl", evl),
    ]:
        p = out / name
        with open(p, "w") as f:
            for t in texts:
                f.write(json.dumps({"text": t}) + "\n")
        print(f"wrote {len(texts)} -> {p}")


if __name__ == "__main__":
    main()
