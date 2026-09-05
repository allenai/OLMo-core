"""Build short-heavy length-mix training arms for xabsence / contradiction / oolong.

This is the outlier/qdmatch/nq length-mix recipe (per-rung pool -> nested-prefix arms at several
token budgets -> SFT shards) ported to three more tasks, and moved off mooney: it runs as a
CPU-only Beaker job (debug/taskscale_lengthmix/beaker_cpu_job.py) with weka mounted, so pools,
arms and shards land where the trainers read them and no S3 round trip is needed.

Stages, in order (`--stage all` runs them back to back):

  pools     generate one training pool per rung. All three generators are LLM- and GPU-free here:
            xabsence assembles from a pre-built exact-twin pool, contradiction RESIZES existing
            gold rows with --expand-from-train (1-in-1-out, each row keeps its own co-sampled
            distractors), oolong re-draws items from HF oolong-synth.
  measure   tokenize 200 examples of each pool and record the MEDIAN token count. Rung labels are
            not token counts for most tasks ([[ctc-rung-labels-not-tokens]]) and the mix shares
            below are token fractions, so every share is computed from a measurement, never a name.
  arms      compose the short-heavy mixes: token shares 61.6/21.9/11.0/5.5 over the four rungs,
            nested prefixes of each pool, one arm per budget.
  tokenize  convert each arm to an SFT shard (Qwen3.5 eos 248044, --query-position after).

Task-specific hazards this build is written around:
  * contradiction -- must be `realistic` mode, never the `both` gold and never the recombined
    pool (its global filler sourcing destroys the hard negatives and caps f1 at ~.585). Rung
    pools take DISJOINT slices of the source rows, so no gold pair is reused across rungs.
  * xabsence -- EXACT variant only, and the ladder starts at P=5, not P=2: with k=3 orphans among
    2P+k candidates the set-f1 floor is 3/7=0.43 at P=2, which would put 62% of the token budget
    on a rung that is mostly floor.
  * oolong -- items are drawn WITHOUT replacement and the generator silently emits a SHORT example
    when a pool runs dry, so `measure` doubles as the realized-band check; never pass --item-regex.
"""

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
GEN = REPO / "src" / "corpus_reasoning" / "data"
WEKA = pathlib.Path(
    os.environ.get(
        "TASKSCALE_ROOT",
        "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/taskscale_lengthmix",
    )
)
SRC = WEKA / "src"  # staged inputs -- fixed, so --smoke reroutes only the OUTPUTS

# Family overrides (2026-09-05, Qwen3 flex-routing arms): TASKSCALE_TOKENIZER=Qwen/Qwen3-4B
# TASKSCALE_EOS=151643 TASKSCALE_TOK_SUBDIR=arms_tokenized_qwen3 re-tokenizes the SAME composed arms
# for the dense Qwen3 family into a sibling tree; pools/arms are untouched.
TOKENIZER = os.environ.get("TASKSCALE_TOKENIZER", "Qwen/Qwen3.5-0.8B-Base")
EOS = int(os.environ.get("TASKSCALE_EOS", "248044"))
TOK_SUBDIR = os.environ.get("TASKSCALE_TOK_SUBDIR", "arms_tokenized")
MAX_SEQ = 40960
QUERY_POSITION = "after"
SHUFFLE_SEED = 7113
SHARES = (0.616, 0.219, 0.110, 0.055)  # short-heavy, renormalized to four rungs

# rung key -> (generator knob, pool examples to generate)
TASKS = {
    "xabsence": {
        "rungs": [("4k", 5, 13000), ("8k", 11, 2400), ("16k", 23, 620), ("32k", 46, 160)],
        "budgets": [20e6, 40e6, 80e6],
        "prompt_task": "xabsence",
    },
    "contradiction": {
        "rungs": [("2k", 56, 16000), ("8k", 187, 1800), ("16k", 379, 450), ("32k", 762, 115)],
        "budgets": [14e6, 28e6, 56e6],
        "prompt_task": "contradiction",
    },
    "oolong": {
        # narrow bands around each target so a rung is a LENGTH, not a range: the shipped
        # ctc bands span an octave (2k-4k, 8k-16k, ...) which smears the length axis.
        # Pools deepened 2026-09-02 to carry a 4th budget and again 2026-09-03 for a 5th: oolong
        # has the campaign's only complete 3x4 grid on BOTH variants. The 160M point already
        # showed why depth matters -- it deleted two of three fitted crossovers and doubled the
        # third, because a 3-point sparse fit leaves fmax pinned at its bound. 320M is the next
        # test of whether the now-pinned ceilings hold. Items are redrawn combinatorially per
        # example, so extra depth costs CPU time only.
        "rungs": [
            ("2k", (1800, 2400), 104000),
            ("8k", (7200, 9200), 9600),
            ("16k", (14500, 18500), 2400),
            ("32k", (29000, 36000), 620),
        ],
        "budgets": [20e6, 40e6, 80e6, 160e6, 320e6],
        "prompt_task": "oolong",
    },
    "absence": {
        # Gutenberg text-diff. Rung knobs are the SHIPPED eval rungs' own --n-sents (90/180/360/720)
        # rather than token-matched values: the eval labels are 3.0-3.6x below the measured medians
        # (rung "2048" really measures 7.5k), so training at the label would misalign train and eval
        # by 3x. The eval ladder has no 32k rung.
        #
        # Yield collapses with N -- 4000 examples cost 2,512 books at n=90 but only 168 examples
        # came out of 20,000 books at n=720 -- so the big budgets may not compose; compose() skips
        # what the pools cannot cover.
        # MEASURED SFT-path medians: n90 -> 5602, n180 -> 10896, n360 -> 26811. The shipped n=720
        # rung is dropped: it tokenizes past 49k (5 of 8 probe examples skipped), and its eval file
        # holds only 148 examples anyway. Absence carries a ~21k intercept -- queries[0] is the
        # whole corpus minus K sentences -- so its lengths run far above its rung labels.
        "rungs": [("2k", 90, 20000), ("4k", 180, 3000), ("8k", 360, 900)],
        "budgets": [20e6, 40e6, 80e6],
        "prompt_task": "absence",
        # The rungs are long AND wide: n=360 measures 26.8k median but its tail runs past 40k, so a
        # 40960 window silently dropped 9 of 200 probe examples. 65536 matches the training seq-len,
        # which is the real ceiling.
        "max_seq": 65536,
    },
    "grouping": {
        # docs-per-example -> MEASURED medians 1964 / 8042 / 16692 / 32808 (rung_token_audit.json).
        # Well-calibrated: labels are within 4% of measurement, unusual for this suite.
        "rungs": [("2k", 10, 30000), ("8k", 43, 2600), ("16k", 88, 640), ("32k", 176, 170)],
        "budgets": [20e6, 40e6, 80e6],
        "prompt_task": "grouping",
        "max_seq": 49152,       # p90 at n=176 runs past the 40960 default
    },
    "reorder": {
        # n-chunks at --target-words 100 -> MEASURED 1912 / ~4k / 9002 / 17505. No 32k rung exists
        # on the eval side (the ladder was capped at 16k), so this mix is 2k/4k/8k/16k.
        "rungs": [("2k", 12, 24000), ("4k", 27, 3800), ("8k", 57, 900), ("16k", 116, 220)],
        "budgets": [15e6, 30e6, 50e6],
        "prompt_task": "reorder",
    },
    "textgroups": {
        # num-docs -> MEASURED 1829 / 4817 / 10344 / 21416. The shipped n=210 rung measures 50.8k
        # tokens, past our 40960 window, so the ladder tops out at n=103 and the arms use the same
        # n values the eval rungs were built at rather than interpolated ones.
        "rungs": [("2k", 11, 30000), ("4k", 24, 4000), ("8k", 50, 950), ("16k", 103, 230)],
        "budgets": [20e6, 40e6, 80e6],
        "prompt_task": "textgroups",
    },
}


def log(m):
    print(f"[taskscale {time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(cmd, **kw):
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def max_seq(task):
    return TASKS[task].get("max_seq", MAX_SEQ)


def pool_dir(task, label):
    return WEKA / "pools" / task / f"rung_{label}"


def pool_file(task, label):
    """The train JSONL for a rung, preferring the DEEPEST one present.

    Deepening a pool leaves the shallower file behind (the generator names outputs by example
    count), so a rung directory can legitimately hold more than one. Take the largest and log which
    -- asserting on exactly one file made the first deepening pass fail outright.
    """
    d = pool_dir(task, label)
    hits = [q for q in d.glob("*.jsonl") if "train" in q.name and "heldout" not in q.name]
    assert hits, f"{d}: no train jsonl"
    if len(hits) > 1:
        hits.sort(key=lambda q: sum(1 for _ in open(q)))
        log(f"{task} {label}: {len(hits)} pools present, using the deepest ({hits[-1].name})")
    return hits[-1]


# ---------------------------------------------------------------- stage: pools
def build_pools(task, force=False):
    for label, knob, count in TASKS[task]["rungs"]:
        d = pool_dir(task, label)
        if d.exists() and not force:
            try:
                have = sum(1 for _ in open(pool_file(task, label)))
                if have >= count:
                    log(f"[skip] {task} {label}: {have} examples already")
                    continue
            except AssertionError:
                pass
        d.mkdir(parents=True, exist_ok=True)
        if task == "xabsence":
            run(
                [
                    sys.executable,
                    GEN / "generate_xabsence_data.py",
                    "--pool",
                    SRC / "xabsence" / "pool_exact_train.jsonl",
                    "--num-pairs",
                    knob,
                    "--num-unmatched",
                    3,
                    "--num-train",
                    count,
                    "--num-eval",
                    0,
                    "--src-tag",
                    "pubmed",
                    "--output-dir",
                    d,
                    "--seed",
                    1300 + knob,
                ]
            )
        elif task == "contradiction":
            # disjoint row slice per rung -> a gold pair appears in at most one rung
            full = SRC / "contradiction" / "contradiction_train_pubmed_realistic_n50-950_k3.jsonl"
            off = 0
            for lab2, _k2, c2 in TASKS[task]["rungs"]:
                if lab2 == label:
                    break
                off += c2
            slice_path = d / "src_slice.jsonl"
            with open(full) as fh, open(slice_path, "w") as out:
                for i, line in enumerate(fh):
                    if i < off:
                        continue
                    if i >= off + count:
                        break
                    out.write(line)
            got = sum(1 for _ in open(slice_path))
            assert got == count, f"{label}: source exhausted, got {got} of {count} rows"
            run(
                [
                    sys.executable,
                    GEN / "generate_pubmed_contradiction_data.py",
                    "--expand-from-train",
                    slice_path,
                    "--num-docs",
                    knob,
                    "--num-contradictions",
                    3,
                    "--mode",
                    "realistic",
                    "--pool-abstracts",
                    200000,
                    "--seed",
                    42,
                    "--filler-pool-seed",
                    43,
                    "--output-dir",
                    d,
                ]
            )
        elif task == "oolong":
            lo, hi = knob
            run(
                [
                    sys.executable,
                    GEN / "generate_oolong_ladder_data.py",
                    "--num-examples",
                    count,
                    "--len-min",
                    lo,
                    "--len-max",
                    hi,
                    "--pool-max-ctx",
                    262144,
                    "--tokenizer",
                    TOKENIZER,
                    "--seed",
                    3000 + lo,
                    "--output-dir",
                    d,
                ]
            )
        elif task == "absence":
            run([sys.executable, GEN / "generate_absence_data.py", "--gutenberg",
                 "--n-sents", knob, "--k-remove", 3, "--min-sentence-words", 4,
                 "--examples-per-book", 3, "--max-books-to-scan", 20000,
                 "--num-train", count, "--num-eval", 0,
                 "--output-dir", d, "--seed", 42])
        elif task == "grouping":
            run([sys.executable, GEN / "generate_arxiv_grouping_data.py",
                 "--compact-in", SRC / "grouping" / "openalex_compact.jsonl",
                 "--num-train", count, "--num-eval", 0,
                 "--docs-per-example", knob, "--out-dir", d, "--seed", 0])
        elif task == "reorder":
            # --max-books-to-scan 20000 is NOT a performance knob: the eval split was drawn from
            # books 20,001+, so scanning past 20k walks into the eval books.
            run([sys.executable, GEN / "generate_reorder_data.py",
                 "--n-chunks", knob, "--num-examples", count + 1, "--eval-frac", 0,
                 "--target-words", 100, "--out-suffix", "100w", "--examples-per-book", 2,
                 "--max-books-to-scan", 20000, "--out-dir", d, "--seed", 42])
        elif task == "textgroups":
            run([sys.executable, GEN / "generate_textgroups_data.py",
                 "--num-docs", knob, "--num-groups", 2, "--group-size", 3, "--target", 70,
                 "--num-train", count, "--num-eval", 0, "--output-dir", d, "--seed", 42])
        log(f"{task} {label}: pool -> {pool_file(task, label)}")


# -------------------------------------------------------------- stage: measure
def measure(task):
    out = {}
    for label, _knob, _count in TASKS[task]["rungs"]:
        probe = WEKA / "probe" / task / label
        probe.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                GEN / "convert_unified_to_sft.py",
                "--task",
                TASKS[task]["prompt_task"],
                "--input",
                pool_file(task, label),
                "--out-dir",
                probe,
                "--tokenizer",
                TOKENIZER,
                "--max-seq-len",
                max_seq(task),
                "--eos",
                EOS,
                "--query-position",
                QUERY_POSITION,
                "--limit",
                200,
            ]
        )
        meta = json.loads((probe / "metadata.json").read_text())
        assert meta["num_skipped"] == 0, f"{task} {label}: {meta['num_skipped']} skipped at 200"
        out[label] = meta["median_len"]
        log(f"{task} {label}: median {meta['median_len']} tok (max {meta['max_len']})")
    path = WEKA / "pools" / task / "MEDIANS.json"
    path.write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------- stage: arms
def compose(task):
    med = json.loads((WEKA / "pools" / task / "MEDIANS.json").read_text())
    rungs = [r[0] for r in TASKS[task]["rungs"]]
    manifest = {}
    for B in TASKS[task]["budgets"]:
        # SHARES is written for four rungs; a task with fewer (absence tops out at three usable
        # lengths) keeps the same short-heavy SHAPE by renormalizing the prefix rather than
        # silently dropping the tail's token share.
        _sh = SHARES[:len(rungs)]
        _sh = [x / sum(_sh) for x in _sh]
        counts = [max(1, round(s * B / med[lab])) for s, lab in zip(_sh, rungs)]
        lines, spec, short = [], {}, None
        for lab, c in zip(rungs, counts):
            pool = open(pool_file(task, lab)).read().splitlines()
            if len(pool) < c:
                # Skip the whole budget rather than quietly building a smaller arm under its name:
                # a short rung would change the mix's length composition, not just its size, and the
                # scaling point would silently stop being comparable to the others.
                short = f"{lab} pool {len(pool)} < {c}"
                break
            lines += pool[:c]
            spec[lab] = c
        if short:
            log(f"[SKIP] {task} {B/1e6:.0f}M budget is not buildable: {short}. "
                f"Generate a deeper pool or lower the budget.")
            continue
        random.Random(SHUFFLE_SEED).shuffle(lines)
        arm = f"{task}_mix_s{int(B/1e6)}M"
        if (WEKA / TOK_SUBDIR / arm / "metadata.json").exists():
            # Already tokenized and already measured. Recomposing it from a deeper
            # pool would silently change what that budget means.
            log(f"[skip] {arm}: already tokenized, leaving as-is")
            continue
        d = WEKA / "arms" / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / "arm.jsonl").write_text("\n".join(lines) + "\n")
        tokens = sum(med[lab] * c for lab, c in spec.items())
        manifest[arm] = {
            "spec": spec,
            "n_examples": len(lines),
            "target_tokens": B,
            "measured_tokens": tokens,
            "medians": med,
            "shares": dict(zip(rungs, _sh)),
            "shuffle_seed": SHUFFLE_SEED,
        }
        log(f"{arm}: {len(lines)} ex, {tokens/1e6:.1f}M tok (target {B/1e6:.0f}M) {spec}")
    (WEKA / "arms" / f"MANIFEST_{task}.json").write_text(json.dumps(manifest, indent=2))
    return list(manifest)


# ------------------------------------------------------------- stage: tokenize
def tokenize(task):
    arms = json.loads((WEKA / "arms" / f"MANIFEST_{task}.json").read_text())
    for arm in arms:
        d = WEKA / TOK_SUBDIR / arm
        if (d / "metadata.json").exists():
            log(f"[skip] tokenize {arm}")
            continue
        d.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                GEN / "convert_unified_to_sft.py",
                "--task",
                TASKS[task]["prompt_task"],
                "--input",
                WEKA / "arms" / arm / "arm.jsonl",
                "--out-dir",
                d,
                "--tokenizer",
                TOKENIZER,
                "--max-seq-len",
                max_seq(task),
                "--eos",
                EOS,
                "--query-position",
                QUERY_POSITION,
            ]
        )
        meta = json.loads((d / "metadata.json").read_text())
        log(
            f"{arm}: {meta['num_instances']} instances, {meta['num_tokens']/1e6:.1f}M tokens, "
            f"skipped {meta['num_skipped']}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument(
        "--stage", default="all", choices=["pools", "measure", "arms", "tokenize", "all"]
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny pools under <root>/smoke -- proves the generator CLIs and the "
        "prompt/tokenize path before spending hours on the real build",
    )
    a = ap.parse_args()
    if a.smoke:
        global WEKA
        WEKA = WEKA / "smoke"
        for t in TASKS.values():
            t["rungs"] = [(lab, knob, 8) for lab, knob, _ in t["rungs"]]
            t["budgets"] = [t["budgets"][0] / 1000]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if a.stage in ("pools", "all"):
        build_pools(a.task, force=a.force)
    if a.stage in ("measure", "all"):
        measure(a.task)
    if a.stage in ("arms", "all"):
        compose(a.task)
    if a.stage in ("tokenize", "all"):
        tokenize(a.task)
    log(f"DONE {a.task} {a.stage}")


if __name__ == "__main__":
    main()
