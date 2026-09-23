"""
Build a CTC SFT training set — one command, one declarative roster, reproducible by construction.

Replaces the three-script chain (``build_ctc_sft_sets.sh`` -> ``run_ctc_sft_build_beaker.sh`` ->
``convert_ctc_setA_13task_gantry.sh``), whose per-task knowledge lived in three places and whose
failures were silent: a synthetic task rejecting ``--pool`` killed 10 of 65 builds while the job
reported ``rc=0``, and the merge carried on with 11 tasks of 13.

Everything task-specific is declared ONCE, in :data:`SETS`, and every omission is recorded with its
reason in ``manifest.json`` rather than showing up as a quietly smaller set.

Reproducibility rests on three pins, all recorded in the manifest:

* ``--seed`` (default 42). ``ctc-data`` keys its streams by ``(seed, split, rung)``, so a given seed
  reproduces a rung byte-for-byte regardless of how many examples are asked for.
* the resolved ``ctc`` package commit, read from the installed distribution.
* the OLMo-core commit this script ran from.

``--check`` re-hashes an existing build against its manifest and reports any drift, so "same inputs
-> same bytes" is a claim the tool verifies rather than one you take on faith.

    # build + tokenize set A, 13 tasks x 5 buckets, straight to weka
    python src/scripts/data/ctc_sft/build_ctc_sft.py --set set-a \
        --out /weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_sft_sets --convert

    # verify a previous build reproduces
    python src/scripts/data/ctc_sft/build_ctc_sft.py --set set-a --out <root> --check
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

#: Tokens a rung label stands for. Rung LABELS are not token counts (contradiction runs ~1.5x under
#: its label); this is only the divisor that turns a token budget into an example count.
BUCKET_TOKENS = {
    "2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384,
    "32k": 32768, "64k": 65536, "128k": 131072, "256k": 262144,
}


def rung_tokens(label: str) -> int:
    """
    Tokens a rung label stands for, parsed rather than looked up.

    A CEILING is not always a bucket: ``strmatch`` tops out at ``48k``, which sits between two
    rungs and has no entry in :data:`BUCKET_TOKENS`. Looking ceilings up in that table raised
    ``KeyError: '48k'`` and killed the whole build before a single cell ran.

    :param label: A rung label such as ``32k``, ``48k`` or ``1m``.

    :returns: Its token count.

    :raises ValueError: If the label is not ``<number><k|m>``.
    """
    t = label.strip().lower()
    if t.endswith("k"):
        return int(float(t[:-1]) * 1024)
    if t.endswith("m"):
        return int(float(t[:-1]) * 1024 * 1024)
    raise ValueError(f"unparseable rung label {label!r}; want <number>k or <number>m")


@dataclasses.dataclass(frozen=True)
class Task:
    """One ladder in a set, with everything that differs between ladders."""

    name: str
    #: The ``build_prompt`` spec. Ladder names are NOT spec names -- ``nq`` renders as ``retrieval``,
    #: ``hotpotqa`` as ``cot_retrieval``, ``qdmatch_nq`` as ``qdmatch``. A ladder name passed where a
    #: spec is expected raises on EVERY row, which reads as a 100% error rate rather than a crash.
    spec: str
    #: ``document`` = each ``documents[i]`` is a chunk. ``line`` = items are ``||``-delimited lines
    #: inside one document (oolong only).
    chunk_by: str = "document"
    #: ``--pool auto`` fetches a seed pool. A pure-synthetic generator has no corpus and REJECTS the
    #: flag, failing every one of its buckets. Cross-checked against ctc at startup.
    pooled: bool = True
    #: Hard ceiling; buckets above it are skipped with a reason rather than attempted and silently
    #: lost. Need not be a rung label -- ``strmatch`` tops out at ``48k``, between two rungs.
    max_rung: Optional[str] = None
    #: Why :attr:`max_rung` is where it is, printed on every skip.
    max_rung_reason: str = ""
    #: Buckets withheld for a data-quality reason, as ``{bucket: why}``. These are deliberate.
    withhold: Dict[str, str] = dataclasses.field(default_factory=dict)
    #: Build with an in-repo generator (a key of :data:`EXTERNAL`) instead of ``ctc``, for eval
    #: rows ``ctc`` has no generator for. Its train pool is read from ``--external-pools``.
    external: Optional[str] = None


_TEXTGROUPS_SHORTCUT = (
    "fails the gold_length_bias shortcut audit (0.620 vs 0.228 chance at 8k, 0.485 vs 0.114 at "
    "16k): the gold group is findable from length alone, so training on it teaches length-matching "
    "instead of the task"
)

#: ``absence`` was dropped (prasann, 2026-09-22): its examples are one contiguous sentence run from
#: a single Gutenberg book, so 128k needs 1,736 sentences where the longest prose run supplies
#: 1,367. Reaching it means exporting a larger pool, which was judged not worth it.
SET_A: List[Task] = [
    Task("nq", "retrieval"),
    # `retrieval`, NOT `cot_retrieval`: the suite grades ctc_hpqa with spec="retrieval", and the
    # eval's registry has no cot_retrieval at all. The two are not interchangeable -- cot_retrieval
    # PREFIXES its target (ctc calls the pair "inconsistent" in qa/spec.py), so rendering training
    # targets through it teaches a CoT prefix the eval's retrieval parser never expects. The
    # converter accepts either name silently, because it renders via olmo_core's build_prompt
    # rather than the eval's registry.
    Task("hotpotqa", "retrieval"),
    Task("qdmatch_nq", "qdmatch"),
    Task("outlier", "outlier"),
    Task("oolong", "oolong", chunk_by="line"),
    Task("contradiction", "contradiction"),
    # The suite's xabsence rows are the ONE-SIDED EXACT-COPY Gutenberg construction (orphan_side=A,
    # P=18/39/81/165/333 at 2k-32k) -- not ctc's PubMed paraphrase generator, which is a different
    # task under the same name. Built from the disjoint Gutenberg TRAIN pool with the generator the
    # eval ladder came from.
    Task("xabsence", "xabsence", external="xabsence_gutenberg_onesided"),
    Task("reorder", "reorder"),
    # Bounded at 32k by the SEED POOL, not by MS MARCO: `msmarco.load_pool` defaults
    # `max_docs=250`, which is exactly the 32k rung, and the per-query fill is drawn and CE-scored
    # at export time. 64k asks for 501 and the draw rejects 50 times running, which reads as "the
    # corpus is too small" for an 8.8M-passage index. Lifting it needs the foreign-fill change
    # (borrow unscored passages from other queries' pools), not a bigger corpus.
    Task("rerank", "rerank", max_rung="32k",
         max_rung_reason="msmarco.load_pool defaults max_docs=250 -- exactly the 32k rung -- and "
                         "the per-query fill is drawn and CE-scored at EXPORT time, so the seed "
                         "pool physically holds ~250 candidates per query. Not a corpus bound"),
    # Synthetic: no corpus, so --pool is rejected outright. Ceiling is the frozen 20,045-word
    # vocabulary -- every non-planted word is unique WITHIN an example, which is what makes the
    # planted pairs the only ones meeting the criterion by construction rather than by a check.
    Task("strmatch", "strmatch", pooled=False, max_rung="48k",
         max_rung_reason="the frozen 20,045-word vocabulary caps ~1.9k documents at ~9.8 words "
                         "each; every non-planted word is unique WITHIN an example, which is what "
                         "makes the planted pairs the only ones meeting the criterion"),
    # A ceiling, NOT an enumerated withhold list: the bias is roughly flat in absolute terms while
    # the chance baseline collapses as documents multiply, so every rung ABOVE the measured ones is
    # worse, not unknown. Enumerating {8k,16k,32k} re-admitted it at 64k+ the moment the ladder grew.
    Task("textgroups", "textgroups", pooled=False, max_rung="4k",
         max_rung_reason=_TEXTGROUPS_SHORTCUT),
    # The suite's ctc_grouping row: UNLABELED OpenAlex partition (k per the concept level, up to
    # ~1 doc per group at the finest), graded by the `grouping` spec. grouping_labeled -- ctc's only
    # grouping generator -- asks for coarse named groups instead, so it is not this task. Built by
    # the generator the eval ladder came from, on papers before its 2024 eval cut-off.
    Task("grouping", "grouping", external="openalex_grouping"),
]

#: CTC-BENCH-10 with the substitutions that keep the FiQA corpus out of training (rerank for fiqa,
#: reorder for qdmatch_fiqa), so both stay clean held-out probes.
SET_B: List[Task] = [t for t in SET_A if t.name in {
    "nq", "hotpotqa", "rerank", "oolong", "reorder", "qdmatch_nq", "outlier", "xabsence",
    "contradiction"}]

SETS = {"set-a": (SET_A, "setA_max20"), "set-b": (SET_B, "setB_bench10")}


# ---------------------------------------------------------------- provenance


def _run(cmd: List[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _git_commit(repo: str) -> Optional[str]:
    p = _run(["git", "-C", repo, "rev-parse", "HEAD"])
    return p.stdout.strip() if p.returncode == 0 else None


def ctc_provenance() -> Dict[str, Optional[str]]:
    """:returns: Version + resolved source commit of the installed ``ctc``, for the manifest."""
    out: Dict[str, Optional[str]] = {"version": None, "commit": None, "url": None}
    try:
        from importlib import metadata

        dist = metadata.distribution("ctc")
        out["version"] = dist.version
        raw = dist.read_text("direct_url.json")
        if raw:
            d = json.loads(raw)
            out["url"] = d.get("url")
            out["commit"] = (d.get("vcs_info") or {}).get("commit_id")
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never a build failure
        pass
    return out


def synthetic_tasks(names: List[str]) -> List[str]:
    """:returns: Which of ``names`` have no corpus, asked of ctc rather than hardcoded."""
    from ctc.data.generators import base as gens
    from ctc.tasks import load_all

    load_all()
    return [n for n in names if gens.get(n).corpus is None]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- the stages


#: External generators: ``name -> (pool file under --external-pools, command builder)``. The command
#: builder gets ``(repo, pool, n_docs, rows, seed, out_dir)`` and returns ``(argv, produced path)``.
def _xabsence_cmd(repo, pool, n_docs, rows, seed, out_dir):
    pairs = max(1, (n_docs - 3) // 2)  # one-sided: A holds P + 3 orphans, B holds P
    argv = [os.path.join(repo, "src", "corpus_reasoning", "data", "generate_xabsence_data.py"),
            "--pool", pool, "--num-pairs", str(pairs), "--num-unmatched", "3",
            "--orphan-side", "A", "--src-tag", "gutenberg", "--num-train", str(rows),
            "--num-eval", "0", "--output-dir", out_dir, "--seed", str(seed)]
    return argv, os.path.join(out_dir, f"xabsence_train_gutenberg_p{pairs}_k3.jsonl")


def _grouping_cmd(repo, pool, n_docs, rows, seed, out_dir):
    argv = [os.path.join(repo, "src", "corpus_reasoning", "data",
                         "generate_arxiv_grouping_data.py"),
            "--compact-in", pool, "--num-train", str(rows), "--num-eval", "0",
            "--docs-per-example", str(n_docs), "--out-dir", out_dir, "--seed", str(seed)]
    return argv, os.path.join(
        out_dir, f"openalex_grouping_n{n_docs}_levels_train_{rows}.jsonl")


EXTERNAL = {
    "xabsence_gutenberg_onesided": ("pool_gutenberg_train.jsonl", _xabsence_cmd),
    "openalex_grouping": ("openalex_compact.jsonl", _grouping_cmd),
}

#: ``(documents, rendered Qwen3.5 tokens)`` of the eval's top-rung rows, for extrapolating buckets
#: the eval does not have. Scaled by MEASURED tokens, not by rung label: grouping's "32k" rows are
#: 35.6k tokens, so label-scaling would put its 256k rows at ~285k, past the window.
EXTERNAL_TOP = {"xabsence": (669, 28_588), "grouping": (176, 35_636)}
#: tokens an extrapolated row may use (the 262,144 window less chat wrapper + boundary markers)
_FIT_TOKENS = 250_000


def _external_n_docs(task: Task, bucket: str, cal: Optional[dict]) -> int:
    if cal:
        return int(cal["n_docs"])
    n, tokens = EXTERNAL_TOP[task.name]
    return int(n * min(BUCKET_TOKENS[bucket], _FIT_TOKENS) / tokens)


def _bucket_allowed(task: Task, bucket: str, cal: Optional[dict] = None) -> Optional[str]:
    """:returns: ``None`` if the bucket should be built, else the reason it is skipped."""
    if bucket in task.withhold:
        return f"withheld: {task.withhold[bucket]}"
    # A foreign-filled cell asks ctc for the fixed scored set only, so a ceiling set by the
    # per-query candidate pool (rerank's 250) does not bound it.
    if cal and "fill_to" in cal:
        return None
    if task.max_rung and BUCKET_TOKENS[bucket] > rung_tokens(task.max_rung):
        why = f"above the task ceiling ({task.max_rung})"
        return f"{why}: {task.max_rung_reason}" if task.max_rung_reason else why
    return None


def _build_cell(job) -> dict:
    task, bucket, n, build_root, seed, python, cal, ext_pools, top_cal = job
    out = os.path.join(build_root, f"_b{bucket}")
    target = os.path.join(out, task.name, "train.jsonl")
    rec = {"task": task.name, "bucket": bucket, "want": n, "path": target}
    # A cell is reused only if it was built the same way. External cells always carry a stamp; a
    # ctc cell without one predates stamping and is accepted, but a ctc cell is never reused for an
    # external task (xabsence switched generators under the same name and path).
    stamp_path = target + ".how.json"
    how = {"generator": task.external or "ctc", "n_docs": (cal or {}).get("n_docs"),
           "set": (cal or {}).get("set"), "gen_args": (cal or {}).get("gen_args")}
    if os.path.exists(target) and os.path.getsize(target) > 0:
        prev = json.load(open(stamp_path)) if os.path.exists(stamp_path) else None
        if prev == how or (prev is None and not task.external):
            rec.update(status="cached", rows=sum(1 for _ in open(target)))
            return rec
        os.remove(target)
    if task.external:
        pool_name, builder = EXTERNAL[task.external]
        pool = os.path.join(ext_pools or "", pool_name)
        if not os.path.exists(pool):
            rec.update(status="FAILED", error=f"external pool missing: {pool} (--external-pools)")
            return rec
        n_docs = _external_n_docs(task, bucket, cal)
        tmp = os.path.join(out, task.name, "_gen")
        os.makedirs(tmp, exist_ok=True)
        repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            *[os.pardir] * 4))
        # per-bucket seed so buckets draw different examples, reproducibly
        argv, produced = builder(repo, pool, n_docs, n, seed * 1000 + BUCKET_TOKENS[bucket] // 1024,
                                 tmp)
        # generator settings measured from the eval (level quotas, k ranges); a bucket above the
        # eval's top rung borrows the top rung's
        argv += (cal or {}).get("gen_args") or (top_cal or {}).get("gen_args") or []
        rec["gen_args"] = argv[1:]
        p = _run([python] + argv)
        if p.returncode != 0 or not os.path.exists(produced):
            tail = (p.stdout + p.stderr).strip().splitlines()
            rec.update(status="FAILED", error=" | ".join(tail[-3:])[:600])
            return rec
        os.replace(produced, target)
        json.dump(how, open(stamp_path, "w"))
        rec.update(status="built", rows=sum(1 for _ in open(target)), n_docs=n_docs,
                   generator=task.external)
        return rec
    env = None
    if cal:
        # sizes measured from the eval itself (measure_eval_ladder.py), applied by patching
        # ctc's docs_for_rung for this one cell; generation is otherwise ctc as published
        entry = [python, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "ctc_calibrated.py")]
        env = {**os.environ,
               "CTC_DOCS_OVERRIDE": json.dumps({f"{task.name}:{bucket}": cal["n_docs"]})}
        rec["n_docs"] = cal["n_docs"]
    else:
        entry = [python, "-m", "ctc.data.cli"]
    cmd = entry + ["build", "--task", task.name, "--split", "train",
                   "--rungs", bucket, "--train", str(n), "--seed", str(seed), "--out", out]
    for kv in (cal or {}).get("set", []):
        cmd += ["-C", kv]
    if task.pooled:
        cmd += ["--pool", "auto"]
    p = _run(cmd, env=env)
    if p.returncode != 0 or not os.path.exists(target):
        tail = (p.stdout + p.stderr).strip().splitlines()
        rec.update(status="FAILED", error=" | ".join(tail[-3:])[:600])
        return rec
    json.dump(how, open(stamp_path, "w"))
    rec.update(status="built", rows=sum(1 for _ in open(target)))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_name", required=True, choices=sorted(SETS))
    ap.add_argument("--out", required=True, help="root; the set writes to <out>/<tag>/")
    ap.add_argument("--buckets", nargs="+", default=["2k", "4k", "8k", "16k", "32k"])
    ap.add_argument("--tokens-per-bucket", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=42, help="pinned; ctc keys streams by (seed,split,rung)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--convert", action="store_true", help="also tokenize to docchunk shards")
    ap.add_argument("--marker-set", default="qwen3_5")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--seq-len", type=int, default=40960)
    ap.add_argument("--shards-dir", default="",
                    help="shard dir name under the set root; default shards_<marker-set>. Name it "
                         "when converting at a different --seq-len: the trainer REFUSES a --seq-len "
                         "below a shard's max_example_len, so one dir cannot serve both windows.")
    ap.add_argument("--query-position", default="both", choices=("before", "after", "both"))
    ap.add_argument("--cot-mode", default="none")
    ap.add_argument("--calibration", default="",
                    help="eval_calibration.json from measure_eval_ladder.py: size every "
                         "(task, bucket) it names to the EVAL's own rows instead of ctc's ladder "
                         "table. Builds into <set tag>_evaliid so no table-sized cell is reused.")
    ap.add_argument("--external-pools", default="",
                    help="dir holding the train pools of the external generators "
                         f"({', '.join(v[0] for v in EXTERNAL.values())})")
    ap.add_argument("--check", action="store_true",
                    help="re-hash an existing build against its manifest and report drift")
    # this file is <repo>/src/scripts/data/ctc_sft/build_ctc_sft.py -> five levels up
    ap.add_argument("--repo", default=os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 4)))
    args = ap.parse_args()
    if args.convert and not os.path.exists(os.path.join(
            args.repo, "src", "scripts", "data", "convert_unified_to_document_landmark.py")):
        raise SystemExit(f"--repo {args.repo} has no src/scripts/data/convert_unified_to_document_landmark.py")

    for b in args.buckets:
        if b not in BUCKET_TOKENS:
            raise SystemExit(f"unknown bucket {b!r}; have {', '.join(BUCKET_TOKENS)}")

    tasks, tag = SETS[args.set_name]
    calibration: Dict[str, Dict[str, dict]] = {}
    if args.calibration:
        calibration = json.load(open(args.calibration))["tasks"]
        tag = f"{tag}_evaliid"
    root = os.path.join(args.out, tag)
    build_root = os.path.join(root, "per_task")
    manifest_path = os.path.join(root, "manifest.json")
    python = sys.executable

    if args.check:
        return _check(manifest_path)

    os.makedirs(build_root, exist_ok=True)

    # The roster declares which generators are synthetic; ctc is the authority. Drift between them
    # is exactly the bug that cost 10 builds, so it is a hard failure, not a warning.
    declared = {t.name for t in tasks if not t.pooled and not t.external}
    actual = set(synthetic_tasks([t.name for t in tasks if not t.external]))
    if declared != actual:
        raise SystemExit(
            f"roster/ctc disagree on which tasks are synthetic.\n"
            f"  roster says: {sorted(declared) or '(none)'}\n"
            f"  ctc says:    {sorted(actual) or '(none)'}\n"
            f"A synthetic task REJECTS --pool and loses every bucket. Fix `pooled=` in SETS.")

    jobs, skipped = [], []
    for t in tasks:
        for b in args.buckets:
            why = _bucket_allowed(t, b, calibration.get(t.name, {}).get(b))
            if why:
                skipped.append({"task": t.name, "bucket": b, "reason": why})
                continue
            # A bucket the eval has no rung for keeps ctc's own table: nothing grades it, so there
            # is nothing to match, and dropping it would quietly shrink the long-context mix.
            cal = calibration.get(t.name, {}).get(b)
            task_cal = calibration.get(t.name, {})
            top_cal = task_cal[max(task_cal, key=lambda r: BUCKET_TOKENS[r])] if task_cal else None
            jobs.append((t, b, args.tokens_per_bucket // BUCKET_TOKENS[b], build_root,
                         args.seed, python, cal, args.external_pools, top_cal))

    print(f"=== {tag} | {len(tasks)} tasks x {len(args.buckets)} buckets "
          f"| {len(jobs)} to build, {len(skipped)} skipped | seed={args.seed} "
          f"| -P {args.jobs} ===", flush=True)
    for s in skipped:
        print(f"  [skip] {s['task']}@{s['bucket']}: {s['reason']}", flush=True)

    # `--pool auto` downloads a pool per task; 65 cold processes would fetch each many times over.
    # One serial pass at the cheapest bucket warms the cache, then the rest runs wide against it.
    first = args.buckets[0]
    warm = [j for j in jobs if j[1] == first]
    print(f"=== warming seed-pool cache ({len(warm)} serial) ===", flush=True)
    results = [_build_cell(j) for j in warm]
    for r in results:
        print(f"  [{r['status']}] {r['task']}@{r['bucket']} rows={r.get('rows', 0)}"
              + (f"  {r.get('error', '')}" if r["status"] == "FAILED" else ""), flush=True)

    rest = [j for j in jobs if j[1] != first]
    print(f"=== building {len(rest)}, {args.jobs}-way parallel ===", flush=True)
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, r in enumerate(pool.map(_build_cell, rest), 1):
            results.append(r)
            if i <= 2 or i % 5 == 0 or i == len(rest):
                el = time.time() - t0
                print(f"  [{i}/{len(rest)}] {r['task']}@{r['bucket']} {r['status']} "
                      f"rows={r.get('rows', 0)} | {el:.0f}s, ETA {el / i * (len(rest) - i):.0f}s",
                      flush=True)
            if r["status"] == "FAILED":
                print(f"    FAILED {r['task']}@{r['bucket']}: {r.get('error')}", flush=True)

    failed = [r for r in results if r["status"] == "FAILED"]

    for t in tasks:
        fills = {b: c["fill_to"] for b, c in calibration.get(t.name, {}).items()
                 if "fill_to" in c and b in args.buckets}
        if fills:
            _foreign_fill(t.name, fills, build_root, args.seed)

    print("=== merging buckets per task ===", flush=True)
    per_task = {}
    for t in tasks:
        d = os.path.join(build_root, t.name)
        os.makedirs(d, exist_ok=True)
        merged, rows = os.path.join(d, "train.jsonl"), 0
        with open(merged, "w") as out:
            for b in args.buckets:
                src = os.path.join(build_root, f"_b{b}", t.name, "train.jsonl")
                if os.path.exists(src) and os.path.getsize(src):
                    with open(src) as f:
                        for line in f:
                            out.write(line)
                            rows += 1
        got = [b for b in args.buckets
               if os.path.exists(os.path.join(build_root, f"_b{b}", t.name, "train.jsonl"))]
        per_task[t.name] = {"rows": rows, "buckets": got, "sha256": sha256(merged) if rows else None}
        print(f"  {t.name}: {rows:,} rows over {len(got)} buckets ({','.join(got)})", flush=True)

    empty = [n for n, v in per_task.items() if v["rows"] == 0]

    shards = {}
    if args.convert:
        shards = _convert(tasks, root, build_root, args)

    manifest = {
        "calibration": ({"path": os.path.abspath(args.calibration), "cells": calibration}
                        if args.calibration else None),
        "set": args.set_name, "tag": tag, "root": root,
        "seed": args.seed, "buckets": args.buckets,
        "tokens_per_bucket": args.tokens_per_bucket,
        "olmo_core_commit": _git_commit(args.repo),
        "ctc": ctc_provenance(),
        "roster": [dataclasses.asdict(t) for t in tasks],
        "skipped": skipped, "failed": failed, "per_task": per_task, "shards": shards,
        "convert": ({"marker_set": args.marker_set, "tokenizer": args.tokenizer,
                     "seq_len": args.seq_len, "query_position": args.query_position,
                     "cot_mode": args.cot_mode} if args.convert else None),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest -> {manifest_path}", flush=True)

    print(f"\n=== {tag}: {len(per_task) - len(empty)}/{len(tasks)} tasks, "
          f"{sum(v['rows'] for v in per_task.values()):,} rows ===", flush=True)
    bad_shards = sorted(n for n, v in shards.items() if v.get("status") == "FAILED")
    if failed or empty or bad_shards:
        for n in bad_shards:
            print(f"  SHARD FAILED {n}: {shards[n].get('error')}", flush=True)
        for r in failed:
            print(f"  FAILED {r['task']}@{r['bucket']}: {r.get('error')}", flush=True)
        for n in empty:
            print(f"  EMPTY  {n}: no bucket produced rows", flush=True)
        raise SystemExit(1)


def _foreign_fill(task: str, fills: Dict[str, int], build_root: str, seed: int) -> None:
    """
    Pad each row's CE-scored candidate set to the eval's document count with unscored passages.

    This is how the eval's rerank rows are built: a fixed set of CE-scored candidates at every rung
    plus foreign passages with ``ce_scores`` None, which the scorer excludes from the target. The
    fill is drawn from OTHER queries' candidates across every bucket of this build (the eval drew
    random MS MARCO passages; both are unrelated to the query), inserted at uniformly random
    positions, with gold / hard-negative / CE indices remapped. ``ctc``'s own output is kept as
    ``train.ctc.jsonl`` and ``train.jsonl`` is always regenerated from it, so this is idempotent.
    """
    import random

    raws = {}
    for b in fills:
        d = os.path.join(build_root, f"_b{b}", task)
        raw, out = os.path.join(d, "train.ctc.jsonl"), os.path.join(d, "train.jsonl")
        if not os.path.exists(raw) and os.path.exists(out):
            os.replace(out, raw)
        if os.path.exists(raw):
            raws[b] = raw
    donors: List[tuple] = []  # (query, text)
    seen = set()
    for raw in raws.values():
        with open(raw) as f:
            for line in f:
                ex = json.loads(line)
                q = (ex.get("queries") or [""])[0]
                for doc in ex["documents"]:
                    key = doc.get("text", "")
                    if key not in seen:
                        seen.add(key)
                        donors.append((q, doc))
    print(f"=== foreign fill {task}: {len(donors):,} donor passages ===", flush=True)
    for b, raw in sorted(raws.items(), key=lambda kv: BUCKET_TOKENS[kv[0]]):
        rng = random.Random(f"{seed}:foreign_fill:{task}:{b}")
        target = fills[b]
        rows = short = unscorable = 0
        with open(raw) as f, open(os.path.join(os.path.dirname(raw), "train.jsonl"), "w") as out:
            for line in f:
                ex = json.loads(line)
                # The eval grades rerank on ce_pos_recall: the share of CE-POSITIVE documents in the
                # first 10 ids. A row with no CE-positive candidate scores 0 for every answer, and
                # the eval's rows never have one -- so it is filtered out, not trained on.
                if not any(c is not None and c > 0 for c in (ex.get("ce_scores") or [])):
                    unscorable += 1
                    continue
                docs = list(ex["documents"])
                own = {d.get("text", "") for d in docs}
                q = (ex.get("queries") or [""])[0]
                need = target - len(docs)
                fill = []
                while len(fill) < need:
                    dq, doc = donors[rng.randrange(len(donors))]
                    if dq != q and doc.get("text", "") not in own:
                        own.add(doc.get("text", ""))
                        fill.append(doc)
                n = len(docs) + len(fill)
                slots = sorted(rng.sample(range(n), len(fill)))  # where fill lands
                order, it_own, it_fill = [], iter(range(len(docs))), iter(range(len(fill)))
                fill_slots = set(slots)
                for pos in range(n):
                    order.append(("f", next(it_fill)) if pos in fill_slots else ("o", next(it_own)))
                remap = {i: pos for pos, (kind, i) in enumerate(order) if kind == "o"}
                ce = ex.get("ce_scores") or [None] * len(docs)
                ex["documents"] = [docs[i] if k == "o" else fill[i] for k, i in order]
                ex["ce_scores"] = [ce[i] if k == "o" else None for k, i in order]
                for key in ("gold_doc_indices", "hard_neg_indices"):
                    if ex.get(key) is not None:
                        ex[key] = [remap[i] for i in ex[key]]
                meta = dict(ex.get("_meta") or {})
                meta.update(foreign_fill=len(fill), scored=len(docs))
                ex["_meta"] = meta
                out.write(json.dumps(ex) + "\n")
                rows += 1
                short += need < 0
        print(f"  {task}@{b}: {rows} rows filled to {target} docs"
              + (f"; dropped {unscorable} with no CE-positive document" if unscorable else "")
              + (f"  ⚠ {short} rows already above target" if short else ""), flush=True)


def _convert(tasks, root, build_root, args) -> dict:
    conv = os.path.join(args.repo, "src", "scripts", "data",
                        "convert_unified_to_document_landmark.py")
    out_root = os.path.join(root, args.shards_dir or f"shards_{args.marker_set}")
    os.makedirs(out_root, exist_ok=True)
    print(f"=== tokenizing -> {out_root} (parallel) ===", flush=True)
    jobs = []
    shards = {}
    for t in tasks:
        src = os.path.join(build_root, t.name, "train.jsonl")
        if not os.path.exists(src) or not os.path.getsize(src):
            shards[t.name] = {"status": "SKIPPED_EMPTY"}
            continue
        jobs.append((t, src, os.path.join(out_root, t.name)))
    with cf.ProcessPoolExecutor(max_workers=min(len(jobs) or 1, args.jobs)) as pool:
        for name, rec in pool.map(_convert_one, [(t, src, d, conv, args) for t, src, d in jobs]):
            shards[name] = rec
            if rec["status"] == "ok":
                warn = (f"  ⚠ {rec['num_dropped']:,} dropped > seq-len {args.seq_len}"
                        if rec["num_dropped"] else "")
                print(f"  {name}: {rec['num_instances']:,} inst, {rec['num_tokens']:,} tok{warn}",
                      flush=True)
            else:
                print(f"  [{rec['status']}] {name}: {rec.get('error','')}", flush=True)
    ok = [v for v in shards.values() if v.get("status") == "ok"]
    print(f"  TOTAL {sum(v['num_instances'] for v in ok):,} instances, "
          f"{sum(v['num_tokens'] for v in ok):,} tokens", flush=True)
    return shards


def _convert_one(job):
    """One task -> one shard dir. Module-level so it is picklable for the pool."""
    t, src, d, conv, args = job
    # --emit dense serves BOTH arms: `--variant full` reads these ids directly and
    # `--variant sparselandmark` inserts landmarks at LOAD time via LandmarkPackingInstanceSource.
    p = _run([sys.executable, conv, "--emit", "dense", "--task", t.spec,
              "--chunk-by", t.chunk_by, "--marker-set", args.marker_set,
              "--tokenizer", args.tokenizer, "--query-position", args.query_position,
              "--cot-mode", args.cot_mode, "--seq-len", str(args.seq_len),
              "--input-jsonl", src, "--out-dir", d])
    meta_path = os.path.join(d, "metadata.json")
    if p.returncode != 0 or not os.path.exists(meta_path):
        tail = (p.stdout + p.stderr).strip().splitlines()
        return t.name, {"status": "FAILED", "error": " | ".join(tail[-3:])[:600]}
    m = json.load(open(meta_path))
    return t.name, {"status": "ok", "dir": d, "num_instances": m["num_instances"],
                    "num_dropped": m["num_dropped"], "num_tokens": m["num_tokens"],
                    "max_example_len": m["max_example_len"],
                    "num_loss_tokens": m["num_loss_tokens"]}


def _check(manifest_path: str) -> None:
    """Re-hash a previous build against its manifest; drift is a non-zero exit."""
    if not os.path.exists(manifest_path):
        raise SystemExit(f"no manifest at {manifest_path} -- nothing to check")
    m = json.load(open(manifest_path))
    print(f"=== checking {m['tag']} (seed={m['seed']}, ctc={m['ctc'].get('commit')}) ===", flush=True)
    bad = 0
    for name, rec in sorted(m["per_task"].items()):
        path = os.path.join(m["root"], "per_task", name, "train.jsonl")
        if rec["sha256"] is None:
            print(f"  {name:<18} (empty in manifest)", flush=True)
            continue
        if not os.path.exists(path):
            print(f"  {name:<18} MISSING", flush=True); bad += 1; continue
        got = sha256(path)
        if got == rec["sha256"]:
            print(f"  {name:<18} ok  {rec['rows']:,} rows", flush=True)
        else:
            print(f"  {name:<18} DRIFT  manifest={rec['sha256'][:12]} disk={got[:12]}", flush=True)
            bad += 1
    print(f"\n{'DRIFT in ' + str(bad) + ' task(s)' if bad else 'all tasks reproduce'}", flush=True)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
