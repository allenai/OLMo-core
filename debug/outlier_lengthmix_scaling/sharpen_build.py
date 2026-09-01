"""Sharpen-wave arm builder: short-heavy mixes for qdmatch_nq + nq, plus pure-length gap fillers.

Runs on mooney inside sharpen_arms.sbatch. Builds, per task:

  qdmatch (/data/prasann/qdmatch_lengthmix): extends the q9 (2k) pool 25,300 -> 110,300 by
    re-emitting with its original seed 201 (sequential-rng determinism preserves the old prefix;
    guarded by sha256 before overwrite, exactly like debug_build_q32k_big.py), then composes
    qmix_s{64,160,320}M short-heavy mixes and the q32k_16000 ladder point.
  nq (/data/prasann/nq_lengthmix): NO pool changes -- build_nq_pools.py is stream-based (one
    example per source row) so its pools are source-capped at ~19,667; the nq mixes are sized to
    fit (nmix_s{16,32,48}M, nq's much-easier regime) plus nqD32k_4000 / nqD64k_2000 ladder points.
  outlier (/data/prasann/outlier_lengthmix): p32k_4000 mid-point (the 32k K estimate currently
    rests on exactly two points, 2000 and 8000).

Short-heavy token shares are the outlier recipe renormalized to the rungs these tasks have pools
for ({2k:45,8k:16,16k:8,32k:4} -> 61.6/21.9/11.0/5.5%). Example counts below are
round(share*B/tokens_per_example) with rates measured from the arms' own tokenize metadata.

All arms are nested prefixes of their pool train files, shuffled with the shared seed 7113 --
byte-compatible with compose_arms.py / compose_qdmatch_arms.py / compose_nq_arms.py output.
"""
import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys
import time

GENSRC = "/data/prasann/repo/OLMo-core/src"
DEBUG_DIR = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/outlier_lengthmix_scaling"
sys.path.insert(0, DEBUG_DIR)
sys.path.insert(0, GENSRC)

QW = pathlib.Path("/data/prasann/qdmatch_lengthmix")
WN = pathlib.Path("/data/prasann/nq_lengthmix")
WO = pathlib.Path("/data/prasann/outlier_lengthmix")
SHUFFLE_SEED = 7113
HELDOUT = 300
Q9_POOL_COUNT = 110300      # supports the 320M mix's 107,842-example 2k component
Q9_SEED = 201               # q9 pool seed in build_qdmatch_pools.py specs

# arm -> ordered (train-file, count) parts, short rung first (compose_* iteration order)
QD_ARMS = {
    "qmix_s64M":  [("q9", 21568), ("q42", 1851), ("q16k", 460), ("q32k", 116)],
    "qmix_s160M": [("q9", 53921), ("q42", 4627), ("q16k", 1149), ("q32k", 289)],
    "qmix_s320M": [("q9", 107842), ("q42", 9254), ("q16k", 2298), ("q32k", 578)],
    "q32k_16000": [("q32k", 16000)],
}
NQ_ARMS = {
    "nmix_s16M": [("n2k", 5109), ("n8k", 479), ("n16k", 133), ("nD32k", 28)],
    "nmix_s32M": [("n2k", 10218), ("n8k", 958), ("n16k", 266), ("nD32k", 56)],
    "nmix_s48M": [("n2k", 15327), ("n8k", 1438), ("n16k", 400), ("nD32k", 84)],
    "nqD32k_4000": [("nD32k", 4000)],
    "nqD64k_2000": [("nD64k", 2000)],
}
OUT_ARMS = {"p32k_4000": [("n220", 4000)]}


def log(m):
    print(f"[sharpen {time.strftime('%H:%M:%S')}] {m}", flush=True)


def extend_q9_pool():
    import build_qdmatch_pools as B
    pool = QW / "qdmatch_nq_q9_pool.jsonl"
    train = QW / "qdmatch_nq_q9_train.jsonl"
    n_pool = sum(1 for _ in open(pool)) if pool.exists() else 0
    if n_pool != Q9_POOL_COUNT:
        units, _q, audit = B.load_units(str(QW / "src/nq_train_k25-202_clean.jsonl"), "train-src")
        assert 0.05 <= audit["hard_neg_ratio"] <= 0.20, "train-src not in the p10 regime"
        B.emit(units, 9, 9, Q9_POOL_COUNT, Q9_SEED, pool, "qdmatch_nq")
    lines = pool.read_text().splitlines()
    assert len(lines) == Q9_POOL_COUNT
    if train.exists():
        old = train.read_text().splitlines()
        k = min(len(old), len(lines) - HELDOUT)
        assert (hashlib.sha256("\n".join(old[:k]).encode()).hexdigest()
                == hashlib.sha256("\n".join(lines[:k]).encode()).hexdigest()), \
            "q9 PREFIX MISMATCH -- refusing to overwrite (q2k_* arms would silently change)"
        log(f"q9 prefix guard PASSED over {k} lines")
    train.write_text("\n".join(lines[:-HELDOUT]) + "\n")
    (QW / "qdmatch_nq_q9_heldout.jsonl").write_text("\n".join(lines[-HELDOUT:]) + "\n")
    log(f"q9 pool extended: train={len(lines) - HELDOUT} heldout={HELDOUT}")


def train_file(work, tag):
    if work == QW:
        return work / f"qdmatch_nq_{tag}_train.jsonl"
    if work == WN:
        return work / f"nq_{tag}_train.jsonl"
    return work / f"outlier_lm_{tag}_train.jsonl"


def compose(work, arm, parts, pools_cache):
    out = work / "arms" / f"{arm}.jsonl"
    if out.exists() and sum(1 for _ in open(out)) == sum(c for _, c in parts):
        log(f"[skip] compose {arm}")
        return
    lines = []
    for tag, cnt in parts:
        if tag not in pools_cache:
            pools_cache[tag] = train_file(work, tag).read_text().splitlines()
            log(f"pool {tag}: {len(pools_cache[tag])} examples")
        assert len(pools_cache[tag]) >= cnt, f"{arm}: pool {tag} has {len(pools_cache[tag])} < {cnt}"
        lines += pools_cache[tag][:cnt]
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(lines)
    (work / "arms").mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    mpath = work / "arms" / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest[arm] = {"spec": {t: c for t, c in parts}, "n_examples": len(lines),
                     "shuffle_seed": SHUFFLE_SEED,
                     "composition": "nested prefixes of " + ", ".join(
                         train_file(work, t).name for t, _ in parts),
                     "recipe": "short-heavy 61.6/21.9/11.0/5.5 over 2k/8k/16k/32k" if "mix" in arm
                               else "pure-length ladder point"}
    mpath.write_text(json.dumps(manifest, indent=2))
    log(f"arm {arm}: {len(lines)} examples composed")


def tokenize(work, arm, task, maxseq, qpos):
    out = work / "arms_tokenized" / arm
    if (out / "metadata.json").exists():
        log(f"[skip] tok {arm}")
        return
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, f"{GENSRC}/corpus_reasoning/data/convert_unified_to_sft.py",
           "--task", task, "--input", str(work / "arms" / f"{arm}.jsonl"),
           "--out-dir", str(out), "--tokenizer", "Qwen/Qwen3.5-0.8B-Base",
           "--max-seq-len", str(maxseq), "--eos", "248044"]
    if qpos:
        cmd += ["--query-position", qpos]
    log(f"tokenizing {arm} ...")
    r = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": GENSRC})
    assert r.returncode == 0 and (out / "metadata.json").exists(), f"tokenize failed: {arm}"
    log(f"tok {arm} DONE")


def main():
    extend_q9_pool()
    cache_q, cache_n, cache_o = {}, {}, {}
    for arm, parts in QD_ARMS.items():
        compose(QW, arm, parts, cache_q)
    for arm, parts in NQ_ARMS.items():
        compose(WN, arm, parts, cache_n)
    for arm, parts in OUT_ARMS.items():
        compose(WO, arm, parts, cache_o)
    del cache_q, cache_n, cache_o
    for arm in QD_ARMS:
        tokenize(QW, arm, "qdmatch", 40960, None)
    for arm in NQ_ARMS:
        tokenize(WN, arm, "retrieval", 72000, "after")
    for arm in OUT_ARMS:
        tokenize(WO, arm, "outlier", 40960, "after")
    log("ALL DONE")


if __name__ == "__main__":
    main()
