"""Build the q32k_32000 qdmatch arm end-to-end (pool extend -> compose -> tokenize).

This is the escape hatch the certainty-wave bigmix sbatch tries FIRST
(/data/prasann/repo/OLMo-core/debug_build_q32k_big.py): its fallback invocation of
build_qdmatch_pools.py uses flags that do not exist (--n32k 172x172 is type=int, --eval-src is
required, --only takes tags like 'q32k' not '32k'), so this script must succeed.

Extends the q32k (M=N=172, seed 204) pool from 8300 -> 32300 by re-emitting with the same seed:
emit() consumes one rng sequentially, so the first 8300 examples are bit-identical to the old
pool and the nested-prefix property of q32k_2000/q32k_8000 is preserved (guarded by an explicit
hash check against the old train file before anything is overwritten). NOTE the old 300-example
heldout (pool lines 8000-8300) lands INSIDE the new 32000-line train prefix; the graded eval
rung (eval_rungs/qdmatch_nq/rung_32768.jsonl, built from the disjoint validation units) is
unaffected, but the q32k heldout-CE metric is contaminated for this arm. New heldout = pool
lines 32000-32300.

Idempotent: exits 0 immediately if the tokenized arm's metadata.json exists.
"""
import hashlib
import json
import pathlib
import random
import subprocess
import sys
import time

QW = pathlib.Path("/data/prasann/qdmatch_lengthmix")
GENSRC = "/data/prasann/repo/OLMo-core/src"
DEBUG_DIR = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/outlier_lengthmix_scaling"
M = N = 172
POOL_SEED = 204          # q32k pool seed in build_qdmatch_pools.py specs
SHUFFLE_SEED = 7113      # arm shuffle seed shared with compose_qdmatch_arms.py
POOL_COUNT = 32300
HELDOUT = 300
ARM = "q32k_32000"

sys.path.insert(0, DEBUG_DIR)
sys.path.insert(0, GENSRC)
import build_qdmatch_pools as B  # noqa: E402


def log(m):
    print(f"[q32kbig {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    tok_out = QW / "arms_tokenized" / ARM
    if (tok_out / "metadata.json").exists():
        log("tokenized arm already present -- DONE (idempotent skip)")
        return

    pool = QW / "qdmatch_nq_q32k_pool.jsonl"
    train = QW / "qdmatch_nq_q32k_train.jsonl"
    heldout = QW / "qdmatch_nq_q32k_heldout.jsonl"

    # 1. pool: re-emit at 32300 with the original seed unless already extended
    n_pool = sum(1 for _ in open(pool)) if pool.exists() else 0
    if n_pool != POOL_COUNT:
        units, _q, audit = B.load_units(str(QW / "src/nq_train_k25-202_clean.jsonl"), "train-src")
        hr = audit["hard_neg_ratio"]
        assert 0.05 <= hr <= 0.20, f"train-src hard_neg_ratio={hr} is not the p10 regime"
        B.emit(units, M, N, POOL_COUNT, POOL_SEED, pool, "qdmatch_nq")
    else:
        log(f"pool already has {n_pool} lines")

    lines = pool.read_text().splitlines()
    assert len(lines) == POOL_COUNT, f"pool has {len(lines)} != {POOL_COUNT}"

    # 2. nested-prefix guard BEFORE overwriting the old train file
    if train.exists():
        old = train.read_text().splitlines()
        k = min(len(old), len(lines) - HELDOUT)
        h_old = hashlib.sha256("\n".join(old[:k]).encode()).hexdigest()
        h_new = hashlib.sha256("\n".join(lines[:k]).encode()).hexdigest()
        assert h_old == h_new, (
            "PREFIX MISMATCH: re-emitted pool does not reproduce the old train prefix -- "
            "refusing to overwrite (q32k_2000/q32k_8000 arms would silently change)")
        log(f"prefix guard PASSED over {k} lines")

    train.write_text("\n".join(lines[:-HELDOUT]) + "\n")
    heldout.write_text("\n".join(lines[-HELDOUT:]) + "\n")
    log(f"split: train={len(lines) - HELDOUT} heldout={HELDOUT}")

    # 3. compose the arm exactly as compose_qdmatch_arms.py would
    arm_lines = lines[: POOL_COUNT - HELDOUT]
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(arm_lines)
    arms_dir = QW / "arms"
    arms_dir.mkdir(exist_ok=True)
    (arms_dir / f"{ARM}.jsonl").write_text("\n".join(arm_lines) + "\n")
    mpath = arms_dir / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest[ARM] = {"spec": {"172": 32000}, "n_examples": len(arm_lines),
                     "shuffle_seed": SHUFFLE_SEED, "task": "qdmatch", "source": "qdmatch_nq",
                     "shape": {"172": {"M": M, "N": N}},
                     "composition": "nested prefixes of qdmatch_nq_q32k_train.jsonl"}
    mpath.write_text(json.dumps(manifest, indent=2))
    log(f"arm {ARM}: {len(arm_lines)} examples composed")

    # 4. tokenize with the exact recipe of the earlier qdmatch arms (build_qdmatch_lengthmix.sbatch
    # tok(): no --query-position flag, same tokenizer/seq-len/eos)
    tok_out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, f"{GENSRC}/corpus_reasoning/data/convert_unified_to_sft.py",
           "--task", "qdmatch", "--input", str(arms_dir / f"{ARM}.jsonl"),
           "--out-dir", str(tok_out), "--tokenizer", "Qwen/Qwen3.5-0.8B-Base",
           "--max-seq-len", "40960", "--eos", "248044"]
    log("tokenizing: " + " ".join(cmd))
    r = subprocess.run(cmd, env={**__import__("os").environ, "PYTHONPATH": GENSRC})
    assert r.returncode == 0, f"tokenize failed rc={r.returncode}"
    assert (tok_out / "metadata.json").exists(), "tokenize left no metadata.json"
    log("DONE: pool + arm + tokenized")


if __name__ == "__main__":
    main()
