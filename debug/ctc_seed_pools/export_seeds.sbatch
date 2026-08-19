#!/usr/bin/env bash
# Export seed pools (`ctc-data pool export`) for the corpus-backed ladders whose expensive
# inputs live on this node. One driver, three invocations:
#
#   sbatch --nodelist=mooney --partition=jsteinhardt --qos=preemptive_high --gres=gpu:1 \
#       export_seeds.sbatch gpu       # nq hotpotqa fiqa scifact rerank + qdmatch_nq qdmatch_hpqa
#   sbatch --nodelist=mooney --partition=jsteinhardt --qos=preemptive_high \
#       export_seeds.sbatch mooney_cpu   # outlier oolong contra_fever outlier_review
#   sbatch --nodelist=horton --partition=berkeleynlp \
#       export_seeds.sbatch horton_cpu   # absence reorder   (gutenberg is cached on horton)
#
# Every task is independent: a failure logs and moves on, so one broken corpus does not cost the
# night. Seeds land in /data/prasann/ctc_seed_pools on the node; collect over /net for upload.
#
#SBATCH --job-name=ctc-seed-export
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/data/prasann/joblogs/ctc_seed_export_%j.log

set -uo pipefail

GROUP="${1:?usage: export_seeds.sbatch gpu|mooney_cpu|horton_cpu}"
REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
# corpus-reasoning-eval is the only env with pyserini (nq/rerank/fiqa/scifact), but it lacks
# nltk; the gutenberg group needs nltk+punkt and no pyserini, so it runs on -olmo.
if [ "$GROUP" = horton_cpu ]; then
    PY="${CTC_EXPORT_PY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python}"
else
    PY="${CTC_EXPORT_PY:-/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python}"
fi
OUT=/data/prasann/ctc_seed_pools
mkdir -p "$OUT" /data/prasann/joblogs

# The interpreter is on NFS (pyserini lives only there); the import tax is minutes at worst and
# these are hour-scale one-off jobs. Everything else -- caches, indexes, outputs -- is node-local.
export PYTHONPATH="$REPO/ctc/src"
export HF_HOME=/data/prasann/hf_cache
export PYSERINI_CACHE=/data/prasann/pyserini_cache
export NLTK_DATA=/data/prasann/nltk_data
export TMPDIR=/data/prasann/tmp
mkdir -p "$TMPDIR"
[ -d /data/prasann/hf ] && export HF_HOME=/data/prasann/hf   # horton's cache dir has this name

run() {
    local task="$1"; shift
    echo "=== $(date +%H:%M:%S) pool export --task $task $*"
    if "$PY" -m ctc.data.cli pool export --task "$task" --out "$OUT" "$@"; then
        echo "=== $(date +%H:%M:%S) OK $task"
    else
        echo "=== $(date +%H:%M:%S) FAILED $task (continuing)"
    fi
}

case "$GROUP" in
gpu)
    run nq
    run hotpotqa
    run fiqa
    run scifact
    run rerank
    # qdmatch pools project a BUILT unified-retrieval file; build train-only sets from the seeds
    # just written, then export from those files. Fast: no model, no index.
    for src in nq hotpotqa; do
        qd="qdmatch_${src/hotpotqa/hpqa}"
        if [ -f "$OUT/$src.seed.jsonl.gz" ]; then
            "$PY" -m ctc.data.cli build --task "$src" --pool "$OUT/$src.seed.jsonl.gz" \
                --split train --train 20000 --out "$TMPDIR/qd_src" \
                && run "$qd" -C "path=$TMPDIR/qd_src/$src/train.jsonl"
        else
            echo "=== skipping $qd: no $src seed"
        fi
    done
    ;;
rerank_only)
    # Rerun slot. First pass died on the _pool_scores fill-shadowing bug (fixed same day); the
    # second exposed the default num_queries=5000 as unusable at suite scale -- one query per
    # example means a 500-query eval split has ZERO slack for a 500-example eval, and train caps
    # at 4.5k. 25k queries supports the standard 20k train + 500 eval with headroom.
    run rerank -C num_queries=25000
    ;;
mooney_cpu)
    run outlier -C cache=/data/prasann/single_task_ladders_20k/wiki100w_article_pool.pkl -C build=false
    run oolong
    run contra_fever
    run outlier_review
    ;;
horton_cpu)
    run absence
    run reorder
    ;;
*)
    echo "unknown group $GROUP" >&2
    exit 2
    ;;
esac

echo "=== $(date +%H:%M:%S) done; wrote:"
ls -lh "$OUT"
