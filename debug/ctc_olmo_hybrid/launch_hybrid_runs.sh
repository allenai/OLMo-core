#!/usr/bin/env bash
# Launch the Olmo-Hybrid-7B arms of the hybrid-vs-not comparison: {contradiction, qdmatch_hpqa}
# x {full, chunked-mix}, against the marker-audited base built by convert_base_gantry.sh.
#
#   bash debug/ctc_olmo_hybrid/launch_hybrid_runs.sh
#   TASKS=contradiction bash debug/ctc_olmo_hybrid/launch_hybrid_runs.sh
#
# ── WHAT THIS IS COMPARED AGAINST ─────────────────────────────────────────────────────────────
# The Olmo-3-1025-7B side already exists and is NOT re-run -- results/ctc_suite/olmo3_dense_vs_chunked.md:
#
#   contradiction (set_f1)   2k .829/.767   4k .742/.649   8k .646/.449   16k .426/.194   (dense/cmix)
#   qdmatch_hpqa  (pair_f1)  2k .997/.987   4k .996/.790   8k .985/.448   16k .962/.216
#
# Both tasks are run because the two candidate ladders answer different questions and both baselines
# are already paid for: contradiction is the O(N^2) task where the chunked gap is the headline
# result, and qdmatch_hpqa is where the gap is widest (+0.536 at 8k). Running one and not the other
# would leave the obvious follow-up unanswerable for the cost of two more jobs.
#
# ⚠ NOT qdmatch_fiqa. The existing OLMo-3-7B qdmatch ladder is qdmatch_HPQA (trained on
# shards/qdmatch_hpqa_train). Pointing the hybrid at qdmatch_fiqa would compare it against nothing,
# or silently against a different task's baseline -- the two are both "qdmatch" to the eval catalog.
#
# ── EVERY KNOB BELOW IS COPIED FROM THE OLMo-3 RUN IT IS COMPARED TO ──────────────────────────
# Read off debug/ctc_olmo3/launch_contra-full-swa.log and launch_ctc-olmo3-7b-hpqa-full.log, not
# from the suite defaults: seq_len differs per task (20480 contradiction / 16384 qdmatch), and the
# launcher's own default node count would change the global batch and therefore the optimizer step
# count, making a backbone comparison partly a batch-size comparison.
#
# ⚠ --activation-checkpointing full --shard-degree 8 are REQUIRED and not defaulted: _LARGE_SCALES
# is ("2b","4b","9b"), so a 7b scale falls through to AC=none/shard=1. That exact omission is what
# OOM'd the 0.8B contradiction cell four times before the cause was found.
set -uo pipefail
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
LOGD=debug/ctc_olmo_hybrid
LEDGER=$LOGD/LAUNCH_LEDGER.tsv
mkdir -p "$LOGD"
[ -f "$LEDGER" ] || printf 'launched_at\trun_name\ttask\tarm\tscale\tseq_len\tcluster\texperiment_id\tnotes\n' > "$LEDGER"

OLMO3_ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_olmo3
BASE="${BASE:-$OLMO3_ROOT/bases/olmo-hybrid-7b-base-fixmark/model_and_optim}"
CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
TASKS="${TASKS:-contradiction qdmatch}"
TS=$(date +%m%d%H%M)

# task : shard dir : seq_len   (shard + seq_len both matched to the Olmo-3 run)
declare -A SHARD_FOR=( [contradiction]=contradiction_train [qdmatch]=qdmatch_hpqa_train )
declare -A SEQ_FOR=(   [contradiction]=20480              [qdmatch]=16384 )

n=0
for task in $TASKS; do
  shard="${SHARD_FOR[$task]:-}"; seq="${SEQ_FOR[$task]:-}"
  [ -n "$shard" ] || { echo "no shard mapped for task=$task"; continue; }
  for variant in full chunked-mix; do
    tag=$([ "$variant" = "full" ] && echo full || echo cmix)
    RUN="ctc-olmohyb-7b-${task}-${tag}-${TS}"
    echo "=== launching $RUN (variant=$variant seq_len=$seq shard=$shard) ==="
    python -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
      --task "$task" --variant "$variant" \
      --model-family olmo3 --model-scale 7b-hybrid \
      --base-checkpoint "$BASE" \
      --data-root "$OLMO3_ROOT/shards/$shard" \
      --run-name "$RUN" --num-nodes 1 --epochs 1 \
      --seq-len "$seq" --lr 5e-5 --global-batch 8 --micro-batch-instances 1 \
      --no-compile --activation-checkpointing full --shard-degree 8 \
      --wandb-group "ctc-olmohyb-${task}" \
      --cluster "$CLUSTER" --priority urgent --no-follow launch \
      > "$LOGD/launch_${task}_${tag}.log" 2>&1
    rc=$?
    EXP=$(grep -oE 'beaker\.org/ex/[A-Z0-9]+' "$LOGD/launch_${task}_${tag}.log" | head -1 | sed 's#.*/##')
    printf '%s\t%s\t%s\t%s\t7b-hybrid\t%s\t%s\t%s\t%s\n' \
      "$(date -Iseconds)" "$RUN" "$task" "$variant" "$seq" "${CLUSTER##*/}" \
      "${EXP:-SUBMIT-FAILED-rc=$rc}" "Olmo-Hybrid-7B vs Olmo-3-7B backbone control" >> "$LEDGER"
    echo "    rc=$rc exp=${EXP:-NONE}"
    n=$((n+1))
  done
done
echo "=== submitted $n runs; ledger: $LEDGER ==="
