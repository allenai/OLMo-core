#!/bin/bash
# PURE-CHUNKED arm (--variant chunked: document-chunked mask, NO curriculum mask mixing) for the
# five tasks the mask-mixing question is asked on. This is the missing control: every one of the
# 31 chunked runs in debug/ctc_modelscale/LAUNCH_LEDGER.tsv is `chunked-mix`, so "how much does
# mask-mixing help?" has never actually been measured -- the suite's chunked column IS the mixed
# arm. These runs supply the unmixed baseline it should be read against.
#
#   bash debug/ctc_purechunk/launch_purechunk.sh
#
# ── WHY THESE seq_len VALUES, AND WHY THEY ARE NOT FREE PARAMETERS ─────────────────────────────
# seq_len must be >= the shard's max_example_len: the converter DROPS over-length examples, so a
# short seq_len silently deletes the long tail -- which on a length ladder is the part being
# measured. Read from each shard's metadata.json on S3 and cross-checked against what the
# chunked-mix baseline actually launched with (July fan-out debug/ctc_suite_4b_fanout/
# batch_launch_beaker.sh, and debug/ctc_modelscale/launch_fiqa_training.sh for qdmatch_fiqa):
#
#   task          max_example_len   seq_len   baseline source
#   contradiction      40957         40960    ctc_modelscale launch logs
#   reorder            40957         40960    July fan-out table
#   fiqa               25661         26112    July fan-out table
#   qdmatch_fiqa       40399         40960    launch_fiqa_training.sh
#   outlier            40558         40960    (not in the fan-out table; sized from the shard,
#                                              consistent with outlier_amzn's 40960)
#
# ⚠ `outlier` IS THE SCALE-K ROW. The artifact page's "outlier scale-k" maps to task dir `outlier`
# (paperdraft/figures/export_ctc_suite_data.py LG_TO_ROW); "outlier fix-k" is `outlier_fixedM`.
# Launching --task outlier_fixedM here would produce a control for the wrong row.
#
# ── COMPARABILITY KNOBS (must match the chunked-mix baselines or the delta is not the mask) ────
# --num-nodes 1 -> global-batch 8, which is what every 4B suite run launched with (the launcher's
# own default is 2 nodes / batch 16). --epochs 1 and --lr 5e-5 are the plan defaults both arms use.
#
# ⚠ ONE KNOWN ASYMMETRY, DELIBERATELY LEFT: train_ctc_suite.py force-disables torch.compile for
# `chunked-mix` (the per-step mix coin is not compilable) but NOT for plain `chunked`, so these
# runs compile and their baselines did not. That is a speed difference, not a semantics one -- the
# mask itself is identical. If a run dies in compilation, relaunch it with --no-compile rather
# than assuming the chunked mask is broken.
set -uo pipefail
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
LOGD=debug/ctc_purechunk
LEDGER=$LOGD/LAUNCH_LEDGER.tsv
mkdir -p "$LOGD"
[ -f "$LEDGER" ] || printf 'launched_at\trun_name\ttask\tarm\tscale\tseq_len\tcluster\texperiment_id\tnotes\n' > "$LEDGER"

# task : seq_len
ENTRIES="
contradiction 40960
reorder 40960
fiqa 26112
qdmatch_fiqa 40960
outlier 40960
"

TS=$(date +%m%d%H%M)
n=0
while read -r task seq; do
  [ -z "$task" ] && continue
  RUN="ctc-4b-${task}-purechunk-${TS}"
  echo "=== launching $RUN (seq_len=$seq) ==="
  # --no-follow is REQUIRED when batch-launching: the launcher's default streams the job's logs
  # and blocks for the WHOLE training run, so without it this loop submits one run and then sits
  # on it for hours while the other four never launch.
  python -u src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py \
    --task "$task" --variant chunked --model-scale 4b \
    --run-name "$RUN" --num-nodes 1 --epochs 1 \
    --seq-len "$seq" --priority urgent --no-follow launch \
    > "$LOGD/launch_${task}_purechunk.log" 2>&1
  rc=$?
  EXP=$(grep -oE 'beaker\.org/ex/[A-Z0-9]+' "$LOGD/launch_${task}_purechunk.log" | head -1 | sed 's#.*/##')
  printf '%s\t%s\t%s\tchunked\t4b\t%s\tjupiter\t%s\t%s\n' \
    "$(date -Iseconds)" "$RUN" "$task" "$seq" "${EXP:-SUBMIT-FAILED-rc=$rc}" "pure-chunked mask-mixing control" >> "$LEDGER"
  echo "    rc=$rc exp=${EXP:-NONE}"
  n=$((n+1))
done <<< "$ENTRIES"
echo "=== submitted $n runs; ledger: $LEDGER ==="
