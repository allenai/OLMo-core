#!/bin/bash
# Score many checkpoints with eval_devce.py across all GPUs of one node: each (checkpoint, shard) is split into
# row chunks, chunks run one per GPU, then merge_devce.py joins them into the usual one-JSON-per-(ckpt, shard).
# Used inside a driver allocation (Berkeley) or a Beaker job; no scheduler calls.
#   SPECS="D-480:runs/o3cd-D-480:4 stage2-yarn4:base/stage2-step47684:4" \
#   SHARDS="dev32k_1024:256 short4k_v2:256" ROOT=/data/prasann/olmo3_cptdrop CODE=<tree> PY=<python> \
#     bash devce_multi.sh
# SPEC = label:ckpt_dir(relative to ROOT or absolute):yarn ; SHARD = name:rows_per_chunk
# EVAL_EXTRA = extra eval_devce flags (e.g. "--ctx-window 8192"); OUT_DIR keeps such runs apart from plain ones
set -uo pipefail
ROOT="${ROOT:?}"; CODE="${CODE:?}"; PY="${PY:?}"
OUT_DIR="${OUT_DIR:-$ROOT/devce}"; NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
EV="$CODE/research/token_routing/olmo3_cptdrop/eval_devce.py"
MG="$CODE/research/token_routing/olmo3_cptdrop/merge_devce.py"
export PYTHONPATH="$CODE/src:$CODE/research/token_routing/olmo3_cptdrop" HF_HOME="$ROOT/hf_cache"
export TMPDIR="${TMPBASE:-$ROOT/tmp}" TRITON_CACHE_DIR="${TMPBASE:-$ROOT/tmp}/triton"
mkdir -p "$OUT_DIR/chunks" "$TMPDIR"
log() { echo "[devce-multi $(date '+%F %T')] $*"; }

tasks=()   # label|ckpt|yarn|shard|start|rows
for spec in ${SPECS:?}; do
  IFS=: read -r label ck yf <<<"$spec"; [[ "$ck" == /* ]] || ck="$ROOT/$ck"
  for shs in ${SHARDS:?}; do
    IFS=: read -r sh per <<<"$shs"
    n=$("$PY" -c "import json; print(json.load(open('$ROOT/data/$sh/metadata.json'))['num_instances'])")
    for ((s = 0; s < n; s += per)); do tasks+=("$label|$ck|$yf|$sh|$s|$per"); done
  done
done
log "${#tasks[@]} chunk tasks on $NGPU GPUs"

free=(); for ((g = 0; g < NGPU; g++)); do free+=("$g"); done
declare -A gpu_of
fail=0
for t in "${tasks[@]}"; do
  while [ "${#free[@]}" -eq 0 ]; do   # bash >= 5.1: wait -n -p names the reaped pid
    wait -n -p done_pid; rc=$?
    [ "$rc" -eq 0 ] || fail=1
    free+=("${gpu_of[$done_pid]}"); unset "gpu_of[$done_pid]"
  done
  g=${free[0]}; free=("${free[@]:1}")
  IFS='|' read -r label ck yf sh s per <<<"$t"
  CUDA_VISIBLE_DEVICES=$g "$PY" -u "$EV" --ckpt "$ck" --yarn-factor "$yf" --dev "$ROOT/data/$sh" --label "$label" \
    --row-start "$s" --rows "$per" ${EVAL_EXTRA:-} --out "$OUT_DIR/chunks/${label}_${sh}_r${s}.json" \
    > "$OUT_DIR/chunks/${label}_${sh}_r${s}.log" 2>&1 &
  gpu_of[$!]=$g
done
for p in "${!gpu_of[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || log "WARN: some chunk failed (see $OUT_DIR/chunks/*.log)"

for spec in $SPECS; do
  IFS=: read -r label ck yf <<<"$spec"
  for shs in $SHARDS; do
    sh=${shs%%:*}
    "$PY" "$MG" --out "$OUT_DIR/${label}_${sh}.json" "$OUT_DIR"/chunks/"${label}_${sh}"_r*.json || fail=1
  done
done
exit $fail
