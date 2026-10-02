#!/usr/bin/env bash
# Dev-loss grid on the Berkeley cluster: one GPU, every kept task in manifest.json x rungs x all
# schemes, one driver process per (task, rung) cell -> /data/prasann/devloss_grid/results/<task>_<rung>.json
#
#   sbatch debug/devloss_grid/run_grid_local.sbatch                       # full grid
#   ONLY_TASKS=oolong RUNGS=2k ROWS=4 sbatch debug/devloss_grid/run_grid_local.sbatch   # smoke
#
# Resumable: a cell whose JSON exists is skipped. Checkpoints are staged by stage_to_node.sh (job
# 3598100); each task waits for its checkpoint's completion sentinel (hf: .text_export_done,
# distcp: model_and_optim/.metadata), so this can start while staging is still running -- tasks
# whose checkpoint is already complete run first. A crashed cell is logged to results/FAILED.tsv
# and the loop continues. Env: ONLY_TASKS (space/comma list of manifest keys), RUNGS (default
# 2k,8k,32k), ROWS (single int overriding the per-rung default 48/48/32), SCHEMES, WAIT_MAX_S,
# RES_DIR (results dir; ALWAYS set it for a smoke so the full grid does not skip that cell).
#SBATCH --job-name=devloss-grid
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --exclude=mooney
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=/data/prasann/joblogs/devloss_grid_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"
source src/scripts/local_env.sh
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
case "$PY" in /data/*) ;; *) echo "!!! interpreter not node-local"; exit 1;; esac
# lorax/thidwick have no node-local env: ONE bulk copy from sneetches (never run from /net). Check by
# IMPORT, not by -x: a cancelled first copy left lorax a python without urllib.parse (2026-09-21).
if ! "$PY" -c "import urllib.parse, torch, olmo_core" 2>/dev/null; then
  echo ".. bootstrapping conda env from sneetches ($(date +%T))"; mkdir -p /data/prasann/conda/envs
  rsync -a /net/sneetches/data/prasann/conda/envs/corpus-reasoning-olmo/ /data/prasann/conda/envs/corpus-reasoning-olmo/ && echo ".. env bootstrapped ($(date +%T))"
  [ -x "$PY" ] || { echo "!!! env bootstrap failed"; exit 98; }
fi
export HF_HOME=/data/prasann/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore
export TMPDIR=/data/prasann/tmp; mkdir -p "$TMPDIR"
# Triton/fla JIT + every ~/.cache went to NFS HOME (260 cache entries appeared in ~/.triton within 5 min of
# the first launch; the drivers sat in rpc_wait_bit_killable at 0% GPU, 23 s/row vs 2.3 s in the smoke).
export TRITON_CACHE_DIR=/data/prasann/triton_cache XDG_CACHE_HOME=/data/prasann/xdg_cache TORCHINDUCTOR_CACHE_DIR=/data/prasann/inductor_cache
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR"
TOK=/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
ROOT=/data/prasann/devloss_grid
# Any H200 node may run this. Everything was staged on sneetches; on another node, bootstrap lazily
# from /net/sneetches (ONE bulk rsync per checkpoint right before its cells -- ~8-17 GB at ~78 MB/s,
# amortised over ~3 cells of hours each; never job I/O over /net).
SRC=/net/sneetches/data/prasann/devloss_grid
if [ "$(hostname -s)" != "sneetches" ]; then
  # mooney's /data was 100% full on 2026-09-21: rsync then copies the tiny sentinel and fails on the
  # 9.6 GB safetensors, so every cell ran against an empty checkpoint dir. Refuse without headroom.
  avail_gb=$(( $(df --output=avail -k /data | tail -1) / 1048576 ))
  [ "$avail_gb" -ge 300 ] || { echo "!!! $(hostname -s):/data has only ${avail_gb} GB free; need >=300 GB to bootstrap checkpoints -- exclude this node"; exit 99; }
  mkdir -p "$ROOT" /data/prasann/hf_cache/hub
  [ -f "$TOK/tokenizer.json" ] || rsync -a /net/sneetches/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base /data/prasann/hf_cache/hub/
  rsync -a "$SRC/data/" "$ROOT/data/" && echo "data bootstrapped from sneetches"
fi
[ -f "$TOK/tokenizer.json" ] || { echo "!!! tokenizer snapshot missing at $TOK"; exit 1; }
bootstrap_ckpt() {  # $1 local ckpt dir, $2 fmt -- copy from sneetches if the local sentinel is missing
  sentinel_ok "$1" "$2" && return 0
  local rel=${1#$ROOT/}; local src="$SRC/$rel"
  if sentinel_ok "$src" "$2"; then
    echo ".. bootstrapping $rel from sneetches ($(date +%T))"; local t0=$SECONDS
    # sentinel copied LAST and only if the bulk rsync succeeded, so a partial copy never looks complete
    mkdir -p "$1" && rsync -a --exclude .text_export_done --exclude .metadata "$src/" "$1/" \
      && rsync -a --include '.text_export_done' --include '*/' --include '.metadata' --exclude '*' "$src/" "$1/" \
      && echo ".. bootstrapped $rel in $((SECONDS-t0))s" || echo "!! bootstrap of $rel FAILED (rsync rc=$?)"
  fi
  sentinel_ok "$1" "$2"
}
# results go to the REPO (NFS, one 16 KB JSON per cell): the 4 group jobs run on different nodes and
# the skip-existing check + the collector need ONE place. Driver work stays on node-local /data.
RES="${RES_DIR:-$REPO/debug/devloss_grid/results}"; mkdir -p "$RES" /data/prasann/joblogs   # RES_DIR: smokes write elsewhere so the grid never skips a 4-row cell
MANIFEST=$REPO/debug/devloss_grid/manifest.json
RUNGS="${RUNGS:-2k,8k,32k}"
ONLY_TASKS="${ONLY_TASKS:-}"
# QUICK pass (2026-09-21 deadline): 7 schemes, rows 16/16/8, rung-major so every task lands at 2k first.
SCHEMES="${SCHEMES:-full,k0,first64,fl64,idf64,rand33,gold_fl64}"
ATTN_BACKEND="${ATTN_BACKEND:-flash_2}"
WAIT_MAX_S="${WAIT_MAX_S:-14400}"
rows_for() { if [ -n "${ROWS:-}" ]; then echo "$ROWS"; elif [ "$1" = "32k" ]; then echo 8; else echo 16; fi; }

echo "== devloss grid on $(hostname -s) job ${SLURM_JOB_ID:-?} ($(date)) rungs=$RUNGS only='$ONLY_TASKS' rows='${ROWS:-default}'"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# manifest -> "key<TAB>driver_task<TAB>ckpt<TAB>fmt" for kept tasks (+cpt80), in manifest order
mapfile -t ENTRIES < <("$PY" - "$MANIFEST" "$ONLY_TASKS" <<'PYEOF'
import json, sys
man = json.load(open(sys.argv[1])); only = sys.argv[2].replace(",", " ").split()
for k, t in man["tasks"].items():
    if t.get("dropped") or not t.get("ckpt"): continue
    if only and k not in only: continue
    fmt = "distcp" if t["ckpt_format"] == "distcp" else "hf"
    task = "cpt80" if k == "cpt80" else f"ctc_{k}"
    print(f"{k}\t{task}\t{t['ckpt']}\t{fmt}")
PYEOF
)
echo "${#ENTRIES[@]} tasks: $(printf '%s ' "${ENTRIES[@]}" | cut -f1 | tr '\n' ' ')"

sentinel_ok() {  # $1 ckpt dir, $2 fmt
  if [ "$2" = "distcp" ]; then [ -f "$1/model_and_optim/.metadata" ] || [ -f "$1/step"*"/model_and_optim/.metadata" ];
  else [ -f "$1/.text_export_done" ]; fi
}

run_cell() {  # key task ckpt fmt rung
  local key=$1 task=$2 ckpt=$3 fmt=$4 rung=$5 out="$RES/${1}_${5}.json"
  if [ -f "$out" ]; then echo "-- skip $key@$rung (exists)"; return 0; fi
  local n; n=$(rows_for "$rung")
  local extra=(); [ -n "$SCHEMES" ] && extra+=(--schemes "$SCHEMES")
  [ "$task" = "cpt80" ] && extra+=(--cpt-source "$ROOT/data/cpt80/long_docs.jsonl")
  echo "== cell $key@$rung rows=$n ckpt=$ckpt ($fmt) $(date +%T)"
  local t0=$SECONDS
  local tmp="$ROOT/work/${key}_${rung}.json"; mkdir -p "$ROOT/work"
  "$PY" -u debug/devloss_grid/ctc_devloss_grid.py --task "$task" --rung "$rung" --rows "$n" \
      --ckpt "$ckpt" --ckpt-format "$fmt" --data-root "$ROOT/data" --tokenizer "$TOK" \
      --attn-backend "$ATTN_BACKEND" --out "$tmp" "${extra[@]}"
  local rc=$?
  # copy via a temp name so a quota failure never leaves a truncated JSON the skip-check would trust;
  # the node-local $tmp is the durable copy (sync_results.sh re-copies it later)
  if [ $rc -eq 0 ] && [ -f "$tmp" ]; then { cp "$tmp" "$out.part" && mv "$out.part" "$out"; } && echo "== done $key@$rung in $((SECONDS-t0))s ($(hostname -s))" || { echo "!! $key@$rung finished but repo copy failed (quota?); kept $tmp"; rm -f "$out.part"; }
  else echo "!! FAILED $key@$rung rc=$rc after $((SECONDS-t0))s"; printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$key" "$rung" "$rc" "$(hostname -s):${SLURM_JOB_ID:-?}" >> "$RES/FAILED.tsv"; fi
}

# RUNG-MAJOR: all tasks at 2k, then 8k, then 32k (a checkpoint is bootstrapped once, at its first cell).
for r in ${RUNGS//,/ }; do
  pending=("${ENTRIES[@]}"); waited=0
  while [ ${#pending[@]} -gt 0 ]; do
    next=(); ran=0
    for e in "${pending[@]}"; do
      IFS=$'\t' read -r key task ckpt fmt <<<"$e"
      if bootstrap_ckpt "$ckpt" "$fmt"; then run_cell "$key" "$task" "$ckpt" "$fmt" "$r"; ran=1; else next+=("$e"); fi
    done
    pending=("${next[@]}")
    [ ${#pending[@]} -eq 0 ] && break
    if [ $ran -eq 0 ]; then
      if [ $waited -ge "$WAIT_MAX_S" ]; then echo "!! gave up waiting for: $(printf '%s ' "${pending[@]}" | cut -f1)"; break; fi
      echo ".. waiting for checkpoints of: $(printf '%s\n' "${pending[@]}" | cut -f1 | tr '\n' ' ') ($waited s)"; sleep 60; waited=$((waited+60))
    fi
  done
done
echo "== grid finished ($(date)); results:"; ls -la "$RES" | tail -n +2 | awk '{print $NF, $5}' ; [ -f "$RES/FAILED.tsv" ] && { echo "FAILED:"; cat "$RES/FAILED.tsv"; }
exit 0
