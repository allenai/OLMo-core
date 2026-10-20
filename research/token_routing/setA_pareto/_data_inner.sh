#!/bin/bash
# Inside the gantry job of data_beaker.sh (env: WEKA MODE SEED JOBS). Repo = cwd.
set -uo pipefail
PYB=/opt/conda/bin/python
SETS=$WEKA/ctc_sft_sets
OUT=$SETS/setA_pareto
mkdir -p $OUT/reports
$PYB -m pip install -q -e . 2>&1 | tail -2
$PYB -m pip install -q -e ctc/ 2>&1 | tail -2
TASKS="nq hotpotqa qdmatch_nq outlier oolong contradiction xabsence reorder rerank strmatch grouping"
merge() {  # merge <set root> <shard dir name> <out name>
  local root=$1 sd=$2 name=$3 srcs=""
  for t in $TASKS; do [ -s $root/$sd/$t/metadata.json ] && srcs="$srcs $root/$sd/$t" || echo "!!! no shard for $t"; done
  if [ -s $OUT/$name/metadata.json ]; then echo "[skip] $OUT/$name exists"; return 0; fi
  $PYB research/token_routing/ds64_fast8k/merge_tasks.py --out $OUT/$name --name setA32k --max-row-len 32768 --src $srcs
}
case "$MODE" in
gold)
  ROOT=$SETS/setA_max20_evaliid
  $PYB research/token_routing/setA_pareto/setA_gold_sidecars.py --root $ROOT --jobs 11 --num-proc 3
  rc=$?
  cp $ROOT/shards_qwen35_256k_gold/sidecar_report.json $OUT/reports/sidecar_report_s42.json 2>/dev/null
  echo "sidecar rc=$rc (non-zero = bytes differ from the shipped shards; see report)"
  merge $ROOT shards_qwen35_256k_gold setA32k_s42 || exit 1
  ;;
expand)
  POOLS=""
  for d in $SETS/_external_pools $SETS/external_pools $WEKA/ctc_suite_data_shared $WEKA/ctc_sft_sets/setA_max20_evaliid/_pools; do
    [ -s $d/pool_gutenberg_train.jsonl ] && POOLS=$d && break
  done
  if [ -z "$POOLS" ]; then
    echo "searching weka for external pools ..."
    G=$(timeout 600 find $WEKA -maxdepth 5 -name pool_gutenberg_train.jsonl 2>/dev/null | head -1)
    [ -n "$G" ] && POOLS=$(dirname $G)
  fi
  echo "external pools: ${POOLS:-NOT FOUND (xabsence/grouping cells will FAIL and be reported)}"
  [ -n "$POOLS" ] && ls -la $POOLS
  BASE=$SETS/setA_seed$SEED
  $PYB recipes/data/ctc_sft/build_ctc_sft.py --set set-a --out $BASE --seed $SEED \
    --buckets 2k 4k 8k 16k 32k --calibration recipes/data/ctc_sft/eval_calibration.json \
    ${POOLS:+--external-pools $POOLS} --jobs $JOBS \
    --convert --seq-len 262144 --shards-dir shards_qwen35_256k --gold-sidecar
  echo "build rc=$?"
  ROOT=$BASE/setA_max20_evaliid
  cp $ROOT/manifest.json $OUT/reports/manifest_s$SEED.json
  $PYB research/token_routing/setA_pareto/expansion_overlap.py --new $ROOT --ref $SETS/setA_max20_evaliid \
    --buckets 2k 4k 8k 16k 32k --out $OUT/reports/overlap_s$SEED.json
  merge $ROOT shards_qwen35_256k setA32k_s$SEED || exit 1
  ;;
*) echo "unknown MODE=$MODE"; exit 2 ;;
esac
ls -la $OUT; echo DATA_DONE
