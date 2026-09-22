#!/usr/bin/env bash
# setA (13-task CTC SFT mix) -> Qwen3.5 document-chunked SFT shards, on one Beaker CPU node.
#
# Reads the build's per-task JSONL straight off weka (no /input Beaker dataset: the build already
# wrote there) and writes shards back to weka, so training reads them with no staging hop.
#
# ONE shard set serves BOTH arms. `--variant full` reads these token ids directly, and
# `--variant sparselandmark` inserts landmark tokens at LOAD time via LandmarkPackingInstanceSource
# -- so `--emit dense` is correct for both and there is no landmark-emit pass.
#
# Per-task (unified task name + chunk-by) MUST match the ladder provenance. Ladder names are not
# spec names: nq is `retrieval`, hotpotqa is `cot_retrieval`, qdmatch_nq is `qdmatch`. Passing a
# ladder name where a spec is expected raises on EVERY row, which reads as a 100% error rate.
# Everything is --chunk-by document (each documents[i] a chunk) EXCEPT oolong, whose items are
# `||`-delimited lines inside one document.
#
# ⚠ NEVER pass --item-regex '||' unescaped: as a regex that is an alternation of empty branches and
# matches every line, which is the measured oolong chunk-leak. The converter default is correct.
#
# Usage:
#   src/scripts/data/convert_ctc_setA_13task_gantry.sh
# Overridable env: CLUSTER WORKSPACE BUDGET WEKA PRIORITY CPUS NAME ROOT OUT_ROOT SEQ_LEN
#                  MARKER_SET QUERY_POSITION COT_MODE TOKENIZER IMAGE
set -euo pipefail

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"
CPUS="${CPUS:-64}"
NAME="${NAME:-ctc-setA-13task-convert}"
ROOT="${ROOT:-/weka/${WEKA}/ai2-llm/checkpoints/prasanns/ctc_sft_sets/setA_max20}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/shards_qwen35}"
SEQ_LEN="${SEQ_LEN:-40960}"
MARKER_SET="${MARKER_SET:-qwen3_5}"
QUERY_POSITION="${QUERY_POSITION:-both}"
COT_MODE="${COT_MODE:-none}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-4B}"
IMAGE="${IMAGE:-tylerr/olmo-core-tch291cu128-2025-11-25}"

CLUSTER_ARGS=()
IFS=',' read -ra _CLUSTERS <<< "${CLUSTER}"
for c in "${_CLUSTERS[@]}"; do CLUSTER_ARGS+=(--cluster "$c"); done

# --python-manager conda --system-python is REQUIRED with this image. Without it gantry builds its
# own uv venv, which ships without pip AND without numpy: `pip install -e .` is a no-op and the job
# dies on `ModuleNotFoundError: No module named 'numpy'` seconds after "Setup complete".
read -r -d '' REMOTE <<REMOTE_EOF || true
set -uo pipefail
export PYTHONPATH="\$(pwd)/src:\${PYTHONPATH:-}"
CONV=src/scripts/data/convert_unified_to_document_landmark.py
ROOT=${ROOT}; OUT=${OUT_ROOT}; SEQ=${SEQ_LEN}
mkdir -p \$OUT
rc_all=0
run() { # ladder_name unified_task chunk_by
  name=\$1; task=\$2; chunk=\$3
  outdir=\$OUT/\$name
  echo "=== convert \$name (task=\$task chunk=\$chunk) -> \$outdir \$(date '+%T') ==="
  python \$CONV --emit dense --task \$task --chunk-by \$chunk \
    --marker-set ${MARKER_SET} --tokenizer ${TOKENIZER} \
    --query-position ${QUERY_POSITION} --cot-mode ${COT_MODE} --seq-len \$SEQ \
    --input-jsonl \$ROOT/per_task/\$name/train.jsonl --out-dir \$outdir
  rc=\$?; [ \$rc -ne 0 ] && rc_all=1
  echo "  rc=\$rc \$(date '+%T')"
}
run nq               retrieval        document
run hotpotqa         cot_retrieval    document
run qdmatch_nq       qdmatch          document
run outlier          outlier          document
run oolong           oolong           line
run contradiction    contradiction    document
run xabsence         xabsence         document
run absence          absence          document
run reorder          reorder          document
run rerank           rerank           document
run strmatch         strmatch         document
run textgroups       textgroups       document
run grouping_labeled grouping_labeled document
echo "=== CONVERSIONS DONE rc_all=\$rc_all \$(date '+%T') ==="

echo "=== PER-TASK REALIZED TOKENS (metadata.json) ==="
python - <<'PYEOF'
import glob, json, os
OUT = "${OUT_ROOT}"
rows, tot_tok, tot_inst = [], 0, 0
for d in sorted(glob.glob(f"{OUT}/*/")):
    mp = os.path.join(d, "metadata.json")
    if not os.path.exists(mp):
        print(f"  {os.path.basename(d.rstrip('/')):<18} NO metadata.json"); continue
    m = json.load(open(mp))
    rows.append((os.path.basename(d.rstrip("/")), m))
    tot_tok += m["num_tokens"]; tot_inst += m["num_instances"]
print(f"{'task':<18}{'instances':>11}{'dropped':>9}{'tokens':>15}{'max_len':>9}{'loss_tok':>11}")
for name, m in rows:
    print(f"{name:<18}{m['num_instances']:>11,}{m['num_dropped']:>9,}{m['num_tokens']:>15,}"
          f"{m['max_example_len']:>9,}{m['num_loss_tokens']:>11,}")
print(f"{'TOTAL':<18}{tot_inst:>11,}{'':>9}{tot_tok:>15,}")
# A shard whose max_example_len equals --seq-len means examples were silently truncated out.
for name, m in rows:
    if m["num_dropped"]:
        print(f"  WARN {name}: {m['num_dropped']:,} examples dropped (> --seq-len ${SEQ_LEN})")
PYEOF
exit \$rc_all
REMOTE_EOF

set -x
gantry run \
  --name "${NAME}" \
  --description "setA 13-task CTC SFT -> Qwen3.5 docchunk dense shards on weka" \
  --workspace "${WORKSPACE}" --budget "${BUDGET}" "${CLUSTER_ARGS[@]}" \
  --python-manager conda --system-python \
  --weka "${WEKA}:/weka/${WEKA}" \
  --cpus "${CPUS}" --gpus 0 --priority "${PRIORITY}" \
  --allow-dirty --shared-memory 32GiB --timeout 0 \
  --env TOKENIZERS_PARALLELISM=false \
  --beaker-image "${IMAGE}" \
  --install "pip install -e . && pip install dataclass-extensions" \
  --yes \
  -- bash -c "${REMOTE}"
