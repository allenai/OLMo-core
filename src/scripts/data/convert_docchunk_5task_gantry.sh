#!/usr/bin/env bash
# Launch the doc-chunked 5-task (contradiction / nq / oolong / rerank / outlier) -> Qwen3 SFT
# conversion on ONE Beaker CPU node via gantry, writing the shards straight to weka. Produces BOTH
# the dense (DocumentChunkedAttention / hierarchical_dilated) and landmark (DocumentLandmarkAttention)
# layouts for every task, so the 32k no-CPT doc-chunked matrix rows can read them.
#
# Wraps src/scripts/data/convert_unified_to_document_landmark.py. Mirrors
# convert_longctx_tasks_to_sft_gantry.sh (weka mount + an /input Beaker dataset holding the unified
# JSONL). The input dataset must contain one combined JSONL per task at /input:
#   contra_combined.jsonl  oolong_combined.jsonl  nq_combined.jsonl
#   rerank_combined.jsonl   outlier_combined.jsonl
# (built locally then `beaker dataset create`d -- see the 5-task no-CPT doc-chunked plan.)
#
# Per-task (task name + chunk-by) MUST match the ladder40k provenance:
#   contra  -> --task contradiction --chunk-by document  (each claim a chunk)
#   nq      -> --task retrieval     --chunk-by document  (each retrieved passage a chunk)
#   oolong  -> --task oolong        --chunk-by line --item-regex '||'  (each item line a chunk)
#   rerank  -> --task rerank        --chunk-by document  (each candidate doc a chunk)
#   outlier -> --task outlier       --chunk-by document  (each item a chunk)
#
# Usage:
#   beaker dataset create --name docchunk-5task-jsonl /accounts/.../docchunk_stage
#   INPUT_DATASET=prasanns/docchunk-5task-jsonl \
#     src/scripts/data/convert_docchunk_5task_gantry.sh
#
# Overridable env: CLUSTER WORKSPACE BUDGET WEKA PRIORITY CPUS NAME OUT_ROOT SEQ_LEN MEM_FREQ IMAGE
set -euo pipefail

if [[ -z "${INPUT_DATASET:-}" ]]; then
  echo "error: INPUT_DATASET env var required (Beaker dataset with the 5 *_combined.jsonl)." >&2
  exit 2
fi

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"
CPUS="${CPUS:-16}"
NAME="${NAME:-docchunk-5task-convert}"
SEQ_LEN="${SEQ_LEN:-40960}"
MEM_FREQ="${MEM_FREQ:-63}"
OUT_ROOT="${OUT_ROOT:-/weka/${WEKA}/ai2-llm/checkpoints/prasanns/cptmix_docchunk_ladder40k}"
IMAGE="${IMAGE:-tylerr/olmo-core-tch291cu128-2025-11-25}"

CLUSTER_ARGS=()
IFS=',' read -ra _CLUSTERS <<< "${CLUSTER}"
for c in "${_CLUSTERS[@]}"; do CLUSTER_ARGS+=(--cluster "$c"); done

# Unlike the longctx converter, this one imports olmo_core (document_chunk_landmark), so the package
# must be importable. The baked image has the heavy deps but NOT olmo_core or its small pure-python
# deps (dataclass_extensions, ...), so run an editable install (deps mostly satisfied -> fast).
INSTALL_ARGS=()
if [[ -n "${IMAGE}" ]]; then
  INSTALL_ARGS+=(--beaker-image "${IMAGE}" --install "pip install -e . && pip install dataclass-extensions")
else
  INSTALL_ARGS+=(--install "pip install -e .")
fi

# Remote driver: 5 tasks x {dense,landmark}. Each line: name|task|chunk_by.
read -r -d '' REMOTE <<REMOTE_EOF || true
set -uo pipefail
export PYTHONPATH="\$(pwd)/src:\${PYTHONPATH:-}"   # baked image has deps but NOT the olmo_core pkg
CONV=src/scripts/data/convert_unified_to_document_landmark.py
OUT=${OUT_ROOT}
SEQ=${SEQ_LEN}; MF=${MEM_FREQ}
run() { # name task chunk emit
  name=\$1; task=\$2; chunk=\$3; emit=\$4
  extra=""; [ "\$chunk" = line ] && extra="--item-regex ||"
  outdir=\$OUT/\${name}_\${emit}
  echo "=== convert \$name (\$task,\$chunk) emit=\$emit -> \$outdir \$(date '+%T') ==="
  python \$CONV --emit \$emit --task \$task --chunk-by \$chunk \$extra \
    --cot-mode none --seq-len \$SEQ --mem-freq \$MF \
    --input-jsonl /input/\${name}_combined.jsonl --out-dir \$outdir
  echo "  rc=\$? \$(date '+%T')"
}
for emit in dense landmark; do
  run contra  contradiction document \$emit
  run nq      retrieval     document \$emit
  run oolong  oolong        line     \$emit
  run rerank  rerank        document \$emit
  run outlier outlier       document \$emit
done
echo "=== ALL CONVERSIONS DONE \$(date '+%T') ==="
ls -la \$OUT
echo "=== SANITY: first instance per task/emit (chunk count + monotonic + landmark mod64) ==="
python - <<'PYEOF'
import glob, numpy as np
OUT="${OUT_ROOT}"; MF=${MEM_FREQ}
EOS=151643; DS=151648; DE=151649; MEM=151860; PAD=151863
for emit in ("dense","landmark"):
    for name in ("contra","nq","oolong","rerank","outlier"):
        d=f"{OUT}/{name}_{emit}"
        fs=sorted(glob.glob(f"{d}/token_ids_part_*.npy"))
        if not fs: print(f"[{name}_{emit}] NO SHARDS"); continue
        a=np.fromfile(fs[0],dtype=np.uint32)
        e=np.where(a==EOS)[0]
        inst=a[:e[0]+1] if len(e) else a
        # chunk ids via box markers
        depth=0; idx=-1; ctx=[]
        for t in inst:
            if t==DS: idx+=1; depth=1; ctx.append(idx)
            elif t==DE: ctx.append(idx if depth else -1); depth=0
            elif depth: ctx.append(idx)
        nstart=int((inst==DS).sum()); nend=int((inst==DE).sum())
        mono = ctx==sorted(ctx)
        extra=""
        if emit=="landmark":
            extra=f" len_mod{MF+1}={ (len(inst)-1)%(MF+1) } nmem={int((inst==MEM).sum())} npad={int((inst==PAD).sum())}"
        print(f"[{name}_{emit}] inst_len={len(inst)} nchunks(start/end)={nstart}/{nend} maxid={idx} mono={mono}{extra}")
PYEOF
echo "=== SANITY DONE \$(date '+%T') ==="
REMOTE_EOF

set -x
gantry run \
  --name "${NAME}" \
  --description "Doc-chunked 5-task (contra/nq/oolong/rerank/outlier) dense+landmark SFT shards -> weka" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  "${CLUSTER_ARGS[@]}" \
  --python-manager conda \
  --system-python \
  --weka "${WEKA}:/weka/${WEKA}" \
  --dataset "${INPUT_DATASET}:/input" \
  --cpus "${CPUS}" \
  --gpus 0 \
  --priority "${PRIORITY}" \
  --allow-dirty \
  --shared-memory 32GiB \
  --timeout 0 \
  --env TOKENIZERS_PARALLELISM=false \
  "${INSTALL_ARGS[@]}" \
  --yes \
  -- bash -c "${REMOTE}"
