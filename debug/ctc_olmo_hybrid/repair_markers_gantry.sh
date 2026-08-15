#!/usr/bin/env bash
# Audit + repair the document-marker embedding rows of the converted Olmo-Hybrid-7B base.
#
#   bash debug/ctc_olmo_hybrid/repair_markers_gantry.sh
#
# ── WHY THIS IS A SEPARATE JOB FROM convert_base_gantry.sh ────────────────────────────────────
# The conversion already SUCCEEDED and wrote
# ctc_olmo3/bases/olmo-hybrid-7b-base-fixmark (load_state_dict(strict=True) OK on all 523
# tensors). Only step 3/3 failed, because olmo3_marker_audit.py hardcoded olmo3_7B for its repair
# path. Re-running the whole convert would redo a 15GB download and a 10-minute distcp write to
# redo a 2-minute repair.
#
# ── WHAT THE AUDIT FOUND, AND WHY THE REPAIR IS NOT OPTIONAL ──────────────────────────────────
#   embedding matrix (100352, 3840); trained-row norm median 7.8023 (p05 4.2294, p95 10.0881)
#   doc_start id=100266 norm=1.2062 (0.15x median)
#   doc_end   id=100267 norm=1.2132 (0.16x median)
#   pairwise cosine -0.0230, bit_identical=False
#   VERDICT: cosine_gate=PASS  norm_gate=FAIL  -> REPAIR REQUIRED
#
# So this base has the SECOND marker bug, not the first: the two markers are perfectly
# distinguishable from each other (cosine ~0), but they sit at ~1/6 the norm of a real token.
# RMSNorm rescales every position to unit norm, so an undertrained marker row does not arrive as a
# weak signal -- it arrives as a FULL-STRENGTH NOISE vector at every marker position. That is the
# failure in records/n100-chunked-marker-position-bug.md, where it flatlined training at CE ~0.79
# for *every* mask including plain causal, and read as "the mask is too restrictive" when it was
# not. Training either arm from the unrepaired base would produce a clean-looking null result.
#
# ⚠ TRAIN FROM THE -repaired COPY. The repair writes a new base rather than mutating the original,
# so the unrepaired one stays on weka and is easy to point at by accident.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH

CLUSTERS=(--cluster ai2/ceres --cluster ai2/saturn --cluster ai2/neptune --cluster ai2/jupiter)
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"

WEKA_ROOT="/weka/${WEKA}"
OLMO3_ROOT="${WEKA_ROOT}/ai2-llm/checkpoints/prasanns/ctc_olmo3"
TOK_DIR="${OLMO3_ROOT}/tokenizer"
BASE_DIR="${OLMO3_ROOT}/bases/olmo-hybrid-7b-base-fixmark"
OUT_DIR="${BASE_DIR}-repaired"

gantry run \
  --name ctc-olmo-hybrid-7b-marker-repair \
  --description "Audit + repair document-marker embedding rows of the Olmo-Hybrid-7B base" \
  --workspace "${WORKSPACE}" --budget "${BUDGET}" \
  "${CLUSTERS[@]}" --gpus 0 --priority "${PRIORITY}" \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka "${WEKA}:${WEKA_ROOT}" \
  --no-python --allow-dirty --timeout 0 --yes \
  -- bash -c "
set -uo pipefail
REPO=\$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v /root/.cache | head -1 | xargs -r dirname)
export PYTHONPATH=\"\$REPO/src:\$REPO/src/scripts/train/memexpress/ctc_suite:\${PYTHONPATH:-}\"
python -m pip install --quiet --no-deps 'dataclass-extensions>=0.3.0' 2>&1 | tail -3
# fla with deps under a torch pin -- --no-deps yields a partial package whose fla.modules is
# missing, and has_fla() does not catch that (it is just \`fla is not None\`).
python -c 'import torch; print(torch.__version__)' > /tmp/torchver.txt
printf 'torch==%s\n' \$(cat /tmp/torchver.txt) > /tmp/pipconstraint.txt
PIP_CONSTRAINT=/tmp/pipconstraint.txt python -m pip install --quiet 'flash-linear-attention==0.4.1' einops 2>&1 | tail -5
python -c 'from fla.modules import FusedRMSNormGated; print(FusedRMSNormGated)'

test -f '${BASE_DIR}/model_and_optim/.metadata' || { echo 'FATAL: converted base missing'; exit 3; }
python -u \"\$REPO/src/scripts/train/memexpress/ctc_suite/olmo3_marker_audit.py\" \
  --checkpoint '${BASE_DIR}/model_and_optim' --tokenizer '${TOK_DIR}' \
  --arch olmo_hybrid --out '${BASE_DIR}/marker_audit.json' --repair-to '${OUT_DIR}'
rc=\$?
echo \"audit+repair rc=\$rc\"
[ \$rc -ne 0 ] && exit \$rc

echo '=== RESULT ==='
ls -la '${OUT_DIR}/model_and_optim' | head -5
test -f '${OUT_DIR}/model_and_optim/.metadata' || { echo 'FATAL: repaired base has no .metadata -- it would train FROM SCRATCH'; exit 1; }
echo '=== re-audit of the REPAIRED base (must now PASS both gates) ==='
python -u \"\$REPO/src/scripts/train/memexpress/ctc_suite/olmo3_marker_audit.py\" \
  --checkpoint '${OUT_DIR}/model_and_optim' --tokenizer '${TOK_DIR}' \
  --arch olmo_hybrid --out '${OUT_DIR}/marker_audit.json'
"
