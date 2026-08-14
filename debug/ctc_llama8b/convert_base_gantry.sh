#!/usr/bin/env bash
# Download meta-llama/Llama-3.1-8B and convert it to an OLMo-core model-only distcp base WITH the
# marker-embedding repair, entirely on Beaker, writing straight to weka.
#
#   bash debug/ctc_llama8b/convert_base_gantry.sh
#
# ── WHY ON BEAKER RATHER THAN BERKELEY ────────────────────────────────────────────────────────
# The base has to end up on weka for beaker_ctc_suite.py to train from it, and weka is not
# reachable from Berkeley (everything else in this repo goes through a two-step S3 relay). Doing
# the download+convert on Beaker skips a ~16 GB round trip. It is also where the HF credential
# lives: there is no PRASANNS_HF_TOKEN, and Llama is a GATED repo.
#
# ⚠ CREDENTIAL PROVENANCE. This job authenticates to HuggingFace with `amandab_HF_TOKEN`, i.e. it
# downloads Llama-3.1-8B under *amandab's* acceptance of Meta's license, not prasann's. That was
# explicitly authorized (2026-08-14) because no prasanns HF secret exists. If a PRASANNS_HF_TOKEN
# is ever added, switch HF_SECRET to it -- gated-model access is per-account and this is the kind
# of thing that should not quietly become permanent. The repo's Qwen converter defaults to the
# same secret, but Qwen is ungated so that use carries no license implication; this one does.
#
# ── THE THREE STEPS, AND WHY NONE CAN BE SKIPPED ──────────────────────────────────────────────
# 1. snapshot_download            -- the gated fetch.
# 2. make_llama_marker_tokenizer  -- Llama ships no <|box_start|>/<|box_end|>; it has 250 anonymous
#                                    reserved slots. The converter and both evaluators verify that
#                                    those literal strings map to RESERVED_IDS['llama'], so two
#                                    reserved slots get RENAMED (ids untouched) in a local copy.
# 3. convert_llama_base --scale 8b -- shape/param assert against the checkpoint's own config.json,
#                                    plus the marker repair. --scale 8b is load-bearing: it selects
#                                    the 8B shape table (untied LM head, RoPE factor 8.0) instead
#                                    of 3.2-3B's (tied, factor 32.0). Defaulting to 3b here would
#                                    fail the assert rather than train something wrong, which is
#                                    the intended failure mode.
set -euo pipefail

HF_MODEL_ID="${HF_MODEL_ID:-meta-llama/Llama-3.1-8B}"
# ⚠ THIS GANTRY WANTS REPEATED --cluster FLAGS, NOT A COMMA LIST. `--cluster a,b,c` fails with
# "didn't match any allowed clusters" even when every name in the list is valid, which reads like
# a permissions problem and is really a parsing one. Names are the SHORT form (`ai2/jupiter`, not
# `ai2/jupiter-cirrascale-2`); the long spelling resolves only as a single cluster.
# Ordered eager-first on purpose: jupiter is strict-priority with unallocated-only backfill and
# sits at ~989/990 slots, so even `urgent` waits there, while ceres/saturn/neptune are eager and
# start immediately. This job needs no GPU, so it should never wait behind the training fan-out.
CLUSTERS=(--cluster ai2/ceres --cluster ai2/saturn --cluster ai2/neptune --cluster ai2/jupiter)
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"
HF_SECRET="${HF_SECRET:-amandab_HF_TOKEN}"

WEKA_ROOT="/weka/${WEKA}"
CTC_ROOT="${WEKA_ROOT}/ai2-llm/checkpoints/prasanns/ctc_suite"
HF_STAGING="${CTC_ROOT}/hf/Llama-3.1-8B"
TOK_OUT="${CTC_ROOT}/hf/Llama-3.1-8B-marker-tok"
OUT_DIR="${CTC_ROOT}/bases/llama31-8b-base-fixmark"

# CPU-only: the convert loads to CPU tensors and writes safetensors/distcp. No --gpus, so this
# never competes with the training fan-out for jupiter's (currently saturated) GPU slots, and it
# can land on whichever of the three eager clusters has CPU first.
gantry run \
  --name ctc-llama31-8b-base-convert \
  --description "Download ${HF_MODEL_ID} + marker tokenizer + olmo-core distcp with marker repair" \
  --workspace "${WORKSPACE}" --budget "${BUDGET}" \
  "${CLUSTERS[@]}" --gpus 0 --priority "${PRIORITY}" \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka "${WEKA}:${WEKA_ROOT}" \
  --env-secret "HF_TOKEN=${HF_SECRET}" \
  --env-secret "HUGGING_FACE_HUB_TOKEN=${HF_SECRET}" \
  --no-python --allow-dirty --timeout 0 --yes \
  -- bash -c "
set -euo pipefail
REPO=\$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v /root/.cache | head -1 | xargs -r dirname)
echo \"REPO=\$REPO\"
export PYTHONPATH=\"\$REPO/src:\${PYTHONPATH:-}\"
# ⚠ --no-python, NOT --install true. With --install true gantry builds a FRESH uv venv and
# activates it; the literal 'true' install command is a no-op, so that venv ends up EMPTY and
# shadows the baked image's populated conda python -- the job then dies on
# 'ModuleNotFoundError: No module named huggingface_hub' three lines in. --no-python leaves the
# image's own interpreter (which already has torch/transformers/huggingface_hub) in place, and
# PYTHONPATH above is what makes olmo_core/scripts importable from the clone.
# (The oolong eval pipeline survives --install true only because it builds and invokes its own
# venv by absolute path and never relies on gantry's.)
command -v python
python -c 'import huggingface_hub, torch, transformers; print(\"deps OK:\", huggingface_hub.__version__, torch.__version__)'

# The baked image carries the heavy ML stack (torch/transformers/huggingface_hub) but NOT
# olmo-core's own pure-python deps -- importing anything under olmo_core.data pulls
# olmo_core.config -> dataclass_extensions and dies with ModuleNotFoundError. Install it
# --no-deps so pip cannot decide to \"satisfy\" a transitive requirement by moving torch, which is
# the drift that produced the cu13/libnvrtc breakage on the eval pipeline. Idempotent, so the
# retry path (with the download already cached on weka) stays cheap.
python -m pip install --quiet --no-deps 'dataclass-extensions>=0.3.0' 2>&1 | tail -5
python -c 'import dataclass_extensions; print(\"dataclass_extensions OK\")'

echo '=== 1/3 snapshot_download '\"${HF_MODEL_ID}\"' ==='
mkdir -p '${HF_STAGING}'
python -c \"
from huggingface_hub import snapshot_download
p = snapshot_download('${HF_MODEL_ID}', local_dir='${HF_STAGING}',
                      allow_patterns=['*.json','*.safetensors','tokenizer*','*.model'])
print('downloaded ->', p)
\"
ls -la '${HF_STAGING}' | head -20

echo '=== 2/3 marker tokenizer (rename two reserved slots; ids untouched) ==='
python \"\$REPO/src/scripts/data/make_llama_marker_tokenizer.py\" \
  --base '${HF_STAGING}' --out '${TOK_OUT}'

echo '=== 3/3 convert -> olmo-core distcp with marker repair (--scale 8b) ==='
python \"\$REPO/src/scripts/train/memexpress/ctc_suite/convert_llama_base.py\" \
  --base-dir '${HF_STAGING}' --tokenizer '${TOK_OUT}' \
  --scale 8b --out '${OUT_DIR}'

echo '=== RESULT ==='
ls -la '${OUT_DIR}/model_and_optim' | head -5
test -f '${OUT_DIR}/model_and_optim/.metadata' || { echo 'FATAL: no .metadata -- a base without it silently trains FROM SCRATCH'; exit 1; }
cat '${OUT_DIR}/marker_audit.json'
"
