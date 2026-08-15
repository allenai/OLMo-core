#!/usr/bin/env bash
# Download allenai/Olmo-Hybrid-7B and convert it to an olmo-core distcp base for the CTC suite,
# entirely on Beaker, writing straight to weka.
#
#   bash debug/ctc_olmo_hybrid/convert_base_gantry.sh
#
# ── WHY ON BEAKER ─────────────────────────────────────────────────────────────────────────────
# The base has to land on weka for beaker_ctc_suite.py to train from it, and weka is unreachable
# from Berkeley. Converting here skips a ~15 GB round trip through the S3 relay. Unlike the Llama
# job this needs NO HF token -- Olmo-Hybrid is ungated -- so there is no borrowed-credential
# question to answer.
#
# ── WHAT IS AND IS NOT REBUILT ────────────────────────────────────────────────────────────────
# Only the base. Olmo-Hybrid shares Olmo-3's dolma2 tokenizer, its marker ids and its 100352-row
# embedding, so the EXISTING patched marker tokenizer and the EXISTING olmo3 shards are reused
# untouched. That is what makes hybrid-vs-not a clean comparison: both arms read byte-identical
# training data, so nothing but the backbone differs.
#
# ── THE AUDIT IS NOT OPTIONAL, AND IT MAY LEGITIMATELY PASS ───────────────────────────────────
# Two marker bugs have each cost this project a full round of runs: bit-identical open/close marker
# rows (records/document-chunked-marker-embeddings.md) and in-distribution-cosine-but-tiny-norm rows
# that RMSNorm amplifies into noise (records/n100-chunked-marker-position-bug.md). OLMo's markers
# are <|extra_id_1|>/<|extra_id_2|> -- ids INSIDE the real dolma2 vocab, not untrained padding rows
# -- so unlike Qwen3 they may genuinely be healthy. olmo3_marker_audit.py MEASURES and only repairs
# on failure; "repairing" an already-trained row would throw away signal. Read its numbers.
set -euo pipefail
# gantry lives in the corpus-reasoning-olmo env; a bare `gantry` is not on the default PATH here.
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH

HF_MODEL_ID="${HF_MODEL_ID:-allenai/Olmo-Hybrid-7B}"
# ⚠ REPEATED --cluster FLAGS, NOT A COMMA LIST, and the SHORT names. `--cluster a,b,c` fails with
# "didn't match any allowed clusters" even when every name is valid. Eager clusters first: this is
# a 0-GPU job and should never queue behind the training fan-out on jupiter.
CLUSTERS=(--cluster ai2/ceres --cluster ai2/saturn --cluster ai2/neptune --cluster ai2/jupiter)
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-urgent}"

WEKA_ROOT="/weka/${WEKA}"
OLMO3_ROOT="${WEKA_ROOT}/ai2-llm/checkpoints/prasanns/ctc_olmo3"
HF_STAGING="${OLMO3_ROOT}/hf/Olmo-Hybrid-7B"
TOK_DIR="${OLMO3_ROOT}/tokenizer"                     # existing patched dolma2 marker tokenizer
OUT_DIR="${OLMO3_ROOT}/bases/olmo-hybrid-7b-base-fixmark"

gantry run \
  --name ctc-olmo-hybrid-7b-base-convert \
  --description "Download ${HF_MODEL_ID} + olmo-core distcp base + marker audit" \
  --workspace "${WORKSPACE}" --budget "${BUDGET}" \
  "${CLUSTERS[@]}" --gpus 0 --priority "${PRIORITY}" \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka "${WEKA}:${WEKA_ROOT}" \
  --no-python --allow-dirty --timeout 0 --yes \
  -- bash -c "
set -euo pipefail
REPO=\$(find / -maxdepth 3 -iname pyproject.toml 2>/dev/null | grep -v /opt/conda | grep -v /root/.cache | head -1 | xargs -r dirname)
echo \"REPO=\$REPO\"
# The ctc_suite scripts import each other by bare module name (olmo_hybrid_configs), so that
# directory has to be on the path as well as src/.
export PYTHONPATH=\"\$REPO/src:\$REPO/src/scripts/train/memexpress/ctc_suite:\${PYTHONPATH:-}\"
# --no-python, NOT --install true: the latter builds an EMPTY uv venv that shadows the image's
# populated python and the job dies on ModuleNotFoundError three lines in.
python -c 'import huggingface_hub, torch, transformers, safetensors; print(\"deps OK:\", torch.__version__)'
# The image carries the heavy ML stack but not olmo-core's own pure-python deps. --no-deps so pip
# cannot \"satisfy\" a transitive pin by relocating torch.
python -m pip install --quiet --no-deps 'dataclass-extensions>=0.3.0' 2>&1 | tail -5
# ⚠ flash-linear-attention IS REQUIRED TO BUILD THE MODEL AT ALL, not just to run it fast.
# GatedDeltaNet.__init__ does `assert has_fla()` and then `from fla.modules import FusedRMSNormGated`,
# so without a COMPLETE fla the conversion dies after the 15GB download and after the key mapping.
#
# ⚠ has_fla() IS NOT A SUFFICIENT GATE. It is literally `fla is not None` behind a bare
# `try: import fla`, so it passes whenever the top-level package imports -- it said True on a run
# whose very next step died with `ModuleNotFoundError: No module named 'fla.modules'`. The real
# gate is importing the exact symbol GatedDeltaNet needs, which is what is asserted below.
#
# Installed WITHOUT --no-deps this time, under a PIP_CONSTRAINT pinning the image's torch so pip
# cannot relocate it while resolving fla's own requirements (torch/transformers/einops). --no-deps
# was the earlier attempt and produced exactly the partial install described above.
python -c 'import torch; print(torch.__version__)' > /tmp/torchver.txt
# \$(...) escaped: this whole body is inside the launcher's double-quoted `bash -c`, so an
# unescaped $(...) would be evaluated HERE on the login node instead of on the Beaker node.
printf 'torch==%s\n' \$(cat /tmp/torchver.txt) > /tmp/pipconstraint.txt
cat /tmp/pipconstraint.txt
PIP_CONSTRAINT=/tmp/pipconstraint.txt python -m pip install 'flash-linear-attention==0.4.1' einops 2>&1 | tail -15
echo '--- fla diagnostics ---'
python -m pip show flash-linear-attention 2>&1 | head -8
python -c 'import fla; print(fla.__file__)'
python -c 'import torch; print(torch.__version__)'
# THE gate: the exact import GatedDeltaNet performs. Fails fast here (seconds) instead of after
# the download and mapping (minutes).
python -c 'from fla.modules import FusedRMSNormGated; print(FusedRMSNormGated)'

echo '=== 1/3 snapshot_download '\"${HF_MODEL_ID}\"' (ungated; no token needed) ==='
mkdir -p '${HF_STAGING}'
python -c \"
from huggingface_hub import snapshot_download
p = snapshot_download('${HF_MODEL_ID}', local_dir='${HF_STAGING}',
                      allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt','*.model'])
print('downloaded ->', p)
\"
ls -la '${HF_STAGING}' | head -20

echo '=== 2/3 convert -> olmo-core distcp ==='
# Reuses the EXISTING patched olmo3 marker tokenizer; nothing tokenizer-side is rebuilt.
test -d '${TOK_DIR}' || { echo 'FATAL: patched olmo3 marker tokenizer missing on weka'; exit 3; }
python \"\$REPO/src/scripts/train/memexpress/ctc_suite/convert_olmo_hybrid_base.py\" \
  --hf-src '${HF_STAGING}' --out '${OUT_DIR}'

echo '=== 3/3 marker audit (measures; repairs only on failure) ==='
python \"\$REPO/src/scripts/train/memexpress/ctc_suite/olmo3_marker_audit.py\" \
  --checkpoint '${OUT_DIR}/model_and_optim' --tokenizer '${TOK_DIR}' \
  --out '${OUT_DIR}/marker_audit.json' --repair-to '${OUT_DIR}-repaired'

echo '=== RESULT ==='
ls -la '${OUT_DIR}/model_and_optim' | head -5
test -f '${OUT_DIR}/model_and_optim/.metadata' || { echo 'FATAL: no .metadata -- a base without it silently trains FROM SCRATCH rather than failing'; exit 1; }
cat '${OUT_DIR}/marker_audit.json'
"
