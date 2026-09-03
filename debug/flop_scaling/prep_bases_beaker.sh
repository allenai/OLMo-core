#!/bin/bash
# Prepare marker-repaired Qwen3.5 bases on weka for the model-scale ladder.
#   MODE=fix    SCALES="0.8b 2b"  -> fix_marker_embeddings_qwen35.py on the existing *-modelonly bases
#   MODE=fetch  SCALE=9b HF=Qwen/Qwen3.5-9B-Base -> download (anonymous) + convert + fix
# Runs on the image's conda python with PYTHONPATH=src, installing missing pure-python deps on the fly.
set -uo pipefail
MODE="${MODE:?}"; W=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
read -r -d '' PRE <<'EOP'
set -uo pipefail; export PYTHONWARNINGS=ignore PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0; PYBIN=/opt/conda/bin/python
$PYBIN -c "import fla" 2>/dev/null || $PYBIN -m pip install -q flash-linear-attention 2>&1 | tail -1   # GatedDeltaNet asserts has_fla() even on CPU
for i in 1 2 3 4 5 6 7 8; do
  MISSING=$($PYBIN -c "import olmo_core.nn.transformer, olmo_core.distributed.checkpoint, transformers, safetensors, fla" 2>&1 | grep -oE "No module named '[^']+'" | sed "s/No module named '//; s/'//" | cut -d. -f1)
  [ -z "$MISSING" ] && break; echo "installing $MISSING"; $PYBIN -m pip install -q "$(echo $MISSING | tr _ -)" 2>&1 | tail -1
done
$PYBIN -c "import olmo_core.nn.transformer; from olmo_core.nn.attention.recurrent import has_fla; assert has_fla(); print('imports OK, fla', has_fla())" || exit 1
EOP
if [ "$MODE" = fix ]; then
  SCALES="${SCALES:-0.8b 2b}"
  WORK="$PRE"$'\n'"for S in $SCALES; do T=\$(echo \$S | tr -d .); SRC=$W/ctc_suite/bases/q35-\${T}-base-modelonly/model_and_optim; OUT=$W/ctc_suite/bases/q35-\${T}-base-markerfix; [ -f \$OUT/model_and_optim/.metadata ] && { echo \"[skip] \$OUT\"; continue; }; echo \"--- fix \$S: \$SRC -> \$OUT\"; \$PYBIN src/scripts/data/fix_marker_embeddings_qwen35.py --base \$SRC --out \$OUT --model-scale \$S || { echo \"!!! fix failed \$S\"; exit 1; }; done; echo PREP_DONE"
  NAME="fs35-prepfix-$(date +%m%d%H%M)"; MEM=64GiB; CPUS=8
else
  SCALE="${SCALE:?}"; HF="${HF:?}"; T=$(echo $SCALE | tr -d .); HFDIR=$W/hf_models/$(basename $HF)
  WORK="$PRE"$'\n'"mkdir -p $HFDIR; \$PYBIN - <<'PY'
import time, os
from huggingface_hub import snapshot_download
for i in range(10):
    try:
        p = snapshot_download('$HF', local_dir='$HFDIR', allow_patterns=['*.json','*.safetensors','*.txt','*.py','tokenizer*','*.jinja'], max_workers=4); print('downloaded', p, flush=True); break
    except Exception as e:
        print('download retry', i, type(e).__name__, str(e)[:160], flush=True); time.sleep(min(60*2**i, 900))
else:
    raise SystemExit('download failed')
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT0=$W/ctc_suite/bases/q35-${T}-base-modelonly; OUT1=$W/ctc_suite/bases/q35-${T}-base-markerfix
[ -f \$OUT0/model_and_optim/.metadata ] || \$PYBIN src/scripts/train/memexpress/ctc_suite/convert_qwen35_base.py --base-dir $HFDIR --out \$OUT0 || { echo '!!! convert failed'; exit 1; }
[ -f \$OUT1/model_and_optim/.metadata ] || \$PYBIN src/scripts/data/fix_marker_embeddings_qwen35.py --base \$OUT0/model_and_optim --out \$OUT1 --model-scale $SCALE || { echo '!!! fix failed'; exit 1; }
ls \$OUT1; echo PREP_DONE"
  NAME="fs35-prep${T}-$(date +%m%d%H%M)"; MEM="${MEM:-200GiB}"; CPUS=16
fi
gantry run --name "$NAME" -w ai2/flex2 -b ai2/oe-other --cluster ai2/jupiter-cirrascale-2 --cluster ai2/ceres-cirrascale --cluster ai2/saturn-cirrascale --cluster ai2/neptune-cirrascale --gpus 0 --cpus $CPUS --memory $MEM --priority urgent --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false --weka oe-training-default:/weka/oe-training-default --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -oE "ex/[A-Z0-9]{26}" | head -1 | cut -d/ -f2
