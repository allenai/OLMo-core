#!/bin/bash
# In-container: run the three vision-alignment eval suites on one checkpoint (8 GPUs).
# usage: run_alignment_eval.sh <checkpoint> <output_dir> <panel_checkpoint> [suites: fast-text,decoded,academic]
set -euo pipefail
CK=$1; OUT=$2; PANEL_CK=$3; SUITES=${4:-fast-text,decoded,academic}
J=/weka/oe-training-default/jasonr/olmo35-vision-alignment/eval-inputs
HUB=/weka/oe-training-default/jasonr/hf-home/hub
TOK=$HUB/models--allenai--dolma2-tokenizer/snapshots/5292e5d6c0f40b67cc765fe41bec991cf4345b5c/tokenizer.json
export PYTHONPATH=src:src/scripts/eval TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
pip install -q ai2-olmo-eval==0.9.0 'datasets==3.6.0' 'pyarrow==19.0.1'
pip install -q --no-deps --only-binary=:all: scipy==1.18.1
python -c "import torch, datasets, olmo_eval, scipy; print('torch', torch.__version__, 'datasets', datasets.__version__, 'scipy', scipy.__version__)"
mkdir -p $OUT
RUN="torchrun --standalone --nnodes=1 --nproc-per-node=8 src/scripts/eval/Vision-Align.py"
if [[ ,$SUITES, == *,fast-text,* ]]; then
  A=(fast-text --checkpoint $CK --tokenizer $TOK --output $OUT/fast_text.json)
  python src/scripts/eval/Vision-Align.py "${A[@]}" --dry-run > $OUT/fast_text.preflight.json
  [[ -f $OUT/fast_text.json ]] || $RUN "${A[@]}"
  python src/scripts/eval/Vision-Align.py "${A[@]}" --check-complete
  echo "EVAL_SUITE_DONE fast-text"
fi
export HF_HUB_OFFLINE=1
if [[ ,$SUITES, == *,decoded,* ]]; then
  A=(decoded --checkpoint $CK --panel-checkpoint $PANEL_CK --panel-file $J/decoded-panel.json --output-dir $OUT/decoded)
  python src/scripts/eval/Vision-Align.py "${A[@]}" --dry-run > $OUT/decoded.preflight.json
  [[ -f $OUT/decoded/results.json ]] || $RUN "${A[@]}"
  python src/scripts/eval/Vision-Align.py "${A[@]}" --check-complete
  echo "EVAL_SUITE_DONE decoded"
fi
if [[ ,$SUITES, == *,academic,* ]]; then
  A=(academic --checkpoint $CK --manifest $J/academic-manifest-6ff70cf8.json --hf-cache $HUB --output $OUT/academic.json)
  python src/scripts/eval/Vision-Align.py "${A[@]}" --dry-run > $OUT/academic.preflight.json
  [[ -f $OUT/academic.json ]] || $RUN "${A[@]}"
  python src/scripts/eval/Vision-Align.py "${A[@]}" --check-complete
  echo "EVAL_SUITE_DONE academic"
fi
echo "ALL_EVALS_DONE"
