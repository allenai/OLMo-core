#!/usr/bin/env bash
# Beaker (1 B300, text image): golden fwd/bwd + GPU pin tests for the tree in this checkout ("stack")
# against the text pin b5db31dbd cloned at runtime. Big dumps go to Weka, logs to /results.
set -uo pipefail
OUT=${GOLDEN_OUT:-/weka/oe-training-default/jasonr/olmo35-gates/${BEAKER_EXPERIMENT_ID:-local}}
mkdir -p "$OUT" /results
G=$PWD/gates
PIN=/tmp/pin
git clone -q https://github.com/allenai/OLMo-core.git $PIN && git -C $PIN checkout -q b5db31dbd
echo "stack: $(git rev-parse --short HEAD) | pin: $(git -C $PIN rev-parse --short HEAD)" | tee /results/trees.txt
golden() { # name tree [args]
  local name=$1 tree=$2; shift 2
  PYTHONPATH=$tree/src:$G python gates/gate4_golden.py $OUT/golden_$name.pt --repeat 2 --production "$@" > /results/golden_$name.log 2>&1
  echo "EXIT=$?" >> /results/golden_$name.log; tail -1 /results/golden_$name.log
}
cmp() { # out_name a b pins...
  local out=$1 a=$2 b=$3; shift 3
  PYTHONPATH=$G python gates/gate4_golden.py --compare $OUT/golden_$a.pt $OUT/golden_$b.pt --pins "$@" > /results/cmp_$out.log 2>&1
  echo "EXIT=$?" >> /results/cmp_$out.log; grep -E 'GATE 4|forward:|grad criterion|metrics:' /results/cmp_$out.log
}
golden pin1 $PIN
golden pin2 $PIN
golden stack $PWD
cmp pin_stack pin1 stack $OUT/golden_pin2.pt
cmp pin_pin pin1 pin2
golden pin_compile $PIN --compile
golden stack_compile $PWD --compile
cmp compile pin_compile stack_compile $OUT/golden_pin1.pt $OUT/golden_pin2.pt
# GPU pin tests: the pin's own test files against each tree.
PATHS="src/test/nn/ddp src/test/nn/moe src/test/optim src/test/train src/test/nn/attention src/test/nn/transformer src/test/internal"
for pair in pin:$PIN stack:$PWD; do
  n=${pair%%:*}; t=${pair#*:}
  (cd $PIN && PYTHONPATH=$t/src python -m pytest -q --no-header -rfE -p no:cacheprovider -m gpu $PATHS > /results/gate3_gpu_$n.log 2>&1; echo "EXIT=$?" >> /results/gate3_gpu_$n.log)
  tail -2 /results/gate3_gpu_$n.log
done
echo GOLDEN_PAIR_DONE | tee /results/done.txt
