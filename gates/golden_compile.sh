#!/usr/bin/env bash
# Beaker (1 B300): compiled goldens for pin x2 (floor), vision+#885 (59a70530e) and the stack (this checkout).
set -uo pipefail
OUT=${GOLDEN_OUT:-/weka/oe-training-default/jasonr/olmo35-gates/${BEAKER_EXPERIMENT_ID:-local}}
mkdir -p "$OUT" /results
G=$PWD/gates
git clone -q https://github.com/allenai/OLMo-core.git /tmp/pin && git -C /tmp/pin checkout -q b5db31dbd
git clone -q https://github.com/allenai/OLMo-core.git /tmp/vb && git -C /tmp/vb checkout -q 59a70530e
echo "stack: $(git rev-parse --short HEAD) | pin: b5db31dbd | visionbase: 59a70530e" | tee /results/trees.txt
golden() { local name=$1 tree=$2; shift 2
  PYTHONPATH=$tree/src:$G python gates/gate4_golden.py $OUT/golden_$name.pt --repeat 2 --production --compile "$@" > /results/golden_$name.log 2>&1
  echo "EXIT=$?" >> /results/golden_$name.log; tail -1 /results/golden_$name.log; }
cmp() { local out=$1 a=$2 b=$3; shift 3
  PYTHONPATH=$G python gates/gate4_golden.py --compare $OUT/golden_$a.pt $OUT/golden_$b.pt --pins "$@" > /results/cmp_$out.log 2>&1
  echo "EXIT=$?" >> /results/cmp_$out.log; echo "== $out"; grep -E 'GATE 4|forward:|grad criterion|metrics:|^DIFF' /results/cmp_$out.log | head -12; }
golden pinc1 /tmp/pin
golden pinc2 /tmp/pin
golden vbc /tmp/vb
golden stackc $PWD
cmp pin_pin pinc1 pinc2
cmp pin_visionbase pinc1 vbc $OUT/golden_pinc2.pt
cmp pin_stack pinc1 stackc $OUT/golden_pinc2.pt
cmp visionbase_stack vbc stackc $OUT/golden_pinc1.pt $OUT/golden_pinc2.pt
echo GOLDEN_COMPILE_DONE | tee /results/done.txt
