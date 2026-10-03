#!/usr/bin/env bash
# Beaker (1 GPU, training image): build the Molmo2 Stage-1 loader for vision 05885efa4 and #902 0075fa480 inside the
# training image and compare 30 batches on all 8 data-parallel ranks. Results to /results.
set -uo pipefail
mkdir -p /results/dumps
G=$PWD/gates
for pair in vision:05885efa4 infra:0075fa480; do
  n=${pair%%:*}; sha=${pair#*:}
  git clone -q https://github.com/allenai/OLMo-core.git /tmp/$n && git -C /tmp/$n fetch -q origin $sha && git -C /tmp/$n checkout -q $sha
  echo "$n -> $(git -C /tmp/$n rev-parse --short HEAD)" | tee -a /results/trees.txt
done
python -c "import torch, PIL, numpy, datasets; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'PIL', PIL.__version__, 'numpy', numpy.__version__, 'datasets', datasets.__version__)" | tee /results/env.txt
for n in vision infra; do
  for r in 0 1 2 3 4 5 6 7; do
    (cd /tmp/$n && PYTHONPATH=/tmp/$n/src:$G python $G/loader_parity.py /tmp/$n 1 $r 30 /results/dumps/s1_${n}_r${r}.json > /results/dumps/s1_${n}_r${r}.log 2>&1) &
  done
done
wait
for r in 0 1 2 3 4 5 6 7; do
  echo "== rank $r" >> /results/compare.txt
  python $G/loader_parity.py --compare /results/dumps/s1_vision_r${r}.json /results/dumps/s1_infra_r${r}.json >> /results/compare.txt 2>&1
  echo "rank $r compare exit $?" | tee -a /results/summary.txt
done
python - <<'PY' | tee -a /results/summary.txt
import json
def sig(x): return {k: v.get('sha256') for k, v in x['tensors'].items()}
for r in range(8):
    try:
        a = json.load(open(f'/results/dumps/s1_vision_r{r}.json'))['batches']; b = json.load(open(f'/results/dumps/s1_infra_r{r}.json'))['batches']
    except Exception as e:
        print(f'rank {r}: missing ({e})'); continue
    d = [i for i, (x, y) in enumerate(zip(a, b)) if sig(x) != sig(y)]
    print(f'rank {r}: {min(len(a), len(b))} batches, {len(d)} differ' + (f', first at step {d[0] + 1}' if d else ''))
PY
echo INCONTAINER_PARITY_DONE | tee /results/done.txt
