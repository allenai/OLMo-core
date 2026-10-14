#!/bin/bash
# CPU Beaker job: rebuild the setA shards with hashed document ids (rebuild_doc_id_hash.py) into
# <set>_hashid on weka. Runs the PUSHED commit of the current branch -- commit + push first.
#   bash recipes/data/ctc_sft/rebuild_doc_id_hash_beaker.sh            # setA_max20_evaliid
#   SRC=<set root> DST=<new root> bash recipes/data/ctc_sft/rebuild_doc_id_hash_beaker.sh
# Then: ops register --kind data --cluster beaker --job-id <EXP id> ...; check with ops status.
set -uo pipefail
SETS=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_sft_sets
SRC="${SRC:-$SETS/setA_max20_evaliid}"
DST="${DST:-${SRC%/}_hashid}"
CPUS="${CPUS:-48}"
read -r -d '' WORK <<EOW
set -uo pipefail; export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
# gantry's "--install true" venv is empty; the image's conda env has numpy/transformers but not
# every olmo_core import-time dep -- install whatever the converter's import chain lacks.
PYBIN=/opt/conda/bin/python; export PYTHONPATH=src
for i in 1 2 3 4 5 6; do
  MISSING=\$(\$PYBIN -c "import olmo_core.data.document_chunk_landmark, olmo_core.progress" 2>&1 | grep -oE "No module named '[^']+'" | sed "s/No module named '//; s/'//" | cut -d. -f1)
  [ -z "\$MISSING" ] && break
  echo "installing missing dep: \$MISSING"; \$PYBIN -m pip install -q "\$(echo \$MISSING | tr _ -)" 2>&1 | tail -1
done
\$PYBIN -c "import olmo_core.data.document_chunk_landmark, numpy, transformers; print('imports OK', transformers.__version__)" || { echo "!!! deps unresolved"; exit 1; }
git log -1 --format='commit %H %s'
\$PYBIN -u recipes/data/ctc_sft/rebuild_doc_id_hash.py --src-root $SRC --dst-root $DST --cpus $CPUS
EOW
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
gantry run --name "ctc-setA-hashid-rebuild-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune-cirrascale --cluster ai2/ceres-cirrascale --cluster ai2/saturn-cirrascale \
  --cluster ai2/jupiter-cirrascale-2 --gpus 0 --cpus "$CPUS" --memory 180GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install true \
  --weka oe-training-default:/weka/oe-training-default \
  --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -E "beaker.org/ex|rror" | head -3
