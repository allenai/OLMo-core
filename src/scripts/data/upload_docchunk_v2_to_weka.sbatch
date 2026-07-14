#!/bin/bash
# Verify box markers then upload the single-task v2 doc-chunked DENSE shards from node-local /data
# (mooney) to weka via the real-AWS S3 gateway (s3://ai2-llm == /weka/oe-training-default/ai2-llm).
# Run node-local on mooney (sees /data + has outbound 443). Submit with a dependency on the
# conversion job(s):
#   sbatch --dependency=afterok:<genjob> --job-name=dc-v2-up \
#     --export=ALL,TASKS="contra nq oolong outlier rerank" \
#     src/scripts/data/upload_docchunk_v2_to_weka.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.dcv2up_%x_%j.log

set -uo pipefail
OUT=/data/prasann/single_task_docchunk_v2
S3=s3://ai2-llm/checkpoints/prasanns/single_task_docchunk_v2
TASKS="${TASKS:-contra nq oolong outlier rerank}"
export AWS_PROFILE=S3
echo "=== HOST=$(hostname) upload TASKS='$TASKS' START=$(date '+%F %T') ==="

echo "=== VERIFY box markers before upload ==="
python3 - "$OUT" $TASKS <<'PY'
import sys,glob,numpy as np
out=sys.argv[1]; tasks=sys.argv[2:]
DS=151648; DE=151649; EOS=151643
bad=[]
for name in tasks:
    d=f"{out}/{name}_dense"; fs=sorted(glob.glob(d+"/token_ids_part_*.npy"))
    if not fs: print(f"[{name}] NO SHARDS"); bad.append(name); continue
    a=np.concatenate([np.fromfile(f,dtype=np.uint32) for f in fs])
    ds=int((a==DS).sum()); de=int((a==DE).sum()); eos=int((a==EOS).sum())
    ok = ds>0 and ds==de
    if not ok: bad.append(name)
    print(f"[{name}] toks={len(a)} examples(eos)={eos} box_start={ds} box_end={de} OK={ok}")
sys.exit(1 if bad else 0)
PY
if [ $? -ne 0 ]; then echo "BOX MARKER VERIFY FAILED -- aborting upload"; exit 3; fi

for name in $TASKS; do
  src="$OUT/${name}_dense"; dst="$S3/${name}_dense"
  [ -d "$src" ] || { echo "[$name] MISSING $src -- skip"; continue; }
  echo "=== upload $name -> $dst $(date '+%T') ==="
  aws s3 sync "$src" "$dst" --exclude "*" --include "token_ids_part_*.npy" --include "labels_mask_*.npy"
  echo "  sync rc=$? $(date '+%T')"
done

echo "=== REMOTE LISTING (confirm on s3) ==="
for name in $TASKS; do
  echo "--- ${name}_dense ---"; aws s3 ls "$S3/${name}_dense/" | head
done
echo "=== UPLOAD DONE $(date '+%T') ==="
