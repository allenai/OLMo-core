#!/bin/bash
# Step 1/2: push the 2k->256k 5-task build from cubbins node-local /data to S3.
# Step 2 (S3 -> weka) is a gantry job from the login host -- an S3 push ALONE leaves every file
# missing on weka while the job still exits 0.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=xl5-s3
#SBATCH --output=/data/prasann/xlong5/logs/s3_upload_%A.log
set -uo pipefail
export AWS_PROFILE=S3
SRC=/data/prasann/xlong5
S3=s3://ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
echo "--- marker sanity before upload (qwen3_5: 248049/248050/248044) ---"
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
python3 - "$SRC/shards" <<'PY'
import sys, glob, os, numpy as np
root = sys.argv[1]; DS, DE, EOS = 248049, 248050, 248044
bad = []
for d in sorted(glob.glob(root + "/*_train")):
    fs = sorted(glob.glob(d + "/token_ids_part_*.npy"))
    if not fs:
        print(f"[{os.path.basename(d)}] NO SHARDS"); bad.append(d); continue
    a = np.fromfile(fs[0], dtype=np.uint32)
    ds, de, eos = int((a == DS).sum()), int((a == DE).sum()), int((a == EOS).sum())
    ok = ds > 0 and ds == de
    if not ok: bad.append(d)
    print(f"[{os.path.basename(d):22s}] part0 toks={len(a):>11,} examples={eos:>6} "
          f"box_start={ds:>8,} box_end={de:>8,} OK={ok}")
sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && { echo "MARKER VERIFY FAILED -- aborting"; exit 3; }

echo; echo "--- sync shards ---"
aws s3 sync "$SRC/shards" "$S3/shards" \
  --exclude "*" --include "*/token_ids_part_*.npy" --include "*/labels_mask_part_*.npy" \
  --include "*/metadata.json"
echo "shards sync rc=$?"

echo; echo "--- sync eval ---"
aws s3 sync "$SRC/eval" "$S3/eval" --exclude "*" --include "*.jsonl"
echo "eval sync rc=$?"

echo; echo "--- README ---"
aws s3 cp "$REPO/debug/xlong_5task/README_weka.md" "$S3/README.md"

echo; echo "=== REMOTE LISTING ==="
aws s3 ls "$S3/" --recursive --human-readable --summarize | tail -25
echo "=== DONE $(date '+%F %T') ==="
