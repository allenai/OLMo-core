#!/usr/bin/env bash
# Push the hybridish SFT checkpoints from horton's node-local /data to the S3 buffer.
#
# Beaker cannot read this machine's disks and weka is not mounted here, so the transfer is the
# documented two-step: S3 from here, then a gantry job syncs S3 -> weka. Run FROM horton -- the
# checkpoints are node-local there, and pushing via /net would move ~7GB over the ~5 MB/s NFS link.
set -euo pipefail
S3=s3://ai2-llm/checkpoints/prasanns/ctc_hybridish_sft
srun --partition=berkeleynlp --nodelist=horton --cpus-per-task=8 --mem=32G --time=3:00:00 \
  bash -lc "
set -euo pipefail
for c in sft_4to1_ml_hf sft_7to1_ml_hf; do
  echo \"[push] \$c (\$(du -sh /data/prasann/hybridish/\$c | cut -f1))\"
  aws s3 sync /data/prasann/hybridish/\$c $S3/\$c --only-show-errors
  aws s3 ls $S3/\$c/ | sed 's/^/      /'
done
echo '[push] done'"
