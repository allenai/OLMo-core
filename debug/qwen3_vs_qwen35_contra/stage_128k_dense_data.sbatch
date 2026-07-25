#!/bin/bash
# Stage the dense (qwen3-tokenized) 128k contradiction mix onto horton's node-local /data.
#
# Why S3 and not /scratch: build_128k_data.sbatch wrote the shards to the BUILD node's /data and
# copied only `metadata.json` to /scratch -- so the /scratch dir looks like a valid shard dir but
# holds zero token parts. Pointing a trainer at it would fail late (or, worse, look like a data
# problem). S3 is the only complete copy reachable from here; the dense arm previously trained on
# Beaker straight off weka, which is why this was never needed locally before.
#
# Target is DATA_LOCAL as run_ctc_local.sbatch computes it ($ROOT/data/<basename DATA_SRC>), so the
# training job can pass DATA_SRC=<this path> and its `cp -un` becomes a no-op instead of a second
# multi-GB copy.
#SBATCH --job-name=stage128kdense
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=horton
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/stage128kdense_%j.log
set -uo pipefail
DEST=/data/prasann/ctc_suite/data/contra_mix_qwen3_10k_128k
SRC=s3://ai2-llm/checkpoints/prasanns/ctc_suite/shards/contra_mix_qwen3_10k_128k
mkdir -p "$DEST" /data/prasann/joblogs
echo "=== staging $SRC -> $DEST on $(hostname) $(date '+%F %T') ==="
aws s3 sync "$SRC" "$DEST" --only-show-errors || { echo "FATAL: s3 sync failed"; exit 2; }

# Verify completeness rather than trusting sync's exit code: a partial dir is exactly the failure
# mode that produced the empty /scratch copy this script exists to work around.
python3 - "$DEST" <<'EOF' || exit 3
import json, os, sys
d = sys.argv[1]
meta = json.load(open(os.path.join(d, "metadata.json")))
tok = sorted(f for f in os.listdir(d) if f.startswith("token_ids_part_"))
lab = sorted(f for f in os.listdir(d) if f.startswith("labels_mask_part_"))
if not tok or len(tok) != len(lab):
    sys.exit(f"FATAL: {len(tok)} token parts vs {len(lab)} label parts")
nbytes = sum(os.path.getsize(os.path.join(d, f)) for f in tok)
# token_ids are uint32 on disk; metadata counts TOKENS, so bytes must be exactly 4x.
if nbytes != 4 * meta["num_tokens"]:
    sys.exit(f"FATAL: {nbytes} bytes != 4 * num_tokens ({4*meta['num_tokens']}) -> partial download")
print(f"[verify] {len(tok)} parts, {nbytes/1e9:.2f} GB, num_instances={meta['num_instances']}, "
      f"num_tokens={meta['num_tokens']}, max_example_len={meta['max_example_len']} -- COMPLETE")
EOF
echo "=== stage DONE $(date '+%F %T') ==="
