#!/bin/bash
# Swap the scale-K outlier pool into the xlong5 qafter root, PRESERVING the original.
#
#   /weka/.../prasanns/xlong5_2k256k_qwen35_qafter/shards_full/outlier_train
#
# WHAT CHANGES. K (the number of majority topics) now scales with n instead of being pinned, and
# every majority topic exceeds the 3-doc outlier by >=2. The old pool was built through the generic
# xlong expander, whose outlier entry fills from the example's OWN non-gold docs (pool=
# "self_nongold"), so added documents joined topics that already existed and K never grew. That made
# training agree with an eval carrying the identical freeze -- both measuring the fixed-K variant,
# which behaves like an O(N) task, while the O(NM) claim needs M to scale with N.
#
# THE ORIGINAL IS MOVED, NOT DELETED. A collaborator may be mid-run against this path; a swap under
# them has to be reversible. `mv` on the same filesystem is atomic and instant, so the window where
# neither copy is in place is negligible -- as opposed to deleting after a multi-GB copy.
#
# The replacement was gated at build time: its metadata.json was diffed field-by-field against the
# shard it replaces and all 18 structural fields matched (chunk_by=document, doc_markers=false,
# query_position=after, marker_set=qwen3_5, eos=248044, uint32/bool). Content fields differ, as they
# must -- it is different data.
set -uo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core

STAMP=20260812

WORK='
set -uo pipefail
export PATH=/opt/conda/bin:$PATH
command -v aws >/dev/null 2>&1 || python -m pip install -q awscli
mkdir -p ~/.aws
printf "%s" "$AWS_CREDS" > ~/.aws/credentials
printf "%s" "$AWS_CFG" > ~/.aws/config
export AWS_PROFILE=S3

ROOT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35_qafter
LIVE="$ROOT/shards_full/outlier_train"
KEEP="$ROOT/shards_full/outlier_train_prescalek_'"$STAMP"'"
S3=s3://ai2-llm/checkpoints/prasanns/_transfer/outlier_train_scalek_qafter

echo "=== BEFORE ==="
[ -d "$LIVE" ] || { echo "!!! live shard missing at $LIVE -- refusing"; exit 1; }
echo "live: $(ls "$LIVE"/token_ids_part_*.npy 2>/dev/null | wc -l) parts, $(du -sh "$LIVE" | cut -f1)"
echo "--- metadata being replaced ---"
cat "$LIVE/metadata.json"

if [ -e "$KEEP" ]; then
  echo "!!! $KEEP already exists -- a previous swap ran. Refusing to overwrite the preserved copy."
  exit 1
fi

echo ""
echo "=== 1. preserve the original (atomic mv, same filesystem) ==="
mv "$LIVE" "$KEEP" || { echo "!!! preserve FAILED -- nothing changed"; exit 1; }
echo "preserved -> $KEEP"

echo ""
echo "=== 2. pull the replacement from S3 ==="
mkdir -p "$LIVE"
if ! aws s3 sync "$S3" "$LIVE" --only-show-errors; then
  echo "!!! sync FAILED -- ROLLING BACK"
  rm -rf "$LIVE"; mv "$KEEP" "$LIVE"
  echo "rolled back; original restored at $LIVE"
  exit 1
fi

echo ""
echo "=== 3. verify the replacement is complete and self-consistent ==="
python - "$LIVE" <<"PYEOF" || { echo "!!! verify FAILED -- ROLLING BACK"; rm -rf "$LIVE"; mv "$KEEP" "$LIVE"; echo "rolled back"; exit 1; }
import json, os, sys, glob
d = sys.argv[1]
m = json.load(open(os.path.join(d, "metadata.json")))
tok = sorted(glob.glob(os.path.join(d, "token_ids_part_*.npy")))
lab = sorted(glob.glob(os.path.join(d, "labels_mask_part_*.npy")))
print(f"  parts: {len(tok)} token_ids / {len(lab)} labels_mask")
assert len(tok) == len(lab) and tok, "token/label part count mismatch or empty"
# contiguous numbering -- an orphan from an earlier upload would show up as a gap or an extra
idx = [int(p.rsplit("_", 1)[1].split(".")[0]) for p in tok]
assert idx == list(range(len(idx))), f"non-contiguous parts: {idx}"
# uint32 token ids, bool mask -> 4 bytes vs 1 byte for the same element count
bt = sum(os.path.getsize(p) for p in tok); bl = sum(os.path.getsize(p) for p in lab)
print(f"  bytes: {bt/1e9:.2f} GB token_ids / {bl/1e9:.2f} GB labels_mask")
assert bt == 4 * bl, f"token/label byte ratio {bt/bl:.2f}, expected 4.0"
assert bt // 4 == m["num_tokens"], "bytes imply {} tokens, metadata says {}".format(bt // 4, m["num_tokens"])
# DOUBLE QUOTES ONLY in this payload. The whole WORK string is shell-single-quoted, so a Python
# single quote closes it and the quote characters are stripped before Python sees them -- that turned
# m["num_instances"] into a bare name and failed the verify (harmlessly: it rolled back cleanly).
print("  num_instances={} num_tokens={}".format(m["num_instances"], m["num_tokens"]))
print("  chunk_by={} doc_markers={} query_position={}".format(
    m["chunk_by"], m["doc_markers"], m["query_position"]))
print("  OK: parts contiguous, sizes agree with metadata")
PYEOF

echo ""
echo "=== AFTER ==="
echo "live: $(ls "$LIVE"/token_ids_part_*.npy | wc -l) parts, $(du -sh "$LIVE" | cut -f1)"
echo "--- new metadata ---"
cat "$LIVE/metadata.json"

echo ""
echo "=== siblings (the other four tasks are untouched) ==="
ls -la "$ROOT/shards_full/"

echo ""
echo "SWAP COMPLETE. To revert:"
echo "  rm -rf $LIVE && mv $KEEP $LIVE"
'

gantry run --name outlier-weka-swap -w ai2/flex2 -b ai2/oe-other \
  --cluster 'ai2/*-cirrascale*' --gpus 0 --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --weka oe-training-default:/weka/oe-training-default \
  --env-secret AWS_CREDS=PRASANNS_AWS_CREDENTIALS --env-secret AWS_CFG=PRASANNS_AWS_CONFIG \
  --no-python --allow-dirty --timeout 0 --yes -- bash -c "$WORK"
