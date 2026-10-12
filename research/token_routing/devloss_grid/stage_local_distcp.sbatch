#!/usr/bin/env bash
# oolong/xabsence distcp checkpoints already live on sneetches' own /data; stage_to_node.sh skipped
# them because os.path.exists on their .metadata over /net returned False (mode 0600 + NFS attr
# caching). Copy node-locally instead (no /net), gated on the local sentinel.
#SBATCH --job-name=devloss-stage2
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/data/prasann/joblogs/devloss_stage2_%j.log
set -uo pipefail
D=/data/prasann/devloss_grid/ckpts
for pair in "ctc-oolong-full-4b-fixed:ctc-oolong-full-4b-fixed" "ctc-xabsence-full-4b/step1500:ctc-xabsence-full-4b"; do
  src=/data/prasann/ctc_suite/ckpts/${pair%%:*}; dst=$D/${pair##*:}
  [ -f "$src/model_and_optim/.metadata" ] || { echo "!! no sentinel at $src"; continue; }
  mkdir -p "$dst"; t0=$SECONDS
  rsync -a "$src/model_and_optim/" "$dst/model_and_optim/" && [ -f "$src/config.json" ] && rsync -a "$src/config.json" "$dst/config.json"
  [ -f "$src/../config.json" ] && [ ! -f "$dst/config.json" ] && rsync -a "$src/../config.json" "$dst/config.json"
  chmod -R u+rw,go+r "$dst" 2>/dev/null
  echo "$([ -f "$dst/model_and_optim/.metadata" ] && echo OK || echo FAIL) $src -> $dst ($((SECONDS-t0))s) $(du -sh "$dst" | cut -f1)"
done
