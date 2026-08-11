#!/usr/bin/env bash
# Port validation: reproduce a known contradiction @2k number with the ctc package.
#
# TARGET (results/ctc_suite/all_results.jsonl, 2026-07-19, git e57aaea6c):
#   ckpt      /data/prasann/ctc_suite/ckpts/ctc-t0-contra-n20-full   (step750)
#   data      /data/prasann/ctc_suite_data/eval_rungs/contradiction_eval_pubmed_both_n20_k3.jsonl
#   arm       full (plain causal)
#   f1        0.9583333333333334      exact_match 0.8770491803278688
#   parse_rate 1.0                    eval_size   488
#   max_length 4096                   max_new     512
#
# Both artifacts live on CUBBINS, hence --nodelist. This is a reproduction check, not a science
# run: the question is whether the ported code gets the same number from the same inputs. If the
# FEVER filler leak affects this eval file it changes what 0.9583 MEANS, not whether we reproduce
# it.
#
# Expected deviations, and what each would tell us:
#   * f1 within ~0.001            -> the port is faithful.
#   * f1 moves, parse_rate 1.0    -> most likely stop="pairs": the old run used stop_rule="eos",
#                                    whose should_stop returns False unconditionally, so it decoded
#                                    to the budget and scored whatever the parser made of the tail.
#                                    Re-run with --stop-preset eos to separate the two.
#   * parse_rate < 1.0            -> prompt assembly or tokenization differs. Read the generations.
#
#SBATCH --job-name=ctc-validate-contra2k
#SBATCH --nodelist=cubbins
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/ctc_validate_contra2k_%j.log

set -euo pipefail

# Job logs, caches and work dirs all on node-local /data -- NFS deadlocks concurrent readers.
mkdir -p /data/prasann/joblogs
source "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'Command=\K[^ ]+')")/_env.sh" \
  2>/dev/null || source /accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core/run/_env.sh

CKPT=/data/prasann/ctc_suite/ckpts/ctc-t0-contra-n20-full/step750
DATA=/data/prasann/ctc_suite_data/eval_rungs/contradiction_eval_pubmed_both_n20_k3.jsonl
OUT=/data/prasann/ctc_validate/contra_2k_full.json

for p in "$CKPT" "$DATA"; do
  [[ -e "$p" ]] || { echo "MISSING on $(hostname): $p" >&2; exit 1; }
done

echo "[validate] host=$(hostname) python=$CTC_PYTHON"
echo "[validate] target f1=0.9583333333333334 em=0.8770491803278688 eval_size=488"

exec "$CTC_PYTHON" -m ctc.eval.validate \
  --ckpt "$CKPT" \
  --task contradiction \
  --rung 2k \
  --data "$DATA" \
  --attn full \
  --tokenizer Qwen/Qwen3-4B \
  --max-length 4096 \
  --out "$OUT" \
  --expect-f1 0.9583333333333334 \
  --expect-eval-size 488
