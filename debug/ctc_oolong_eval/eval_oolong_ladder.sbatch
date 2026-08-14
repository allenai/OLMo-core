#!/bin/bash
# Eval one arm of the OOLONG 4B pair (VESSL-trained, REBUILT clean shard) on the 2k-32k CTC ladder.
#
#   sbatch --export=ALL,RUN=ctcms-oolong-full-4b-vsl,ARM=full,RUNGS="2048" \
#          debug/ctc_oolong_eval/eval_oolong_ladder.sbatch      # canary
#   sbatch --export=ALL,RUN=ctcms-oolong-full-4b-vsl,ARM=full \
#          debug/ctc_oolong_eval/eval_oolong_ladder.sbatch      # full ladder
#
# Derived from debug/ctc_modelscale/eval_smallscale_ladder.sbatch; the lambda-pull and
# serving-copy-build stages are dropped because both checkpoints were staged off VESSL and their
# serving copies built ahead of time. Everything else (rung mirroring under flock, stale-grade
# quarantine, serving census, derive-ladder-from-what-exists) is kept -- each of those guards a
# failure that has actually happened on this ladder.
#
# ⚠ QUERY_POSITION=after IS LOAD-BEARING, NOT COSMETIC.
# The oolong_train_ctc shard was tokenized with query_position="after" (metadata.json), but
# build_prefills_any.py's default -- and therefore every previous run of this driver -- renders
# "both", which prepends a second copy of the question INTO THE FREE PREFIX. For the chunked arm
# that prefix attends globally, so the model would be handed the question broadcast to every chunk
# in a layout it never trained on. Measured cost: +52 tokens/example at rung 2048, chunk counts
# unchanged (the delta is entirely free-prefix). This is the same class of mismatch that produced
# contradiction's 0.559-vs-0.946 confusion. BOTH ARMS must use the same value or the dense-vs-
# chunked delta is a prompt-layout artifact rather than a result.
# The drivers verify the rendered value against the request and hard-fail on mismatch, so a stale
# node-local repo copy (which is how this would silently regress) cannot pass silently.
#
# ⚠ COT_MODE=none matches the shard. For oolong this is verified to be a NO-OP on the rendered
# prefill (plan and none produce byte-identical tokens -- oolong's prompt carries no CoT scaffold),
# but it is passed explicitly so the artifact records the intended value rather than "plan".
#
#SBATCH --job-name=ev-oolong
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/data/prasann/joblogs/oolong_eval_%x_%j.log

set -uo pipefail
# ⚠ NO BRACES IN THIS ERROR MESSAGE. bash closes a ${VAR:?msg} expansion at the FIRST unescaped
# `}`, so the obvious `${RUN:?set RUN=ctcms-oolong-{full,cmix}-4b-vsl}` terminates at the brace
# after `cmix` and appends the trailing `-4b-vsl}` to the VALUE -- producing a literal directory
# named `ctcms-oolong-full-4b-vsl-4b-vsl}`. Silent, because the message only renders when RUN is
# unset, so the corruption appears exactly when everything else is correct.
RUN="${RUN:?set RUN to the run name, e.g. ctcms-oolong-full-4b-vsl or ctcms-oolong-cmix-4b-vsl}"
ARM="${ARM:?set ARM=full|cmix}"
RUNGS="${RUNGS:-2048 4096 8192 16384 32768}"

source /data/prasann/node_local_env.sh
LOCAL_REPO=/data/prasann/repo/OLMo-core
BASE_SNAP=/data/prasann/hf_models/Qwen3.5-4B-Base

# ⚠ THE DENSE PATH IS *NOT* THE DRIVER'S DEFAULT, DELIBERATELY.
# gpu_eval_task.sh defaults SERVE to vllm_serving_4b/ctc-4b-${TASK}-full -- i.e.
# `ctc-4b-oolong-full`, which on sneetches is ALREADY OCCUPIED by an unrelated CTC-suite oolong
# model built 2026-07-22 from a different checkpoint (427 tensors, no visual tower). Letting the
# default resolve would have silently evaluated that July model and reported it as the VESSL run:
# right task name, wrong weights, and nothing downstream would have flagged it. The new dense
# serving copy is therefore named after its checkpoint and passed explicitly.
case "$ARM" in
  full) SERVE=/data/prasann/ctc_suite/vllm_serving_4b/ctcms-oolong-full-4b-vsl ;;
  cmix) SERVE=/data/prasann/ctc_suite/vllm_serving_4b_chunked/ctc-4b-oolong-cmix ;;
  *)    echo "FATAL: unknown ARM=$ARM"; exit 2 ;;
esac

echo "=== HOST=$(hostname) RUN=$RUN ARM=$ARM SERVE=$SERVE START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
[ -d "$BASE_SNAP" ] || { echo "FATAL: base snapshot $BASE_SNAP missing"; exit 2; }
[ -f "$SERVE/model.safetensors" ] || { echo "FATAL: serving copy $SERVE not built"; exit 4; }

# ---- serving-copy census (the recipe's own acceptance test) --------------------------------
# model.visual.* AND visual.* both being present is CORRECT -- make_vl_weights grafts the base
# vision tower as model.visual.*, add_dummy_visual adds the visual.* tower vLLM's weight tracker
# looks for. Known-good 4B: language_model=426 / model.visual=297 / visual=297.
python3 - "$SERVE" <<'PY'
import json, os, sys
from safetensors import safe_open
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
with safe_open(os.path.join(d, "model.safetensors"), framework="pt") as f:
    keys = list(f.keys())
lm  = sum(k.startswith("model.language_model.") for k in keys)
vis = sum(k.startswith("visual.") for k in keys)
mv  = sum(k.startswith("model.visual.") for k in keys)
ok = lm > 0 and vis > 0 and "vision_config" in cfg
print(f"  SERVE CENSUS: language_model={lm} model.visual={mv} visual={vis} total={len(keys)} "
      f"vision_config={'yes' if 'vision_config' in cfg else 'NO'} arch={cfg.get('architectures')}")
print("SERVE_OK" if ok else "SERVE_BAD")
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo "FATAL: serving copy failed its census"; exit 4; }

# ---- eval rungs must be node-local ---------------------------------------------------------
# Mirror (not "skip if non-empty"): rsync --delete from the authoritative /scratch copy is
# idempotent and self-heals a dir a previous buggy run poisoned. flock because both arms are
# submitted together and share this one directory.
# RUNG_TREE selects which staged tree the rungs come from. Default `eval_rungs` is the shipped
# 2k-32k ladder; the length-generalization study passes `eval_rungs_lengthgen`, which is a SEPARATE
# tree precisely so a 64k/128k rung can never be confused with a shipped one.
RUNG_TREE="${RUNG_TREE:-eval_rungs}"
RUNG_TASK=oolong
RUNGDIR_LOCAL=/data/prasann/ctc_suite_staged/${RUNG_TREE}/${RUNG_TASK}
RUNG_SRC=/scratch/users/prasann/ctc_suite_staged/${RUNG_TREE}/${RUNG_TASK}
[ -d "$RUNG_SRC" ] || { echo "FATAL: no rung source $RUNG_SRC"; exit 5; }
mkdir -p "$(dirname "$RUNGDIR_LOCAL")"
(
  flock -x -w 900 9 || { echo "FATAL: timed out waiting for the rung-staging lock"; exit 5; }
  echo "--- mirroring eval rungs $RUNG_SRC -> $RUNGDIR_LOCAL ---"
  mkdir -p "$RUNGDIR_LOCAL"
  rsync -a --delete "$RUNG_SRC/" "$RUNGDIR_LOCAL/" || { echo "FATAL: rung staging failed"; exit 5; }
) 9>"/data/prasann/.rungstage_${RUNG_TASK}.lock"
[ $? -ne 0 ] && exit 5
ls -la "$RUNGDIR_LOCAL"

# Derive the ladder from what exists, and say what was dropped -- silently shrinking a curve is
# worse than failing, because a 4-rung plot next to 5-rung plots reads as "32k covered".
HAVE=""; DROPPED=""
for R in $RUNGS; do
  if [ -f "$RUNGDIR_LOCAL/rung_${R}.jsonl" ]; then HAVE="$HAVE $R"; else DROPPED="$DROPPED $R"; fi
done
HAVE="${HAVE# }"; DROPPED="${DROPPED# }"
[ -n "$DROPPED" ] && echo "!!! RUNGS DROPPED:$DROPPED -- report this ladder as covering '$HAVE' only"
[ -n "$HAVE" ] || { echo "FATAL: no requested rung exists in $RUNGDIR_LOCAL"; exit 5; }
RUNGS="$HAVE"
echo "--- final RUNGS='$RUNGS' ---"

# ---- eval ----------------------------------------------------------------------------------
export SERVE BASE_SNAP RUNGS
export TASK=oolong RUNG_TASK=oolong EVAL_TASK=oolong
# TP: vLLM tensor-parallel degree (see gpu_eval_task.sh for the head-count divisibility rule).
# The #SBATCH --gres above requests ONE H200, which is right for the 2k-32k ladder. TP>1 therefore
# requires overriding it at submit time -- `sbatch --gres=gpu:H200:2 ... TP=2` -- because a TP=2
# engine on a 1-GPU allocation does not fall back, it dies at init. Assert it here rather than
# 15 minutes into prefill-building.
export TP="${TP:-1}"
NGPU_VIS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$NGPU_VIS" -ge "$TP" ] || { echo "FATAL: TP=$TP but only $NGPU_VIS GPU(s) visible -- resubmit with --gres=gpu:H200:$TP"; exit 6; }
echo "--- TP=$TP over $NGPU_VIS visible GPU(s) ---"
export RUNGDIR="$RUNGDIR_LOCAL"
export COT_MODE=none
export QUERY_POSITION=after          # see header -- must match the training shard, both arms
# ⚠ WITHOUT THIS THE WHOLE LADDER SCORES ~0 ON A MODEL THAT KNOWS THE ANSWERS.
# The prefill pack's eos_token_id is 248044 (`<|endoftext|>`), but our SFT assistant turns end with
# `<|im_end|>` = 248046. vLLM's stop_token_ids only carried the former, so the model emitted its
# answer, signalled end-of-turn, and then kept sampling to max_new_tokens -- degenerating into
# "1111...", "positivepositive...". OOLONG's answer is a bare number/label with no newline, so
# truncate_generic's text-level rescue (which is what silently saves the other tasks) has nothing
# to cut on, and the grader scored the entire ramble: measured 0.0055 at rung 2048 before the fix.
export EXTRA_STOP_TOKEN_IDS=248046
export DUMP_RAW_GENS="${DUMP_RAW_GENS:-1}"   # keep untruncated decodes; this failure was only
                                             # diagnosable because the generations were retained
export KEEP_RESPONSES="${KEEP_RESPONSES:-1}"   # keep generations: oolong's newline stop rule + the
                                               # trailing <think> is a known truncation trap
                                               # (qwen35-think-newline-stop), so the raw text must
                                               # be inspectable if a rung looks anomalous.
# OUTDIR MUST be per-RUN: it defaults to .../${TASK}, which is identical for both arms, so the two
# arms would overwrite each other's prefills and grades in place.
export OUTDIR=/data/prasann/ctc_oolong_eval/results/${RUN}
mkdir -p "$OUTDIR"

# Quarantine pre-existing grades: every downstream copy is a grade_*.json GLOB, so a leftover file
# from an earlier eval set gets republished as if this job produced it.
STALE=$(ls "$OUTDIR"/grade_*.json 2>/dev/null | wc -l)
if [ "$STALE" -gt 0 ]; then
  QUAR="$OUTDIR/superseded_$(date '+%Y%m%dT%H%M%S')"; mkdir -p "$QUAR"
  mv "$OUTDIR"/grade_*.json "$QUAR/" 2>/dev/null
  echo "--- quarantined $STALE pre-existing grade json(s) -> $QUAR ---"
fi

# ---- optional: fast chunked prefill (VARLEN=1) ---------------------------------------------
# CHUNK_VARLEN_PREFILL routes prefill-only steps through flash_attn_varlen_func instead of
# rebuilding a FlexAttention block mask, measured at 19.45x (rung 2560) and 15.45x (rung 8192) on
# contradiction 4B at eval_size=500, with set_f1 moving only +0.0014/-0.0020. Prefill dominates at
# 64k/128k, which is exactly where the fallback path becomes unaffordable.
#
# ⚠ THE DRIVER CLOBBERS THE CONCURRENCY KNOBS, so exporting them directly does nothing.
# gpu_eval_task_chunked.sh sets CHUNK_MAX_NUM_SEQS/CHUNK_SEQ_HEADROOM to 128/130 for rung<=8192 and
# UNSETS them above that (falling back to 8/10). The only way to pin 16/18 at every rung is to make
# the "fast" branch always taken (CHUNK_FAST_MAX_RUNG huge) and set the *_FAST variables it reads.
# Keep seqs at 16: 64 measured WORSE at 8k (9.86x vs 15.45x) because it fell back on most steps.
#
# ⚠ CHUNK_MAX_BATCHED_TOKENS MUST COVER THE LONGEST PROMPT IN THE RUNG, not the rung label -- a
# prefill longer than MBT is split across steps, which makes it ineligible for the varlen plan and
# silently drops back to the slow path. Size it from the MEASURED max, which is why the length-gen
# study runs ONE RUNG PER JOB rather than looping.
if [ "${VARLEN:-0}" = "1" ] && [ "$ARM" = "cmix" ]; then
  export CHUNK_VARLEN_PREFILL=1
  export CHUNK_FAST_MAX_RUNG=99999999
  export CHUNK_MAX_NUM_SEQS_FAST="${CHUNK_MAX_NUM_SEQS_FAST:-16}"
  export CHUNK_SEQ_HEADROOM_FAST="${CHUNK_SEQ_HEADROOM_FAST:-18}"
  [ -n "${CHUNK_MAX_BATCHED_TOKENS:-}" ] && export CHUNK_MAX_BATCHED_TOKENS
  export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
  echo "--- VARLEN ON: seqs=$CHUNK_MAX_NUM_SEQS_FAST headroom=$CHUNK_SEQ_HEADROOM_FAST" \
       "MBT=${CHUNK_MAX_BATCHED_TOKENS:-<unset>} gpu_mem_util=$GPU_MEM_UTIL ---"
  echo "--- after the run, CONFIRM patch_debug shows varlen >> fallback; if fallback dominates," \
       "the plan did not engage and the timing (not the metric) is invalid ---"
fi

[ "$ARM" = "full" ] && DRIVER=gpu_eval_task.sh || DRIVER=gpu_eval_task_chunked.sh
echo "--- driver=$DRIVER rungs='$RUNGS' query_position=$QUERY_POSITION cot=$COT_MODE ---"
bash "$LOCAL_REPO/debug/ctc_vllm_validation/node_local/$DRIVER"
RC=$?

echo ""
echo "=== GRADES ($RUN) ==="
for R in $RUNGS; do
  g="$OUTDIR/grade_${R}.json"
  [ -f "$g" ] || { echo "  grade_${R}.json  MISSING -- rung $R produced no grade"; continue; }
  python3 -c "
import json,math
d=json.load(open('$g'))
sz=d.get('eval_size', d.get('n')) or 0
v=d.get('metric_value')
se=math.sqrt(v*(1-v)/sz) if isinstance(v,(int,float)) and 0<=v<=1 and sz else float('nan')
warn='' if sz>=500 else f'   <<< eval_size={sz} only -- FLAG INLINE'
print(f\"  rung {'$R':>6}  {d.get('metric_name')}={v}  SE=±{se:.4f}  parse_rate={d.get('parse_rate')}  eval_size={sz}  qpos={d.get('query_position')}{warn}\")
" 2>/dev/null
done

ACC=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ctc_oolong_eval/results/${RUN}
mkdir -p "$ACC"
for R in $RUNGS; do
  [ -f "$OUTDIR/grade_${R}.json" ] && cp "$OUTDIR/grade_${R}.json" "$ACC/"
done
echo "=== DONE $(date '+%F %T') rc=$RC -> $ACC ==="
exit $RC
