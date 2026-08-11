#!/bin/bash
# ON-BEAKER multi-rung NATIVE long-context eval runner (8-GPU data-parallel torchrun).
#
# This is the on-node half of the Beaker eval flow. It is uploaded to the weka eval bundle by
# `upload_lc_eval_bundle.sh` and invoked inside a gantry job by `run_q4b_beaker_multirung_eval.py`.
# It mirrors the LOCAL driver `run_q4b_stl_multirung_eval.sbatch`, but reads EVERYTHING from weka
# (eval code + eval data + the checkpoint) so nothing has to be synced to the Beaker node:
#
#   * eval CODE   : corpus-reasoning `scripts/` tree under  $BUNDLE                (PYTHONPATH + cwd)
#   * `data/...`  : relative ladder/base files under         $BUNDLE/data          (--root=$BUNDLE)
#   * `_600/_500` : the goal-rung ladder files under          $EVAL500             (EVAL500_ROOT env)
#   * checkpoint  : the just-trained distcp step dir under     $RUN_DIR/step*       (auto-globbed)
#
# Native olmo_core generate (NO HF / NO vLLM) so it works for dense / landmark / compressive /
# docchunk, and runs 8-way DP via `torchrun --nproc_per_node=8`.
#
# Env in (set by the launcher):
#   RUN         run name (checkpoints live at $WEKA_LLM/checkpoints/prasanns/$RUN/step*)
#   TASK        contra | nq | rerank | outlier | oolong
#   VARIANT     dense | landmark | compressive | docchunk   (docchunk -> OOLONG only)
#   WEKA_LLM    weka ai2-llm root (e.g. /weka/oe-training-default/ai2-llm)
#   STEP        optional: pin a specific step dir (e.g. step580); empty -> latest complete step
#   MAX_TEST    default 600 ; MAX_LENGTH default 40960 ; BATCH_SIZE default 8 ; NGPU default 8
set -uo pipefail
TASK="${TASK:?set TASK=contra|nq|rerank|outlier|oolong}"
VARIANT="${VARIANT:?set VARIANT=dense|landmark|compressive|docchunk}"
RUN="${RUN:?set RUN=<run name>}"
WEKA_LLM="${WEKA_LLM:?set WEKA_LLM=<weka ai2-llm root>}"
STEP="${STEP:-}"
MAX_TEST="${MAX_TEST:-600}"
MAX_LENGTH="${MAX_LENGTH:-40960}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NGPU="${NGPU:-8}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-4B}"
# Resolve a Hub repo id to its weka-staged copy so the job never touches huggingface.co at startup.
# AutoTokenizer.from_pretrained(<repo id>) is a live network call, and a brief Hub outage or a cold
# node cache kills the job instantly ("We couldn't connect to 'https://huggingface.co'"). On
# 2026-08-10 that took out 20 of 30 eval jobs, some after hours of GPU work. Stage with
# src/scripts/data/stage_tokenizers_weka.py; the '/' -> '__' mapping must match that script.
# If the staged copy is missing we fall back to the Hub id (same tokenizer, just fragile) and say so
# loudly, rather than hard-failing a sweep over a path typo.
TOKENIZER_ROOT="${TOKENIZER_ROOT:-/weka/oe-training-default/amandab/tokenizers}"
case "$TOKENIZER" in
  /*)
    # An absolute path is an explicit local tokenizer -- never rewrite it, even if it is missing
    # (mangling a typo'd path into a "staged" name would only obscure the real error).
    : ;;
  */*)
    _staged="$TOKENIZER_ROOT/${TOKENIZER//\//__}"
    if [ -d "$_staged" ]; then
      echo "    [tokenizer] using weka-staged copy: $TOKENIZER -> $_staged"
      TOKENIZER="$_staged"
    else
      echo "    [tokenizer] WARNING: no staged copy at $_staged -- falling back to the Hub id" \
           "'$TOKENIZER'. This job now depends on huggingface.co being reachable; stage it with" \
           "src/scripts/data/stage_tokenizers_weka.py to make that impossible." >&2
    fi ;;
esac
PROMPT_FORMAT="${PROMPT_FORMAT:-chat}"   # chat=SFT (apply_chat_template) | raw=BASE/CPT | alpaca=legacy
# GQA compressive-landmark checkpoints only: share top-k landmark block selection across each KV
# group's query heads ("mean"/"max") instead of each head retrieving independently. Empty (default)
# keeps independent per-head selection; only takes effect with top-k decode enabled (on by default
# via GenerationConfig.landmark_top_k_fraction).
LANDMARK_GROUP_SELECTION="${LANDMARK_GROUP_SELECTION:-}"
GROUP_ARGS=""
[ -n "$LANDMARK_GROUP_SELECTION" ] && GROUP_ARGS="--landmark-group-selection $LANDMARK_GROUP_SELECTION"
# GROUPED-TRAINED (compressive_gqa_grouped) checkpoints only: how the cross-block gate is computed at
# decode. 'grouped' (Version A) = group-mean gate, matches training; 'selection_only' (Version B) =
# per-head gate, pair with LANDMARK_GROUP_SELECTION=mean to share only the top-k selection. Empty ->
# the module's baked-in default ('grouped'). Ignored by non-grouped checkpoints.
LANDMARK_DECODE_GATE_MODE="${LANDMARK_DECODE_GATE_MODE:-}"
DECODE_GATE_ARGS=""
[ -n "$LANDMARK_DECODE_GATE_MODE" ] && DECODE_GATE_ARGS="--landmark-decode-gate-mode $LANDMARK_DECODE_GATE_MODE"
# Optional output tag so parallel eval configs on the SAME checkpoint (e.g. the two decode-gate modes)
# write to DISTINCT dirs/files instead of overwriting each other. Set by the launcher; empty (all
# existing callers) -> byte-identical paths to before.
EVAL_TAG="${EVAL_TAG:-}"
SUF="${EVAL_TAG:+_$EVAL_TAG}"
# Landmark + compressive attention can't do batched/left-padded generation (blocks tied to absolute
# position) -> force batch_size=1 for those variants. Dense keeps the configured (larger) batch.
case "$VARIANT" in landmark|compressive) BATCH_SIZE=1 ;; esac
# landmark/compressive decode-time knobs (unset -> eval_lc_native.py defaults: 10%-of-prompt top-k,
# checkpoint's trained nonselected_landmark_mass). No effect on dense; docchunk uses its own script.
LANDMARK_TOP_K_BLOCKS="${LANDMARK_TOP_K_BLOCKS:-}"
LANDMARK_NONSELECTED_MASS="${LANDMARK_NONSELECTED_MASS:-}"
LANDMARK_FLAGS=""
[ -n "$LANDMARK_TOP_K_BLOCKS" ] && LANDMARK_FLAGS="$LANDMARK_FLAGS --landmark-top-k-blocks $LANDMARK_TOP_K_BLOCKS"
[ -n "$LANDMARK_NONSELECTED_MASS" ] && LANDMARK_FLAGS="$LANDMARK_FLAGS --landmark-nonselected-mass $LANDMARK_NONSELECTED_MASS"

PRASANNS="$WEKA_LLM/checkpoints/prasanns"
BUNDLE="${BUNDLE:-$PRASANNS/_eval_bundle}"
# LADDER_VERSION: v2 is the ONLY supported ladder -- every rung of a task shares the SAME 500
# questions and only the distractors vary; all rungs (base + rerank + oolong + xlong) live under the
# v2 bundle. v1 is DISABLED (2026-07-29): its rungs each drew their own questions, so every
# rung-to-rung delta carried eval-set resampling noise on top of the length effect. Fail loudly
# rather than silently resolving v2 rung names against a v1 tree.
LADDER_VERSION="${LADDER_VERSION:-v2}"
if [ "$LADDER_VERSION" != "v2" ]; then
  echo "ERROR: LADDER_VERSION=$LADDER_VERSION is not supported -- v2 is the only ladder." >&2
  echo "       v1 resampled questions per rung; rebuild as v2 (build_v2_eval_ladders.py for" >&2
  echo "       2k-32k, build_xlong_rungs.py for 64k-2M) and point EVAL500 at a v2 bundle." >&2
  exit 2
fi
# DEFAULT = the CLEAN bundle (2026-07-29). It carries a verified 2k..2M ladder for every task:
# eval_size>=500 at every rung and PubMed-only contradiction distractors. The previous default
# (_eval_bundle_eval500_v2) has contra 64k..2M at 28-31% FEVER/wiki distractors -- a domain shortcut,
# since the gold pair is the only biomedical text -- and eval_size=300 at 64k/128k/256k for every
# doc-pool task.
#
# The <=32k rungs are BYTE-IDENTICAL between the two bundles (size+ETag verified for all 18), so
# switching does NOT move any <=32k number. What does change: contra at 64k and above is a different
# (harder, domain-homogeneous) task -- clean 256k is n6102 vs the old n6408 -- so contra numbers must
# NOT be compared across this switch without re-running the earlier points. Set
# EVAL500=$PRASANNS/_eval_bundle_eval500_v2 to reproduce a pre-switch run.
EVAL500="${EVAL500:-$PRASANNS/_eval_bundle_eval500_v2_clean}"
VFLAG="--ladder-version v2"
# ---- OPT-IN ultra-long rungs (OFF by default). LADDER_XLONG=1 appends the requested XLONG_RUNGS
# (64k..2M) for every task that has xlong rung files -- contra|nq|outlier plus rerank|oolong, whose
# rungs were built 2026-07-27 -- forces bs=1, and raises MAX_LENGTH so prompts aren't truncated.
LADDER_XLONG="${LADDER_XLONG:-0}"
XLONG_RUNGS="${XLONG_RUNGS:-64k,128k}"   # 256k is huge + needs an 80GB GPU -> opt in explicitly
XLFLAG=""
[ "$LADDER_XLONG" = "1" ] && XLFLAG="--xlong"
RESULTS="${RESULTS:-$PRASANNS/_eval_results}"
RUN_DIR="$PRASANNS/$RUN"
# Where the per-task result JSONs land. Default = the run's own eval/ dir under prasanns/<RUN>.
# Override via the launcher's --results-dir (forwarded as EVAL_OUT_DIR) to write anywhere on weka.
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$RUN_DIR/eval}"
# Auto-suffix (from $SUF, built above from EVAL_TAG = decode-gate-mode + group-selection) so a sweep
# over either knob never overwrites another config's results, regardless of whether EVAL_OUT_DIR was
# defaulted or user-supplied via --results-dir. RUN_TAG carries the same suffix into the flat
# $RESULTS/ copies below (keyed by $RUN, not $EVAL_OUT_DIR, so they'd otherwise collide too).
RUN_TAG="${RUN}${SUF}"
[ -n "$SUF" ] && EVAL_OUT_DIR="${EVAL_OUT_DIR%/}${SUF}"

REPO="${REPO:-$PWD}"                          # cloned OLMo-core repo (gantry cwd); eval CODE = in-repo ctc_eval
export PYTHONPATH="$REPO/src/scripts:$REPO/src:${PYTHONPATH:-}"   # so `import ctc_eval...` resolves (olmo_core also pip -e)
export EVAL500_ROOT="$EVAL500"                # eval_lc_native.py reads the _600/_500 rungs from here (weka data)
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM at long ctx on smaller GPUs
mkdir -p "$EVAL_OUT_DIR" "$RESULTS"

echo "=== BEAKER multirung eval | host=$(hostname) RUN=$RUN TASK=$TASK VARIANT=$VARIANT NGPU=$NGPU START=$(date -u '+%F %T')Z ==="
echo "    BUNDLE=$BUNDLE"
echo "    EVAL500=$EVAL500"
nvidia-smi -L 2>/dev/null | head -8 || true

# ---- resolve the checkpoint step dir (CKPT override > STEP pin > latest complete step) ----
if [ -n "${CKPT:-}" ]; then
  :  # explicit absolute step dir (e.g. for a one-off validation against any weka checkpoint)
elif [ -n "$STEP" ]; then
  CKPT="$RUN_DIR/$STEP"
else
  CKPT=""
  for d in $(ls -d "$RUN_DIR"/step*/ 2>/dev/null | sed 's#/$##' | sort -V); do
    [ -f "$d/model_and_optim/.metadata" ] && CKPT="$d"   # keep last (highest step) that is complete
  done
fi
CKPT="${CKPT%/}"
if [ -z "$CKPT" ] || [ ! -f "$CKPT/model_and_optim/.metadata" ]; then
  echo "ERROR: no complete step dir (config.json + model_and_optim/.metadata) under $RUN_DIR (CKPT='$CKPT')"
  echo "       contents:"; ls -la "$RUN_DIR" 2>/dev/null | head -20
  exit 2
fi
echo "    CKPT=$CKPT"

# ---- make sure the rerank/outlier metric deps are importable (lazy scipy/sklearn) ----
python -c "import scipy, sklearn" 2>/dev/null || pip install --quiet scipy scikit-learn || true

cd "$REPO"   # CODE is in-repo (ctc_eval); DATA comes from weka via --root "$BUNDLE" + EVAL500_ROOT
PORT=$(( 20000 + RANDOM % 20000 ))
TR="torchrun --nproc_per_node=$NGPU --master_port=$PORT src/scripts/ctc_eval/eval/eval_lc_native.py --prompt-format $PROMPT_FORMAT"

# ---- docchunk: box-marker chunked prefill + bs=1 KV-cached decode over the FULL ladder (all 9 tasks
# incl. the 4 OOD ladders) via eval_lc_native_docchunk_ladder.py. It shares the TASK->LTASK/RUNGS case
# blocks below with the dense/landmark/compressive path; only the final torchrun differs (branched at
# the invocation on VARIANT=docchunk). So we do NOT early-exit here -- just fall through.

# NOTE: a v1-only rerank branch used to live here (CE-graded k20/k50/k100 files evaluated at the
# base rung). It is gone with v1: under v2 rerank is a normal shared-question ladder, and the
# docchunk path grades rerank inside the docchunk ladder eval.

# ---- dense / landmark / compressive: standard multi-rung ladder (NDCG/F1/score per rung) ----
# v2 is the only ladder (v1 is rejected above), so the rung table is unconditional: all rungs
# -- base, rerank, oolong, xlong -- come from the v2 bundle via --ladder-version v2, and the
# per-task base-data args (--contra-data/--nq-data/--outlier-data/--rerank-data) are unused.
  # v2: all rungs (incl. base + rerank) come from the v2 bundle via --ladder-version v2;
  # the per-task base-data (--contra-data/--nq-data/--outlier-data/--rerank-data) args are unused.
  case "$TASK" in
    contra)  RUNGS="2k,8k,16k,32k"; LTASK=contradiction; EXTRA="--contra-max-new-tokens 512" ;;
    nq)      RUNGS="3k,8k,16k,32k"; LTASK=nq;            EXTRA="" ;;
    outlier) RUNGS="3k,8k,16k,32k"; LTASK=outlier;       EXTRA="" ;;
    rerank)  RUNGS="3k,8k,16k";     LTASK=rerank;        EXTRA="" ;;  # CE-graded, no 32k pool
    oolong)  RUNGS="8k,16k,32k";    LTASK=oolong;        EXTRA="" ;;
    fiqa)    RUNGS="2k,4k,8k,16k";  LTASK=fiqa;          EXTRA="" ;;  # OOD generalization (BEIR)
    scifact) RUNGS="4k,8k,16k,32k"; LTASK=scifact;       EXTRA="" ;;  # OOD generalization (BEIR)
    outlier_review) RUNGS="3k,8k,16k,32k"; LTASK=outlier_review; EXTRA="" ;;  # OOD outlier (Amazon reviews)
    contra_fever)   RUNGS="2k,8k,16k,32k"; LTASK=contra_fever;   EXTRA="--contra-max-new-tokens 512" ;;  # OOD contradiction (FEVER)
    *) echo "ERROR unknown TASK=$TASK"; exit 2 ;;
  esac
if [ "$LADDER_XLONG" = "1" ]; then
  case "$TASK" in
    contra|nq|outlier|rerank|oolong)
      # XLONG_ONLY=1 REPLACES the base rungs instead of appending, for when the base-rung pass has
      # already been run separately -- otherwise every xlong job re-runs 2k-32k first, at the bs=1
      # the xlong path forces, which is both wasted GPU time and a duplicate of results already on
      # weka (the two passes differ by EVAL_TAG, so they do not overwrite each other).
      if [ "${XLONG_ONLY:-0}" = "1" ]; then
        RUNGS="$XLONG_RUNGS"
      else
        RUNGS="$RUNGS,$XLONG_RUNGS"
      fi
      BATCH_SIZE=1
      # Built prompts run ~0.4-4% OVER the rung label (doc count calibrated from a median, plus the
      # instruction/query/marker wrap), so these caps carry a ~10% margin. The old 256k value of
      # 263168 (= label + 1024) truncated the prompt TAIL -- where the question lives -- scoring
      # f1 0.000 at parse_rate 1.0 for a healthy model. eval_lc_native.py re-raises max_length by
      # the same 10% rule, so it corrects an undersized value here rather than trusting it.
      # ⚠ 256k/512k/1M/2M all exceed Qwen3.5's native 262,144 positions and need a YaRN serving copy
      # (debug/ctx_ceiling_4b/make_yarn_copy.py). 256k is in that list because its prompts land
      # 0.4-3.3% OVER the 262,144 label -- the label sits exactly ON the ceiling, so the overage
      # crosses it. eval_lc_native.py now warns on the realized cap rather than the label.
      #
      # They also need PREFILL_CHUNK_SIZE. A one-shot prefill materializes every intermediate at the
      # full prompt length, and one layer's SwiGLU (~59KiB/token) is nearly twice the KV cache it
      # produces (~32KiB/token) -- so past ~256k the transient, not the cache, is what exhausts the
      # GPU. Measured on an 80GB H100: 512k peaked at 77 of 79.19 GiB (7 of 10 tasks OOMed) and 1M
      # died outright. Chunking bounds activations by the chunk instead of the prompt, putting 512k
      # at ~30 GiB and 1M at ~48 GiB, and is mathematically identical (see the parity test
      # test_generation_module_chunked_prefill_matches_one_shot).
      #
      # 256k was the ONE xlong rung left unchunked, which made it the only rung differing from its
      # neighbours in the prefill path as well as in length. In the 2026-08-04 sweep it scored 5-6x
      # BELOW both the 128k and the 512k rungs in both arms (nq dense .830 -> .146 -> .760; landmark
      # .800 -> .000 -> .792) -- non-monotonic in a way no capability story explains. Chunk it too,
      # so the rung is comparable to the ones on either side of it.
      case ",$XLONG_RUNGS," in
        *,2M,*)   MAX_LENGTH=2308915; PREFILL_CHUNK_SIZE=32768 ;;
        *,1M,*)   MAX_LENGTH=1155482; PREFILL_CHUNK_SIZE=32768 ;;
        *,512k,*) MAX_LENGTH=578765;  PREFILL_CHUNK_SIZE=32768 ;;
        *,256k,*) MAX_LENGTH=290406;  PREFILL_CHUNK_SIZE=32768 ;;
        *,128k,*) MAX_LENGTH=146227 ;;
        # 64k-only: 68608 (cap 68512) -> nq (max real prefill 67679) and outlier (67986) run
        # skipped_too_long=0. NOTE the empirical single-80GB-H100 ceiling for the docchunk
        # FlexAttention eval path: seq_len ~66k fits, ~77k OOMs (measured -- contra's long tail
        # OOMed at seq_len=77167 even after the Tier-1 empty_cache retry). contra's 64k/32k files
        # have a heavy >66k tail (query-dominated), so a cap high enough to clear it (e.g. 81920)
        # makes those examples OOM, while a memory-safe cap skips ~half of them -- i.e. contra 64k
        # is NOT cleanly measurable on one 80GB GPU and needs Tier-2 tensor/context-parallel (same
        # class as 128k). This 68608 keeps nq/outlier exact and stays just above the proven-safe
        # ~66k so contra completes (its extreme tail skipped) rather than OOM-crashing the job.
        *)        MAX_LENGTH=68608  ;;
      esac
      # torchrun inherits the environment, and eval_lc_native.py defaults --prefill-chunk-size from
      # PREFILL_CHUNK_SIZE, so this must be exported rather than left a plain shell variable.
      [ -n "${PREFILL_CHUNK_SIZE:-}" ] && export PREFILL_CHUNK_SIZE
      echo "    [xlong] RUNGS=$RUNGS MAX_LENGTH=$MAX_LENGTH BATCH_SIZE=$BATCH_SIZE PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-off}" ;;
    # fiqa/scifact/outlier_review/contra_fever have no xlong rung files, so they keep their base
    # ladder rather than silently re-running it under an xlong tag.
    *) echo "    [xlong] no xlong rungs for TASK=$TASK; base ladder unchanged." ;;
  esac
fi
OUT="$EVAL_OUT_DIR/${TASK}_multirung.json"
echo "=== EVAL $TASK rungs=$RUNGS ladder=$LADDER_VERSION variant=$VARIANT -> $OUT ($(date -u '+%T')Z) ==="
if [ "$VARIANT" = "docchunk" ]; then
  # box-marker chunked prefill + bs=1 KV-cached decode; same ladder keys ($LTASK/$RUNGS) as the
  # dense/landmark path (incl. the 4 OOD ladders). NO-CoT throughout -> contra keeps its short budget
  # (the CoT --contra-max-new-tokens 512 in $EXTRA is intentionally dropped here). EVAL500_ROOT is
  # already exported so the {E5}/<sub> ladder files resolve on weka.
  # COT_MODE (default none) keeps the no-CoT eval byte-identical; COT_MODE=plan builds the OOLONG
  # prefill WITH the plan CoT (to match a CoT-trained checkpoint) and widens the OOLONG gen budget so
  # the plan+answer fits before the newline-after-"answer:" early-stop.
  COT_ARGS="--cot-mode ${COT_MODE:-none}"
  [ "${COT_MODE:-none}" = plan ] && COT_ARGS="$COT_ARGS --oolong-max-new-tokens 512"
  # Emitter MUST match how the model was TRAINED (_docchunk_5task_32k_nocpt_common.py:
  # emit="landmark" if variant in {landmark,compressive} else "dense"). dense/random_doc use the dense
  # box-marker emitter; landmark/compressive use landmark tokens. Feeding the wrong emitter = garbage.
  DC_EMIT=dense; case "$RUN" in *compressive*|*landmark*) DC_EMIT=landmark ;; esac
  echo "[docchunk] emitter variant = $DC_EMIT (run=$RUN)"
  torchrun --nproc_per_node="$NGPU" --master_port="$PORT" \
    src/scripts/ctc_eval/eval/eval_lc_native_docchunk_ladder.py \
    --variant "$DC_EMIT" --model-path "$CKPT" --out "$OUT" --tokenizer "$TOKENIZER" \
    --root "$BUNDLE" --max-test-samples "$MAX_TEST" --max-length "$MAX_LENGTH" --mem-freq 63 \
    --ladder-version "$LADDER_VERSION" --tasks "$LTASK" --rungs "$RUNGS" $COT_ARGS
  rc=$?
else
  $TR --model-path "$CKPT" --out "$OUT" --tokenizer "$TOKENIZER" --max-length "$MAX_LENGTH" \
      --root "$BUNDLE" --max-test-samples "$MAX_TEST" --batch-size "$BATCH_SIZE" --skip-ruler --skip-gen \
      --ladder $VFLAG $XLFLAG --ladder-tasks "$LTASK" --ladder-rungs "$RUNGS" $EXTRA $LANDMARK_FLAGS $GROUP_ARGS $DECODE_GATE_ARGS
  rc=$?
fi
if [ -f "$OUT" ]; then
  cp "$OUT" "$RESULTS/${RUN_TAG}_${TASK}_multirung.json" 2>/dev/null || true
  GEN="${OUT%.json}.generations.jsonl"
  [ -f "$GEN" ] && cp "$GEN" "$RESULTS/${RUN_TAG}_${TASK}_multirung.generations.jsonl" 2>/dev/null || true
  echo "--- $OUT ---"; cat "$OUT"
  [ -f "$GEN" ] && python src/scripts/ctc_eval/eval/print_gen_sample.py "$GEN" "${GEN_SAMPLE_N:-6}" || true
fi
echo "=== DONE TASK=$TASK rc=$rc result=$RESULTS/${RUN_TAG}_${TASK}_multirung.json $(date -u '+%F %T')Z ==="
exit $rc
