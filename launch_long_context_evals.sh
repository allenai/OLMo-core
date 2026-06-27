#!/bin/bash
#
# Launch long-context evals for an olmo-core checkpoint on weka:
#   * HELMET  @ 8k-128k  -> via ../ai2-helmet/gantry_eval.sh
#   * RULER   @ 4k-128k  -> via olmo-cookbook-eval (oe-eval), like ../olmo-cookbook/run_ruler.sh
#
# HELMET's configs/single_task/* exist at 8k/16k/32k/64k/128k, so a single
# gantry_eval.sh run with MAX_LENGTH=131072 IS exactly the 8k-128k suite. (To
# stop at 64k for an un-extended model, set HELMET_MAX_LENGTH=65536.)
#
# RULER 4k is NOT available in ai2-helmet (configs-dist only has 8k-64k), which
# is why RULER goes through oe-eval instead: ruler:4k..ruler:64k are registered
# task groups there (each = the full 13-subtask suite at that length).
#
# Defaults assume a landmark-attention Qwen3-4B olmo-core checkpoint:
#   olmo_core backend, generation batch size 1 (landmark attention requires bs=1),
#   urgent priority. The tokenizer defaults by model family inferred from the path
#   (Qwen/Qwen3.5-0.8B for qwen3.5 checkpoints, else Qwen/Qwen3-4B); override with
#   OLMO_CORE_TOKENIZER.
#
# Usage:
#   ./launch_long_context_evals.sh /weka/oe-training-default/ai2-llm/checkpoints/<user>/<run>/stepXXXX
#
# All jobs run in workspace ai2/flex2; RULER results report to the memory-LC
# dashboard (HELMET writes outputs to weka, it has no dashboard).
#
# Common overrides (env vars):
#   PRIORITY, NUM_GPUS, OLMO_CORE_TOKENIZER, OLMO_CORE_BATCH_SIZE,
#   CLUSTER, WORKSPACE, RULER_DASHBOARD, SKIP_HELMET=1, SKIP_RULER=1,
#   OLMO_CORE_GATE_LOG (base path -> per-decoded-token landmark-gate log on RULER; needs
#                       OLMO_CORE_LANDMARK_TOP_K), OLMO_GATE_DATASET,
#   HELMET_MAX_LENGTH (cap HELMET suite, e.g. 32768 for an un-extended model),
#   HELMET_TIMEOUT (gantry --timeout, seconds; default 0 = submit & detach so
#                   RULER launches immediately, -1 = follow the run to completion)

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <weka_checkpoint_path>}"

# ---- shared knobs ----
PRIORITY="${PRIORITY:-urgent}"                       # all jobs run at urgent priority
CLUSTER="${CLUSTER:-ai2/jupiter}"                    # all jobs run on jupiter
BUDGET="ai2/oe-other"                                # ai2/oe-base is deprecated
NUM_GPUS_RULER="${NUM_GPUS_RULER:-2}"
NUM_GPUS_HELMET="${NUM_GPUS_HELMET:-8}"
# Tokenizer default depends on the model family, inferred from the checkpoint path, because the
# eval MUST tokenize with the same vocabulary the model was trained on. Qwen3.5 checkpoints were
# trained with the qwen3_5 tokenizer (vocab 248320, EOS <|endoftext|>=248044, = Qwen/Qwen3.5-0.8B),
# a DIFFERENT vocabulary from Qwen3 (vocab 151936, = Qwen/Qwen3-4B). Evaluating a Qwen3.5 checkpoint
# under the Qwen3 tokenizer feeds the model token IDs from the wrong vocab -> garbage outputs and
# collapsed RULER/HELMET scores. Pick the matching tokenizer by family unless OLMO_CORE_TOKENIZER is
# set explicitly (an explicit value always wins).
if [ -z "${OLMO_CORE_TOKENIZER:-}" ]; then
  case "$(echo "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')" in
    *qwen3.5*|*qwen35*|*qwen3_5*) OLMO_CORE_TOKENIZER="Qwen/Qwen3.5-0.8B" ;;
    *)                            OLMO_CORE_TOKENIZER="Qwen/Qwen3-4B" ;;
  esac
fi
OLMO_CORE_BATCH_SIZE="${OLMO_CORE_BATCH_SIZE:-1}"    # landmark attention -> bs=1
# RULER decode cache toggle. true (default) is correct and fastest for both attention/landmark
# (KV cache) and GatedDeltaNet (dense Qwen3.5) models: as of olmo_core commit c3a80d467, GDN has a
# recurrent state cache (conv windows + delta-rule state) so cached decode matches a full forward.
# This requires the eval job to install olmo_core >= c3a80d467 (the default OLMO_CORE_COMMIT=HEAD
# satisfies this; an older pin does not -- there use_cache=true drops prompt context for GDN, so use
# OLMO_CORE_USE_CACHE=false for correct-but-slower decode). Only affects the RULER path.
OLMO_CORE_USE_CACHE="${OLMO_CORE_USE_CACHE:-true}"
# Optional hard top-k landmark block retrieval at decode (the landmark paper's inference). When set
# to an integer, RULER gets `landmark_top_k_blocks=<k>` and HELMET gets `--olmo_core_landmark_top_k`.
# A FIXED top_k (length-independent) is why a single HELMET run over the whole 8k-128k suite is fine
# here (unlike launch_topk_landmark_evals.sh, whose per-length top_k forces one HELMET job per
# length). RULER also needs OE_EVAL_BRANCH=amandab/ruler-memproj for the model-arg to be recognized.
# Empty (default) = dense soft-gated landmark decode. Ignored by non-landmark models.
OLMO_CORE_LANDMARK_TOP_K="${OLMO_CORE_LANDMARK_TOP_K:-}"
# Compressive-landmark checkpoints only: override the fraction of attention mass (in [0,1)) reserved
# at top-k decode for the landmark (compression) tokens of the NON-selected blocks. When set, RULER
# gets `landmark_nonselected_mass=<f>` and HELMET gets `--olmo_core_landmark_nonselected_mass`. Only
# meaningful with OLMO_CORE_LANDMARK_TOP_K set; ignored otherwise. Empty (default) = use the value
# baked into the checkpoint (0.1). See launch_nonselected_mass_evals.sh to sweep this.
OLMO_CORE_LANDMARK_NONSELECTED_MASS="${OLMO_CORE_LANDMARK_NONSELECTED_MASS:-}"
# Optional landmark-gate analysis logging (see olmo_core.nn.attention.landmark_gate_analysis). Set
# to a base output path the RULER job can write (e.g. a /weka/... dir, readable after the run): each
# length-job then logs, per decoded token / layer / head, the landmark BLOCKS that hard top-k
# retrieval opened, to "<base>.ruler<k>k" (the recorder further appends ".rank<N>" per GPU worker).
# Threaded into the RULER container as the env var OLMO_LANDMARK_GATE_LOG (plus OLMO_GATE_DATASET and
# a per-job OLMO_GATE_CONTEXT_LEN=<length>) via gantry env args. ONLY records when
# OLMO_CORE_LANDMARK_TOP_K is also set (gate logging captures hard top-k retrieval; dense soft-gating
# has no gate selection to log). The per-example "subtask" field stays empty because oe-eval runs all
# 13 RULER subtasks in one process, so a static env var can't track it; "dataset" defaults to "ruler"
# (override via OLMO_GATE_DATASET) and "context_len" to the job's nominal length. Empty (default) =
# no gate logging. Only affects the RULER path (HELMET is unaffected).
OLMO_CORE_GATE_LOG="${OLMO_CORE_GATE_LOG:-}"
OLMO_GATE_DATASET="${OLMO_GATE_DATASET:-ruler}"
if [ -n "${OLMO_CORE_GATE_LOG}" ] && [ -z "${OLMO_CORE_LANDMARK_TOP_K}" ]; then
  echo "WARNING: OLMO_CORE_GATE_LOG is set but OLMO_CORE_LANDMARK_TOP_K is not -- landmark gate" >&2
  echo "         logging only records hard top-k retrieval, so no gates will be logged." >&2
fi

# These checkpoints were written by a branch of OLMo-core whose configs use
# fields not present in the released ai2-olmo-core on PyPI (e.g. RoPEConfig's
# no_global_rope). Both eval images install olmo_core from PyPI, so we reinstall
# it from the checkpoint's training commit or the model config won't deserialize.
# Defaults to the current HEAD of this OLMo-core checkout (override if needed).
OLMO_CORE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
OLMO_CORE_COMMIT="${OLMO_CORE_COMMIT:-$(git -C "${OLMO_CORE_REPO}" rev-parse HEAD)}"

# The RULER (oe-eval, gantry) path builds a fresh uv venv that lacks flash-attn,
# which the olmo_core model requires. Install the prebuilt wheel matching the
# gantry env (torch 2.8 + cu12 + cp311, cxx11abiTRUE — the only variant shipped).
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}"

# olmo_core is installed with --no-deps below (to avoid disturbing oe-eval's pinned env), so its
# own dependency floors aren't enforced. oe-eval's lockfile ships dataclass-extensions 0.2.4, whose
# older union coercion can't deserialize hybrid (named-blocks) configs like Qwen3.5 GDN — it fails
# with: TransformerBlockConfig has no field 'gdn'. olmo_core requires >=0.3.0, so reinstall it
# explicitly. (Pure-attention checkpoints have a single block and are unaffected.)
# NB: pin with '==' (no shell-special chars) and an exact version, NOT '>=': the install string is
# threaded through olmo-cookbook-eval's --gantry-args JSON, which strips inner quotes, so an
# unquoted '>=' would be parsed as a shell redirect and an unconstrained reinstall no-ops (uv
# "Audited", keeps 0.2.4). An exact '==0.5.0' has no special chars and forces the upgrade.

# Resolve ../ai2-helmet relative to this script's location.
HELMET_DIR="${HELMET_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../ai2-helmet" && pwd)}"

# Optional chat-template HELMET run. Set HELMET_USE_CHAT_TEMPLATE=True to evaluate with the chat
# template; HELMET's --thinking flag is left off, so hybrid models (e.g. Qwen3) run in non-thinking
# mode (enable_thinking=False). A marker (default "nothink") is appended to the output dir and job
# description so it never collides with the default completion-mode run. Override with
# HELMET_NAME_SUFFIX.
HELMET_USE_CHAT_TEMPLATE="${HELMET_USE_CHAT_TEMPLATE:-False}"
HELMET_NAME_SUFFIX="${HELMET_NAME_SUFFIX:-}"
if [ -z "${HELMET_NAME_SUFFIX}" ] && [ "${HELMET_USE_CHAT_TEMPLATE}" = "True" ]; then
  HELMET_NAME_SUFFIX="nothink"
fi

# ============================================================================
# 1) HELMET @ 8k-64k  (ai2-helmet gantry_eval.sh, olmo_core backend)
# ============================================================================
if [ "${SKIP_HELMET:-0}" != "1" ]; then
  echo "==> Launching HELMET 8k-128k for ${MODEL_PATH}"
  ( cd "${HELMET_DIR}" && \
    MODEL_NAME_OR_PATH="${MODEL_PATH}" \
    MAX_LENGTH="${HELMET_MAX_LENGTH:-131072}" \
    CLUSTER="${CLUSTER}" \
    WORKSPACE="${WORKSPACE:-ai2/flex2}" \
    BUDGET="${BUDGET}" \
    NUM_GPUS="${NUM_GPUS_HELMET}" \
    BACKEND=olmo_core \
    PRIORITY="${PRIORITY}" \
    OLMO_CORE_TOKENIZER="${OLMO_CORE_TOKENIZER}" \
    OLMO_CORE_BATCH_SIZE="${OLMO_CORE_BATCH_SIZE}" \
    OLMO_CORE_COMMIT="${OLMO_CORE_COMMIT}" \
    OLMO_CORE_LANDMARK_TOP_K="${OLMO_CORE_LANDMARK_TOP_K}" \
    OLMO_CORE_LANDMARK_NONSELECTED_MASS="${OLMO_CORE_LANDMARK_NONSELECTED_MASS}" \
    USE_CHAT_TEMPLATE="${HELMET_USE_CHAT_TEMPLATE}" \
    EVAL_NAME_SUFFIX="${HELMET_NAME_SUFFIX}" \
    TIMEOUT="${HELMET_TIMEOUT:-0}" \
    bash ./gantry_eval.sh )
fi

# ============================================================================
# 2) RULER @ 4k-64k  (olmo-cookbook-eval -> oe-eval, olmo_core backend)
#    One job per length; each ruler:Nk group runs the full 13-subtask suite.
#    olmo_core backend reads the tokenizer from the checkpoint config (matches
#    ../olmo-cookbook/run_ruler.sh, which passes no tokenizer for olmo_core).
# ============================================================================
# Lengths in K; max_length = K * 1024. Kept as a plain list (no associative
# arrays) so this runs under macOS's stock bash 3.2.
if [ "${SKIP_RULER:-0}" != "1" ]; then
  # Override with e.g. RULER_LENGTHS_K="4" to smoke-test a single length.
  RULER_LENGTHS_K=( ${RULER_LENGTHS_K:-4 8 16 32 64 128} )

  # Optional non-thinking (ChatML) RULER variant. Set RULER_CHAT_TEMPLATE (e.g. qwen3_nothink) to
  # run RULER through a chat template, and OE_EVAL_BRANCH to the oe-eval branch that carries that
  # template (e.g. amandab/ruler-memproj). ALWAYS pair the non-thinking run with a distinct
  # RULER_DASHBOARD: the cookbook keys the remote output path on the dashboard
  # (s3://.../evaluation/<dashboard>/<model>/<task-hash>), so a separate dashboard is what keeps the
  # thinking and non-thinking partial-results files from ever colliding.
  # Default to amandab/ruler-memproj: the model-args this script always sends
  # (use_cache, and optionally landmark_top_k_blocks / landmark_nonselected_mass)
  # only exist in that branch's MODEL_DEFAULTS. oe-eval's hash_dict rejects any
  # model-arg key without a default ("Hashing error! Key use_cache not in
  # defaults!"), so launching against oe-eval main fails before the job runs.
  OE_EVAL_BRANCH="${OE_EVAL_BRANCH:-amandab/ruler-memproj}"
  OE_EVAL_BRANCH_FLAG="--oe-eval-branch ${OE_EVAL_BRANCH}"

  # Put a marker (default "nothink") in the run name for chat-template variants. --name-suffix
  # appends it to the model name, so it shows up in the beaker display name, the dashboard row, and
  # the S3 output dir -- keeping the variant clearly distinct from the default RULER run. Override
  # the marker with RULER_NAME_SUFFIX.
  NAME_SUFFIX_FLAG=""
  ruler_name_suffix="${RULER_NAME_SUFFIX:-}"
  if [ -z "${ruler_name_suffix}" ] && [ -n "${RULER_CHAT_TEMPLATE:-}" ]; then
    ruler_name_suffix="nothink"
  fi
  if [ -n "${ruler_name_suffix}" ]; then
    NAME_SUFFIX_FLAG="--name-suffix ${ruler_name_suffix}"
  fi

  # Submission watchdog: a `gantry run` occasionally hangs indefinitely (vs the normal ~30s),
  # which under `set -e` would stall the whole sweep. macOS ships no `timeout`, so poll a
  # backgrounded submission and kill its entire process tree if it exceeds RULER_SUBMIT_TIMEOUT,
  # retrying up to RULER_SUBMIT_RETRIES times so one stuck length never blocks the rest.
  RULER_SUBMIT_TIMEOUT="${RULER_SUBMIT_TIMEOUT:-300}"
  RULER_SUBMIT_RETRIES="${RULER_SUBMIT_RETRIES:-3}"
  _kill_tree() {
    # SIGKILL (not TERM): a hung `gantry run` can ignore SIGTERM, which would leave the parent
    # `wait` blocking forever. KILL the whole tree bottom-up so the submission reliably dies and
    # the retry can proceed.
    local p="$1" c
    for c in $(pgrep -P "$p" 2>/dev/null); do _kill_tree "$c"; done
    kill -KILL "$p" 2>/dev/null || true
  }
  _submit_with_watchdog() {
    "$@" &
    local pid=$! waited=0
    while kill -0 "$pid" 2>/dev/null; do
      if [ "${waited}" -ge "${RULER_SUBMIT_TIMEOUT}" ]; then
        echo "    !! submission exceeded ${RULER_SUBMIT_TIMEOUT}s -- killing process tree"
        _kill_tree "$pid"; wait "$pid" 2>/dev/null; return 124
      fi
      sleep 5; waited=$(( waited + 5 ))
    done
    wait "$pid"
  }

  for k in "${RULER_LENGTHS_K[@]}"; do
    task="ruler:${k}k"
    max_length=$(( k * 1024 ))
    model_args="trust_remote_code=true,max_length=${max_length},tokenizer=${OLMO_CORE_TOKENIZER},use_cache=${OLMO_CORE_USE_CACHE}"
    if [ -n "${OLMO_CORE_LANDMARK_TOP_K}" ]; then
      model_args="${model_args},landmark_top_k_blocks=${OLMO_CORE_LANDMARK_TOP_K}"
    fi
    if [ -n "${OLMO_CORE_LANDMARK_NONSELECTED_MASS}" ]; then
      model_args="${model_args},landmark_nonselected_mass=${OLMO_CORE_LANDMARK_NONSELECTED_MASS}"
    fi
    if [ -n "${RULER_CHAT_TEMPLATE:-}" ]; then
      model_args="${model_args},chat_model=true,chat_template=${RULER_CHAT_TEMPLATE}"
    fi
    # Landmark-gate logging: pass the per-length output path + metadata into the container as env
    # vars. oe-eval builds the launch as `gantry run`, where each gantry-arg key becomes a `--key`
    # flag; gantry's env flag is `--env NAME=VALUE`, repeated. oe-eval allows repeating a flag via
    # the `key##N` convention (it strips `##\d+` before running), so env##1/2/3 all render as `--env`
    # -- and the bare `env` key (the cookbook's default VLLM_USE_V1) is left untouched.
    gate_log_gantry_args=""
    if [ -n "${OLMO_CORE_GATE_LOG}" ]; then
      gate_log_gantry_args=",env##1=OLMO_LANDMARK_GATE_LOG=${OLMO_CORE_GATE_LOG}.${task//:/}"
      gate_log_gantry_args="${gate_log_gantry_args},env##2=OLMO_GATE_DATASET=${OLMO_GATE_DATASET}"
      gate_log_gantry_args="${gate_log_gantry_args},env##3=OLMO_GATE_CONTEXT_LEN=${max_length}"
    fi
    echo "==> Launching RULER ${task} (max_length=${max_length}) for ${MODEL_PATH}"
    attempt=1
    until _submit_with_watchdog olmo-cookbook-eval evaluate \
      "${MODEL_PATH}" \
      --model-backend olmo_core \
      -y "${PRIORITY}" \
      -c "${CLUSTER}" \
      -b "${BUDGET}" \
      -d "${RULER_DASHBOARD:-memory-LC}" \
      -w "${WORKSPACE:-ai2/flex2}" \
      -t "${task}" \
      -n "${NUM_GPUS_RULER}" \
      -z "${OLMO_CORE_BATCH_SIZE}" \
      -g \
      ${OE_EVAL_BRANCH_FLAG} \
      ${NAME_SUFFIX_FLAG} \
      -l "install=uv sync --python 3.11 && uv pip install --no-deps ${FLASH_ATTN_WHEEL} && uv pip install --no-deps git+https://github.com/allenai/OLMo-core.git@${OLMO_CORE_COMMIT} && uv pip install dataclass-extensions==0.5.0 && uv pip install flash-linear-attention==0.4.1${gate_log_gantry_args}" \
      --model-args "${model_args}"; do
      if [ "${attempt}" -ge "${RULER_SUBMIT_RETRIES}" ]; then
        echo "    !! RULER ${task} (nsm=${OLMO_CORE_LANDMARK_NONSELECTED_MASS:-default}) failed after ${attempt} attempts -- skipping"
        break
      fi
      attempt=$(( attempt + 1 ))
      echo "    retrying RULER ${task} (attempt ${attempt}/${RULER_SUBMIT_RETRIES})..."
      sleep 10
    done
  done
fi

echo "All long-context eval jobs submitted."
