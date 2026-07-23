#!/usr/bin/env bash
set -euo pipefail

# Run a converted HF OLMoE checkpoint with the current auto oe-eval image.
# This uses the old/direct oe_eval.run_eval path rather than the newer olmo-eval
# Beaker launcher. The auto image's oe_eval defaults do not currently include
# enforce_eager, so the job patches that container-local defaults file before
# calling oe_eval. No repo or checkpoint files are modified by the Beaker job.

WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
CLUSTER="${CLUSTER:-ai2/titan}"
PRIORITY="${PRIORITY:-urgent}"
IMAGE="${IMAGE:-01KPEDSKAH465STZBH0QPBC6KV}" # oe-eval-beaker/oe_eval_auto
GPU_COUNT="${GPU_COUNT:-1}"
TASK="${TASK:-arc_easy:mc}"
LIMIT="${LIMIT:-64}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MODEL_TYPE="${MODEL_TYPE:-vllm}"
TIMEOUT="${TIMEOUT:-2h}"
MODEL_NAME="${MODEL_NAME:-olmoe3-275m-cx1-baseline-step15365}"
MODEL_PATH="${MODEL_PATH:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/hf-checkpoints/olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-olmoe3-oe-eval-auto-vllm-smoke}"
HF_TOKEN_SECRET="${HF_TOKEN_SECRET:-jacobm_HF_TOKEN}"
HF_HOME_PATH="${HF_HOME_PATH:-/weka-mount/oe-eval-default/oyvindt/hf-cache/}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"
TMP_SPEC="$(mktemp /tmp/olmoe3-oe-eval-auto-vllm-smoke.XXXXXX.yaml)"

case "${MODEL_TYPE}" in
  vllm)
    MODEL_ARGS="{\"model_path\": \"${MODEL_PATH}\", \"add_bos_token\": false, \"gpu_memory_utilization\": 0.85, \"trust_remote_code\": true, \"model_type\": \"vllm\", \"enforce_eager\": true${EXTRA_MODEL_ARGS}}"
    ;;
  hf)
    MODEL_ARGS="{\"model_path\": \"${MODEL_PATH}\", \"add_bos_token\": false, \"trust_remote_code\": true, \"model_type\": \"hf\", \"dtype\": \"bfloat16\"${EXTRA_MODEL_ARGS}}"
    ;;
  *)
    echo "Unsupported MODEL_TYPE='${MODEL_TYPE}'. Use 'vllm' or 'hf'." >&2
    exit 2
    ;;
esac

cat > "${TMP_SPEC}" <<YAML
version: v2
tasks:
  - name: main
    image:
      beaker: ${IMAGE}
    command: [/bin/sh, -c]
    arguments:
      - |
        set -eux
        python --version
        rm -rf ${HF_HOME_PATH%/}/modules/transformers_modules/$(basename "${MODEL_PATH}") || true
        python - <<'PYPATCH'
        from pathlib import Path
        p = Path('/stage/oe_eval/default_configs.py')
        text = p.read_text()
        if '"enforce_eager": False' not in text:
            marker = '    "api_base_url": None,  # Used for litellm models\\n'
            if marker not in text:
                raise RuntimeError('Could not find MODEL_DEFAULTS api_base_url marker')
            text = text.replace(marker, marker + '    "enforce_eager": False,\\n')
            p.write_text(text)
        PYPATCH
        python -m oe_eval.run_eval \\
          --task ${TASK} \\
          --limit ${LIMIT} \\
          --batch-size ${BATCH_SIZE} \\
          --save-raw-requests true \\
          --num-workers ${NUM_WORKERS} \\
          --gpus ${GPU_COUNT} \\
          --model ${MODEL_NAME} \\
          --model-args '${MODEL_ARGS}' \\
          --no-datalake \\
          --output-dir /results
        find /results -maxdepth 3 -type f -print
    envVars:
      - name: VLLM_WORKER_MULTIPROC_METHOD
        value: spawn
      - name: HF_HOME
        value: ${HF_HOME_PATH}
      - name: HF_TOKEN
        secret: ${HF_TOKEN_SECRET}
    datasets:
      - mountPath: /weka/oe-training-default
        source:
          weka: oe-training-default
      - mountPath: /weka-mount/oe-eval-default
        source:
          weka: oe-eval-default
    result:
      path: /results
    resources:
      gpuCount: ${GPU_COUNT}
    context:
      priority: ${PRIORITY}
      minRuntime: 0s
      autoResume: true
    constraints:
      cluster:
        - ${CLUSTER}
    timeout: ${TIMEOUT}
YAML

beaker experiment create "${TMP_SPEC}" --workspace "${WORKSPACE}" --name "${EXPERIMENT_NAME}"
