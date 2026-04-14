MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-test/step500
gantry run --cluster ai2/saturn -y --budget ai2/oceo --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras && uv pip install torch && uv pip install flash-attn" \
        --weka=oe-adapt-default:/weka/oe-adapt-default \
        --weka=oe-training-default:/weka/oe-training-default \
        --priority urgent \
        --gpus 8 \
        --beaker-image nathanl/open_instruct_auto \
        -- uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
            -i /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-test/step400 \
            -o /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-test/step400-hf3 \
            --skip-validation \
            --max-sequence-length 65536

        # --ref 3d6f43749887d66983257ca225819f6149069ce3 \


#### ABOVE DOESN'T WORK
"/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-7b-anneal-code-50b/step95368"
"/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-math_anneal-50b/step47684"
"/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-7b-full-mix-150b/step286103"
"/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-code_anneal-50b/step47684"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft/step1062"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step782"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-tool-use-sft-unfrozen/step888"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-safety-sft/step534"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-post-train-sft/step1856"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-continued-post-train-sft/step1856"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-code-sft/step782"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-tool-use-sft/step888"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-safety-sft/step534"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-retrain-mid-train-sft/step1856"
"/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base"
MODEL_PATHS=(
)
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
            -i $MODEL_PATH \
            -o $MODEL_PATH-hf \
            --skip-validation \
            --max-sequence-length 65536
done


# convert FROM HF
MODEL_PATHS=(
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft/step1062-hf/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7/grpo_math_only_retrain_flex-base-2x7b-math-sft-6e-7__1__1775598273_checkpoints/step_300"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step782-hf/grpo_code_only_flex-2x7b-code-6e-7/grpo_code_only_flex-2x7b-code-6e-7__1__1775600356_checkpoints/step_150"
)
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
uv run python src/examples/huggingface/convert_checkpoint_from_hf.py \
            -i $MODEL_PATH \
            -o $MODEL_PATH-oc \
            -c /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft/step1062/config.json \
            --skip-validation 
done &&

"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-code-sft/step782-hf/grpo_code_only_flex-7b-code-6e-7/grpo_code_only_flex-7b-code-6e-7__1__1775600361_checkpoints/step_100"
MODEL_PATHS=(
    #### NEED MATH RL'D 7B!!!!!!
)
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
uv run python src/examples/huggingface/convert_checkpoint_from_hf.py \
            -i $MODEL_PATH \
            -o $MODEL_PATH-oc \
            -c /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-code-sft/step782/config.json \
            --skip-validation 
done

    "${CODE_BASE},${CODE_EXPERT_2},${MATH_EXPERT}|${CKPT_DIR}/FlexOlmo-3x7B-code_base-code_only-math_mixed"
    "${MATH_BASE},${CODE_EXPERT_3},${MATH_EXPERT}|${CKPT_DIR}/FlexOlmo-3x7B-math_base-code_mixed-math_mixed"

MODEL_PATHS=(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_sft-olmo3_code-tool_use"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "Processing: $MODEL_PATH"
    
    mkdir -p "${MODEL_PATH}/model_and_optim" && \
    cp -f "${MODEL_PATH}"/*.distcp "${MODEL_PATH}/model_and_optim/" && \
    cp -f "${MODEL_PATH}/.metadata" "${MODEL_PATH}/model_and_optim/" && \
    python src/examples/huggingface/convert_checkpoint_to_hf.py \
        -i "$MODEL_PATH" \
        -o "${MODEL_PATH}-hf" \
        --skip-validation \
        --max-sequence-length 65536 && \
    cp /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-test-router-hf/chat_template.jinja "${MODEL_PATH}"
    
    if [ $? -ne 0 ]; then
        echo "Failed: $MODEL_PATH"
    fi
done




-----------


uv run python src/examples/huggingface/convert_checkpoint_from_hf.py \
            -i allenai/Olmo-3-1025-7B \
            -o /weka/oe-adapt-default/jacobm/olmo-core-checkpoints/olmo-3-7b-base/ \
            -c /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool-mix-unf-lm-head-embed/step888/config.json \
            --skip-validation 
done