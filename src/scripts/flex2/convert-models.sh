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


# THIS WORKS:
### Go to a beaker session, my flexolmo directory (Olmo-core), activate olmo-core conda env
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-on-base-no-anneal/step594
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed-on-base-no-anneal/step1062
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-router-training-only/step594
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-anneal-frozen-router-mixed-sft/step1062
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-anneal-NO-frozen-router-mixed-sft/step1062
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-on-base-no-anneal/step150
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step150
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-base-no-anneal/step620
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed/step620
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-test-router
# MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-5B/step9537
# MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-anneal-no-expert-bias-5B/step9537
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-code-anneal-no-eb-5B/step620
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-5B/step620
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_5b_ann-math_base_again
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-math_base-math_mixed-code_mixed_olmo3_5b_ann
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_5b-router_sft_all_mixed/step1128
# MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-20B/step38147
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-20B/step620
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_20b-router_sft_all_mixed/step1128
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620
MODEL_PATHS=(
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_20b_sft-router_sft_all_mixed/step1128"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_all_mixed/step1128"
)
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
            -i $MODEL_PATH \
            -o $MODEL_PATH-hf \
            --skip-validation \
            --max-sequence-length 65536
done

    "${CODE_BASE},${CODE_EXPERT_2},${MATH_EXPERT}|${CKPT_DIR}/FlexOlmo-3x7B-code_base-code_only-math_mixed"
    "${MATH_BASE},${CODE_EXPERT_3},${MATH_EXPERT}|${CKPT_DIR}/FlexOlmo-3x7B-math_base-code_mixed-math_mixed"

MODEL_PATHS=(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-code_base-code_only-math_mixed"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-math_base-code_mixed-math_mixed"
    # Add more model paths here...
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