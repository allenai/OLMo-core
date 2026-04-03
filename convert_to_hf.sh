MODEL_NAME="/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo-sft/qwen3-4b-sft-100k/step1476"
HF_MODEL_ID="Qwen/Qwen3-4B-Base"
uv run gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras" \
        --weka=oe-adapt-default:/weka/oe-adapt-default \
        --weka=oe-training-default:/weka/oe-training-default \
        --priority urgent \
        --gpus 1 \
        -- /root/.local/bin/uv run python src/scripts/train/sft/convert_qwen3_to_hf.py \
            -i $MODEL_NAME \
            -o $MODEL_NAME-hf \
            --hf-model-id $HF_MODEL_ID \
            --max-sequence-length 32768