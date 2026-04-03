MODEL_NAME=""
uv run gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras" \
        --weka=oe-adapt-default:/weka/oe-adapt-default \
        --weka=oe-training-default:/weka/oe-training-default \
        --priority urgent \
        --gpus 1 \
        -- /root/.local/bin/uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
            -i "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo-sft/qwen3-1.7b-sft-10k/step1476" \
            -o /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo-sft/qwen3-1.7b-sft-100k/step1476-hf \
            --max-sequence-length 65536