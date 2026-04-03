  # uv run gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
  #         --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras" \
  #         --weka=oe-adapt-default:/weka/oe-adapt-default \
  #         --weka=oe-training-default:/weka/oe-training-default \
  #         --priority urgent \
  #         --gpus 1 \
  #         -- /root/.local/bin/uv run python src/examples/huggingface/convert_checkpoint_from_hf.py \
  #             -i Qwen/Qwen3-1.7B-Base \
  #             -o /weka/oe-adapt-default/jacobm/repos/cse-579/checkpoints/Qwen3-1.7B-Base-oc \
  #             -c /weka/oe-adapt-default/jacobm/repos/cse-579/OLMo-core/src/scripts/train/sft/qwen3-1.7b-config.json \
  #             --skip-validation


uv run gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/flex2 \
          --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras" \
          --weka=oe-adapt-default:/weka/oe-adapt-default \
          --weka=oe-training-default:/weka/oe-training-default \
          --priority urgent \
          -- /root/.local/bin/uv run python src/examples/huggingface/convert_checkpoint_from_hf.py \
      -i Qwen/Qwen3-1.7B-Base \
      -o /weka/oe-adapt-default/jacobm/repos/cse-579/checkpoints/Qwen3-1.7B-Base-oc-validate \
      -c src/scripts/train/sft/qwen3-1.7b-config.json \
      --device cpu \
      --debug