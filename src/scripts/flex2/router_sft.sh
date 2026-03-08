# FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-math_base-code_mixed_no_ann-math_mixed/
# FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed-math_base_again
# FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_20b_ann-math_base_again
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_50b_ann-math_base_again
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_20b_ann_with_sft-math_base_again
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_50b_ann_with_sft-math_base_again
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tulu3-no_code-no_math
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/all
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_sft-olmo3_code-tool_use
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/10k-instance-test-dataset
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tool-use-general-math-code-mix-fixed
# FRACTION=0.75
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tool-use-general-math-code-mix-fixed
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_rl-olmo3_code-tool_use
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tool-use-general-math-code-mix-fixed-$AMOUNT
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_rl_x4
AMOUNT=0.05
LR=1e-4
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_rl-olmo3_code-tool_use-average_all-no_rt
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo3_math_code_tool_use-$AMOUNT
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-4x7B-merge-all-olmo3_3x_domain-$AMOUNT-$LR \
        $FLEX_PATH \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=2048 \
    --launch.num_gpus=8 \
    --num_nodes=4 \
    --budget ai2/oceo \
    --workspace ai2/olmo-instruct \
    --model_name olmoe-4x7b \
    --dataset_path $SFT_DATASET

FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_rl-olmo3_code-math_base
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/all
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_rl-olmo3_code-tool_use
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tool-use-general-math-code-mix
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_50b_ann_with_sft-math_base_again
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-math-code-0.50-redux
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.50-old-seeds \
        $FLEX_PATH \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-4 \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=2048 \
    --launch.num_gpus=8 \
    --num_nodes=4 \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --model_name olmoe-4x7b \
    --dataset_path $SFT_DATASET

# OLD, for comparison
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_50b_ann_with_sft-math_base_again
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/all
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-4x7b-olmo3_code_50b_sft-router_sft_1.0 \
        $BASE_CKPT \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-4 \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=2048 \
    --launch.num_gpus=8 \
    --num_nodes=4 \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --model_name olmoe-4x7b \
    --dataset_path $SFT_DATASET