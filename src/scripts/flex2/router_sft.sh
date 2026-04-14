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
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_rl-olmo3_code-tool_use-average_all-no_rt
AMOUNT=0.05
LR=1e-4
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo3_math_code_tool_use-$AMOUNT
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_rl-code_rl-tool_use
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-4x7B-merge-all-math_rl-code_rl-tool_use_sft-$AMOUNT-$LR \
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
    --workspace ai2/flex2 \
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


# 5x7B
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-math_rl-code_rl-tool_use-safety_sft
AMOUNT=0.05
LR=1e-4
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo3_math_code_tool_use_safety-$AMOUNT
uv run python src/scripts/train/sft/FlexOlmo-SFT-5x7B.py launch \
    flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-$AMOUNT-$LR \
        $BASE_CKPT \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=2048 \
    --launch.num_gpus=8 \
    --num_nodes=5 \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --model_name olmoe-5x7b \
    --dataset_path $SFT_DATASET

# 5x7B Olmo 2 Code
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-math_rl-olmo2_code_sft-tool_use-safety_sft
AMOUNT=0.05
LR=1e-4
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo3_math_olmo2_code_tool_use_safety-$AMOUNT
uv run python src/scripts/train/sft/FlexOlmo-SFT-5x7B.py launch \
    flexolmo-5x7B-math_rl-olmo2_code_sft-tool_use_sft-safety_sft-$AMOUNT-$LR \
        $BASE_CKPT \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=2048 \
    --launch.num_gpus=8 \
    --num_nodes=5 \
    --budget ai2/oceo \
    --workspace ai2/olmo-instruct \
    --model_name olmoe-5x7b \
    --dataset_path $SFT_DATASET



/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo2_code-olmo3_math_tool_use_safety-0.05
/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo2_math-olmo3_code_tool_use_safety-0.05

# /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_4-math_rl
# /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_4-code_rl
# /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_3-math_code_rl



# Final 5x7Bs w/ olmo 3 math?
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_all
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_4-math_rl
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_4-code_rl
# BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_3-math_code_rl



BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_3-olmo2_code-olmo3_math
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo2_code-olmo3_math_tool_use_safety-0.05

BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_3-olmo3_code_rl-olmo2_math
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo2_math-olmo3_code_tool_use_safety-0.05
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_3-olmo2_code_math
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo2_code_math-olmo3_tool_use_safety-0.05
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-final-sft-only
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains
# BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains-tool-first
# BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains-tool-first-FIXED
# BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-final-math_sft
# ACTIVE=2
# BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-math_rl-code_rl-tool_use-safety_sft

BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-final-retrain
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-retrain-final
AMOUNT=0.05
LR=1e-4
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo3_math_code_tool_use_safety-$AMOUNT
uv run python src/scripts/train/sft/FlexOlmo-SFT-5x7B.py launch \
    FlexOlmo-5x7B-retrain-final-for-real-$AMOUNT-$LR \
        $BASE_CKPT \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=2048 \
    --launch.num_gpus=8 \
    --num_nodes=5 \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --model_name olmoe-5x7b \
    --dataset_path $SFT_DATASET


BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-final-no-safety-tool
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo3_math_code-0.05
uv run python src/scripts/train/sft/FlexOlmo-SFT-3x7B.py launch \
    flexolmo-3x7B-math_rl-code_rl-0.05-$LR \
        $BASE_CKPT \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-4 \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=2048 \
    --global_batch_size=786432 \
    --launch.num_gpus=8 \
    --num_nodes=3 \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --model_name olmoe-3x7b \
    --dataset_path $SFT_DATASET

# try fewer experts
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-math_rl-code_rl-tool_use-safety_sft
AMOUNT=0.05
LR=5e-4
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo3_math_code_tool_use_safety-$AMOUNT
for NUM_EXPERTS in 1 2 3 4; do
uv run python src/scripts/train/sft/FlexOlmo-SFT-5x7B.py launch \
    flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-$AMOUNT-$LR-$NUM_EXPERTS-active \
        $BASE_CKPT \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=2048 \
    --launch.num_gpus=8 \
    --num_nodes=5 \
    --budget ai2/oceo \
    --workspace ai2/olmo-instruct \
    --model_name olmoe-5x7b-$NUM_EXPERTS-active \
    --dataset_path $SFT_DATASET
done
