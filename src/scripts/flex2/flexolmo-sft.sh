LR=8e-5
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/math/
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-anneal-no-expert-bias/step95368
python src/scripts/train/sft/OLMo-sft.py launch \
    flexolmo-2x7b-math-sft \
        $BASE_CKPT \
        ai2/titan \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=$LR \
    --seq_len=32768 \
    --launch.num_gpus=8 \
    --num_nodes=2 \
    --global_batch_size=1048576 \
    --model_name flexolmo-2x7b \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --dataset_path $SFT_DATASET \
    --launch.priority=urgent


-----

# need to figure out why strict=False is necessary
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-anneal-no-expert-bias/step95368
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-math-anneal-NO-frozen-router-5b/step9537
# SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/mixed/
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-anneal-no-expert-bias/step95368
# SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/code
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-base
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-anneal-no-expert-bias-5B/step9537
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-5B/step9537
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-20B/step38147
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-50B/step95368
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-frozen-router-5B/step9537
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-50B/step95368
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/code-general-mix
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-olmo3_50b_code_anneal_mixed_SFT_TEST \
        $BASE_CKPT \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-4 \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=4096 \
    --launch.num_gpus=8 \
    --num_nodes=8 \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --model_name olmoe-2x7b \
    --dataset_path $SFT_DATASET


--------


# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-50B/step95368
# SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/code-general-mix
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-olmo3_math_anneal-5b/step9537
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-math-anneal-frozen-router-5b/step9537
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/mixed/
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-5b-olmo2_math-mixed-sft-TEST \
        $BASE_CKPT \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-4 \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=4096 \
    --launch.num_gpus=8 \
    --num_nodes=4 \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --model_name olmoe-2x7b \
    --dataset_path $SFT_DATASET

-----

# TOOL USE
# SFT_DATASET=/weka/oe-adapt-default/jacobm/olmo3-final-datasets/olmo3-32b-instruct-sft-1114
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-50B/step95368
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tool-use-general-mix
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-no_anneal-tool_use_general_mix \
        $BASE_CKPT \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-4 \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=4096 \
    --launch.num_gpus=8 \
    --num_nodes=4 \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --model_name olmoe-2x7b \
    --dataset_path $SFT_DATASET


SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tool-use-general-mix
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex-olmo/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr_step_350/model_and_optim
uv run python src/scripts/train/sft/OLMo-sft.py launch \
    olmo2-7b-BASE-tool_use_general_mix \
        $BASE_CKPT \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-4 \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=4096 \
    --launch.num_gpus=8 \
    --num_nodes=8 \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --model_name olmo2-7b \
    --dataset_path $SFT_DATASET

# TRY NOT FREEZING ANYTHING (DEBUG)
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tool-use-general-mix
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-no_anneal-UNFROZEN-tool_use_general_mix \
        $BASE_CKPT \
        ai2/saturn \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-4 \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=4096 \
    --launch.num_gpus=8 \
    --num_nodes=1 \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --model_name olmoe-2x7b-unfrozen \
    --dataset_path $SFT_DATASET


------

BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-anneal-no-expert-bias/step95368
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tulu3-no_code-no_math/
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-math_anneal-general_sft \
        $BASE_CKPT \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-4 \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=4096 \
    --launch.num_gpus=8 \
    --num_nodes=4 \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --model_name olmoe-2x7b-only-router \
    --dataset_path $SFT_DATASET