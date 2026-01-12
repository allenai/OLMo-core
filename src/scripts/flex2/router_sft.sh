# FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-math_base-code_mixed_no_ann-math_mixed/
# FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed-math_base_again
# FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_20b_ann-math_base_again
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_50b_ann-math_base_again
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_20b_ann_with_sft-math_base_again
FLEX_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_50b_ann_with_sft-math_base_again
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/all
sudo uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-4x7b-olmo3_code_50b_sft-router_sft_all_mixed \
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