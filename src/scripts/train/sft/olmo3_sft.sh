# 7T:
# anneal-round2-100B-olmo3_7b_with-reasoning-anneal-7T-2433efff_step47684-hf

python src/scripts/train/sft/OLMo2-7B-sft.py launch \
    olmo3_r2_7t-tulu_3_sft-5e_-5-2_epochs \
        tulu3-olmo2-mix \
        /weka/oe-training-default/ai2-llm/checkpoints/OLMo3-midtraining/anneal-round2-100B-olmo3_7b_with-reasoning-anneal-7T-2433efff/step47684 \
        ai2/jupiter-cirrascale-2 \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=5e-5 \
    --seq_len=16384 \
    --launch.num_gpus=8 \
    --num_nodes=4 \
    --launch.priority=urgent


# 8T:
# anneal-round2-100B-olmo3_7b_with-reasoning-anneal-8T-a52f1603_step47684-hf

python src/scripts/train/sft/OLMo2-7B-sft.py launch \
    olmo3_r2_8t-tulu_3_sft-5e_-5-2_epochs \
        tulu3-olmo2-mix \
        /weka/oe-training-default/ai2-llm/checkpoints/OLMo3-midtraining/anneal-round2-100B-olmo3_7b_with-reasoning-anneal-8T-a52f1603/step47684 \
        ai2/jupiter-cirrascale-2 \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=5e-5 \
    --seq_len=16384 \
    --launch.num_gpus=8 \
    --num_nodes=4 \
    --launch.priority=urgent