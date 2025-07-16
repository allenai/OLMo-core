# 7T:
# anneal-round2-100B-olmo3_7b_with-reasoning-anneal-7T-2433efff_step47684-hf

python src/scripts/train/sft/OLMo2-7B-sft.py launch \
    olmo3_r2_7t-tulu_3_sft-1e_-6-2_epochs \
        tulu3-olmo2-mix \
        /weka/oe-training-default/ai2-llm/checkpoints/OLMo3-midtraining/anneal-round2-100B-olmo3_7b_with-reasoning-anneal-7T-2433efff/step47684 \
        ai2/jupiter-cirrascale-2 \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-6 \
    --seq_len=16384 \
    --launch.num_gpus=8 \
    --num_nodes=1 \
    --model_name olmo3-7b \
    --launch.priority=urgent


# 8T:
# anneal-round2-100B-olmo3_7b_with-reasoning-anneal-8T-a52f1603_step47684-hf

python src/scripts/train/sft/OLMo2-7B-sft.py launch \
    olmo3_r2_8t-tulu_3_sft-1e_-6-2_epochs \
        tulu3-olmo2-mix \
        /weka/oe-training-default/ai2-llm/checkpoints/OLMo3-midtraining/anneal-round2-100B-olmo3_7b_with-reasoning-anneal-8T-a52f1603/step47684 \
        ai2/jupiter-cirrascale-2 \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-6 \
    --seq_len=16384 \
    --launch.num_gpus=8 \
    --num_nodes=1 \
    --model_name olmo3-7b \
    --launch.priority=urgent

MODEL_NAME=olmo3_r2_7t-tulu_3_sft-5e_-5-2_epochs
### convert:

olmo-cookbook-eval convert \
    "/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/$MODEL_NAME/step1484" \
    -t olmo-core-v2 --use-beaker \
    --olmo-core-v2-commit-hash 326b7b01cc77750343510919801316d5a5622d87 \
    --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
    --huggingface-transformers-commit-hash  5db7e35d42636e86ee37a43f56a1587daadb7c1b \
    --huggingface-output-dir /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/$MODEL_NAME/ \
    --dtype float32




### NEED TO MOVE MODELS TO oe-adapt-default

###### need to update??

rm /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/$MODEL_NAME/step1484-hf/tokenizer_config.json
rm /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/$MODEL_NAME/step1484-hf/vocab.json
rm /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/$MODEL_NAME/step1484-hf/tokenizer.json
rm /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/$MODEL_NAME/step1484-hf/special_tokens_map.json
rm /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/$MODEL_NAME/step1484-hf/generation_config.json

cp /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/usable-tulu/olmo2-7B-lc-tulu3-olmo2-mix/generation_config.json /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/$MODEL_NAME/step1484-hf/

cp /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft-tokenizer-olmo-chat-template/* /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/$MODEL_NAME/step1484-hf/

mkdir /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/$MODEL_NAME/
mv /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/$MODEL_NAME/step1484-hf/* /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/$MODEL_NAME/

USABLE_LENGTH=4096
MODEL_PATH=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/$MODEL_NAME/
WANDB_RUN=placeholder
python scripts/submit_eval_jobs.py \
        --model_name $MODEL_NAME \
        --location $MODEL_PATH \
        --cluster ai2/saturn-cirrascale ai2/jupiter-cirrascale-2 ai2/ceres-cirrascale \
        --is_tuned \
        --priority high \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id $WANDB_RUN \
        --workspace olmo-instruct \
        --oe_eval_max_length $USABLE_LENGTH \
        --process_output r1_style \
        --skip_oi_evals \
        --beaker_image oe-eval-beaker/oe_eval_olmo3_auto