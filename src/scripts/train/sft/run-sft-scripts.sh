# IN OLMO-CORE

gantry run \
    --cluster ai2/saturn-cirrascale \
    --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/jacobm \
    --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
    --weka=oe-training-default:/weka/oe-training-default \
    -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
        --dataset_mixer_list jacobmorrison/OpenThoughts3-743k-QwQ-generations-32k-parsed 1.0 jacobmorrison/OpenThoughts3-456k 1.0 \
        --tokenizer_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921-hf \
        --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/sft/rl-sft-32k/jacobmorrison/openthoughts3-full-regenerated \
        --visualize True \
        --chat_template_name jacobtest2 \
        --dataset_skip_cache True \
        --max_seq_length 32768

# usable:
gantry run \
    --cluster ai2/saturn-cirrascale \
    --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/jacobm \
    --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
    --weka=oe-training-default:/weka/oe-training-default \
    --env-secret HF_TOKEN=HF_TOKEN \
    -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
        --dataset_mixer_list ai2-adapt-dev/oasst1_converted 1.0 \
            ai2-adapt-dev/flan_v2_converted 1.0 \
            allenai/hardcoded-integration-tests 1.0 \
            ai2-adapt-dev/no_robots_converted 1.0 \
            jacobmorrison/wildchat_perturbed_6000_replaced_no_keyword 1.0 \
            ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k 1.0 \
            ai2-adapt-dev/numinamath_tir_math_decontaminated 1.0 \
            ai2-adapt-dev/personahub_code_v2_34999 1.0 \
            ai2-adapt-dev/evol_codealpaca_heval_decontaminated 1.0 \
            saumyamalik/tulu-3-sft-coconot-regenerated 1.0 \
            ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k 1.0 \
            ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k 1.0 \
            ai2-adapt-dev/tulu_v3.9_sciriff_10k 1.0 \
            ai2-adapt-dev/tulu_v3.9_table_gpt_5k 1.0 \
            ai2-adapt-dev/tulu_v3.9_aya_100k 1.0 \
            finbarr/tulu-3-sft-personas-instruction-following-o3 1.0 \
            finbarr/tulu-3-sft-personas-math-o3 1.0 \
            finbarr/tulu-3-sft-personas-math-grade-o3 1.0 \
            finbarr/tulu-3-sft-personas-algebra-o3 1.0 \
            VGraf/toolu-sft-mix-T2-system-prompt 0.3 \
            saurabh5/rlvr-code-data-python-sft 1.0 \
            saurabh5/llama-nemotron-rlvr-code-stdio-sft 1.0 \
            allenai/IF_sft_data_verified 1.0 \
            jacobmorrison/OpenThoughts3-456k-no-cot 1.0 \
            jacobmorrison/verifiable-tasks-o3-7500 1.0 \
        --tokenizer_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921-hf \
        --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/sft/usable-tulu-16k/tulu3-olmo2-mix-remov_replac-100k_toolu-fae_ver-sau_code-val_if-ot3_456k \
        --visualize True \
        --chat_template_name olmo \
        --max_seq_length 16384
        

saurabh5/rlvr-code-data-python-sft 1.0 \
saurabh5/llama-nemotron-rlvr-code-stdio-sft 1.0 \
allenai/IF_sft_data_verified 1.0 \
jacobmorrison/OpenThoughts3-456k-no-cot 1.0 \
jacobmorrison/verifiable-tasks-o3-7500 1.0 \

# usable:
gantry run \
    --cluster ai2/saturn-cirrascale \
    --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
    --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
    --weka=oe-training-default:/weka/oe-training-default \
    --env-secret HF_TOKEN=HF_TOKEN \
    --gpus 1 \
    --priority urgent \
    -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
        --dataset_mixer_list DATASET 0.549 \
        --tokenizer_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921-hf \
        --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/sft/usable-tulu-16k/DATA \
        --visualize True \
        --chat_template_name olmo \
        --max_seq_length 16384

# reasoning:
gantry run \
    --cluster ai2/saturn-cirrascale \
    --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
    --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
    --weka=oe-training-default:/weka/oe-training-default \
    --env-secret HF_TOKEN=HF_TOKEN \
    --gpus 8 \
    --priority urgent \
    -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
        --dataset_mixer_list jacobmorrison/OpenThoughts3-456k 1.0 \
            jacobmorrison/OpenThoughts3-743k-QwQ-generations-32k-parsed-2 1.0 \
        --tokenizer_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921-hf \
        --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/sft/rl-sft-32k/OpenThoughts3-full-regenerations-2 \
        --visualize True \
        --chat_template_name olmo_thinker \
        --max_seq_length 32768


python src/scripts/train/sft/OLMo2-7B-sft.py launch \
    olmo2-7B-sft-8gpu-tulu-3-sft-16k tulu-3-sft-olmo-2 /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921 ai2/titan-cirrascale \
    --trainer.callbacks.wandb.enabled=True \
    --launch.num_gpus=8 \
    --launch.priority=high


openthoughts3-456k
openthoughts3-456k-no-cot
usable-tulu-integration-test-tulu_3_all-toolu_T2_336k
usable-tulu-integration-test-tulu_3_all-toolu_T2_200k
usable-tulu-integration-test-tulu_3_all-toolu_T2_100k
usable-tulu-integration-test-tulu_3_all-toolu_T2_50k

# lc olmo 2: /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_4K_20B/step33379
# normal olmo 2: /weka/oe-training-default/ai2-llm/checkpoints/dustins/OLMo-2-1124-7B

# MAX_LENGTH=32768
MAX_LENGTH=16384

# USABLE!!!!!
python src/scripts/train/sft/OLMo2-7B-sft.py launch \
    olmo2-7B-lc-tulu3-olmo2-mix \
        tulu3-olmo2-mix \
        /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_4K_20B/step33379 \
        ai2/jupiter-cirrascale-2 \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=5e-5 \
    --seq_len=16384 \
    --launch.num_gpus=8 \
    --num_nodes=1 \
    --launch.priority=urgent

# REASONING!!!!!
python src/scripts/train/sft/OLMo2-7B-sft.py launch \
    olmo2-7B-lc-OpenThoughts3-1.2M \
        OpenThoughts3-1.2M \
        /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_4K_20B/step33379 \
        ai2/titan-cirrascale \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=5e-5 \
    --seq_len=32768 \
    --launch.num_gpus=8 \
    --num_nodes=4 \
    --launch.priority=urgent

INPUT_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7B-lc-tulu3-olmo2-mix-remov_replac/step1368
MODEL_NAME=olmo2-7B-lc-tulu3-olmo2-mix-remov_replac
gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras" \
        --weka=oe-adapt-default:/weka/oe-adapt-default \
        --weka=oe-training-default:/weka/oe-training-default \
        --priority urgent \
        --gpus 1 \
        -- /root/.local/bin/uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
            -i $INPUT_PATH \
            -o /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/usable-tulu/$MODEL_NAME \
            --max-sequence-length 65536

olmo2-7B-lc-openthoughts3-456k/step42798
olmo2-7B-lc-openthoughts3-456k-no-cot/step2742

##### REASONING:
cp /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft-tokenizer-olmo_thinker-chat-template/* \
    <OUTPUT_DIR>

#### USABLE:
cp /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft-tokenizer-olmo-chat-template/* \
    <OUTPUT_DIR>



# ALL
EXP_NAME=olmo2-7B-lc-tulu3_toolu100k_base_replacements_removals
MODEL_PATH=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo2-7B-sft/olmo2-7B-lc-tulu3_toolu100k_base_replacements_removals/step1376-hf
WANDB_RUN=placeholder
python scripts/submit_eval_jobs.py \
        --model_name $EXP_NAME \
        --location $MODEL_PATH \
        --cluster ai2/saturn-cirrascale ai2/jupiter-cirrascale-2 ai2/ceres-cirrascale \
        --is_tuned \
        --priority high \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id $WANDB_RUN \
        --workspace tulu-3-results \
        --oe_eval_max_length 32768 \
        --oe_eval_stop_sequences '</answer>,<|endoftext|>' \
        --process_output r1_style \
        --oe_eval_tasks alpaca_eval_v3::hamish_zs_reasoning \
        --skip_oi_evals 











# REASONING:
EXP_NAME=olmo2-7B-sft-16k-jacobm-test-non-reasoning-evals
MODEL_PATH=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo2-7B-original-sft-tulu-3-sft-4k-jacobtest2-chat-template-shuffled-packed-2_epoch-1e-5_lr/step4960-hf
WANDB_RUN=https://wandb.ai/ai2-llm/tylerr-7B-sft/runs/jbu70nxv

python scripts/submit_eval_jobs.py \
    --model_name $EXP_NAME \
    --location $MODEL_PATH \
    --cluster ai2/saturn-cirrascale ai2/jupiter-cirrascale-2 \
    --is_tuned --workspace "tulu-3-results" --priority high --preemptible \
    --use_hf_tokenizer_template --run_oe_eval_experiments --skip_oi_evals \
    --oe_eval_max_length 32768 \
    --run_id $WANDB_RUN \
    --evaluate_on_weka \
    --oe_eval_tasks minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,alpaca_eval_v3::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker \
    --oe_eval_stop_sequences '</answer>' \
    --process_output r1_style


# python scripts/submit_eval_jobs.py \
#     --model_name $EXP_NAME \
#     --location $MODEL_PATH \
#     --cluster ai2/saturn-cirrascale ai2/neptune-cirrascale \
#     --is_tuned --priority high --preemptible --skip_oi_evals \
#     --use_hf_tokenizer_template --run_oe_eval_experiments --workspace tulu-3-results \
#     --evaluate_on_weka \
#     --run_id $WANDB_RUN \
#     --oe_eval_max_length 32768


python scripts/submit_eval_jobs.py \
    --model_name olmo2-7B-original-sft-tulu-3-sft-4k \
    --location /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo2-7B-original-sft-tulu-3-sft-4k/step5444-hf \
    --cluster ai2/saturn-cirrascale ai2/neptune-cirrascale \
    --is_tuned \
    --priority high \
    --preemptible \
    --use_hf_tokenizer_template \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_id https://wandb.ai/ai2-llm/jacobm-7B-sft/runs/oxs74w8z \
    --oe_eval_max_length 16384 \
    --workspace tulu-3-results \
    --skip_oi_evals