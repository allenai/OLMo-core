MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-test/step500
gantry run --cluster ai2/saturn -y --budget ai2/oceo --workspace ai2/flex2 \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras && uv pip install torch && uv pip install flash-attn" \
        --weka=oe-adapt-default:/weka/oe-adapt-default \
        --weka=oe-training-default:/weka/oe-training-default \
        --priority urgent \
        --gpus 8 \
        --beaker-image nathanl/open_instruct_auto \
        -- uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
            -i /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-test/step400 \
            -o /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-test/step400-hf3 \
            --skip-validation \
            --max-sequence-length 65536

        # --ref 3d6f43749887d66983257ca225819f6149069ce3 \


#### ABOVE DOESN'T WORK


# THIS WORKS:
### Go to a beaker session, my flexolmo directory (Olmo-core), activate olmo-core conda env
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-on-base-no-anneal/step594
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed-on-base-no-anneal/step1062
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-router-training-only/step594
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-anneal-frozen-router-mixed-sft/step1062
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-anneal-NO-frozen-router-mixed-sft/step1062
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-on-base-no-anneal/step150
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step150
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-base-no-anneal/step620
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed/step620
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-test-router
# MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-5B/step9537
# MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-anneal-no-expert-bias-5B/step9537
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-code-anneal-no-eb-5B/step620
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-5B/step620
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_5b_ann-math_base_again
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-math_base-math_mixed-code_mixed_olmo3_5b_ann
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_5b-router_sft_all_mixed/step1128
# MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-20B/step38147
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-20B/step620
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_20b-router_sft_all_mixed/step1128
# MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_20b_sft-router_sft_all_mixed/step1128"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_all_mixed/step1128"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_all_mixed-1_active_expert/step1128"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_all_mixed-2_active_expert/step1128"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-50b_olmo3_code_anneal-tool_use_general_mix_sft/step888"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-50b_olmo3_code_anneal-tool_use_only/step422"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-tool_use_general_mix/step888"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-tool_use_general_mix/step888"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_anneal-general_sft/step470"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_mix/step888"
    # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-olmo3_math-mixed-sft/step1062"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_general_only/step394"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_mix-4k-test/step888"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_anneal-general-olmo3_math-mix/step966"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_0.25_mix/step536"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_math_code_mix/step1224"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3-reasoning_sft_0.75/step842"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-reasoning_anneal-general-olmo3_reasoning-mix/step784"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_mix-unf-lm-head/step888"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_code_anneal-olmo3_code-general-mix-unf-lm-head/step782"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool-mix-unf-lm-head-embed/step888"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code/step1128"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-unf-lm-head-embed/step1128"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-unf-rt-4-domain/step1128"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool-mix-unf-lm-head-embed-1-active/step888"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b_code_1a-mix_sft/step782"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_1.0/step1128"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.50-redux/step562"
    "/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-olmo3_reasoning-fixed-20b-8k/step19074"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-reasoning_anneal-FIXED-general-olmo3_reasoning-mix/step784"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-20b_olmo3_math_anneal-math-mixed-sft/step1062"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.50-old-seeds/step562"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_sft-olmo3_code-tool-fixed_rt_sft/step1534"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-tool_use-router_sft-lr_1e-3/step1534"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-tool_use-router_sft-lr_5e-3/step1534"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-tool_use-router_sft-lr_1e-3-old_seeds/step1534"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_sft-olmo3_code-tool_use"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-0.1-1e-4/step152"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-0.05-1e-4/step76"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-full-1e-3/step1534"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-full-1e-4/step1534"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-0.25-1e-4/step382"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-0.5-1e-4/step766"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-1.0-5e-4/step1534"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-1.0-5e-5/step1534"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-0.75-1e-4/step1150"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-0.25-olmo3_3x_domain-1e-4/step400"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_base-olmo3_safety/step62"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_base-olmo3_safety-general-mix/step534"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-0.25-mix-tool-use-1.0/step686"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-0.25-mix-code-1.0/step494"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-0.25-mix-general-1.0/step680"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-0.25-mix-math-1.0/step822"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-code-only/step148"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-general-only/step398"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-math-only/step588"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-tool-use-only/step406"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_rl-olmo3_code-tool_use-average_all-no_rt"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-fixed-rt-merge-all-0.05-4dom-1e-4/step76"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-fixed-rt-merge-all-0.25-4dom-1e-4/step382"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_rl_x4"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-math_rl_x4-merge-all-0.05-4dom-1e-4/step76"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-merge-all-olmo3_3x_domain-0.05-1e-4/step80"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-merge-all-olmo3_3x_domain-0.25-1e-4/step400"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-50b_math2_anneal-olmo3_math_mix-attm2/step966"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-merge-all-olmo3_3x_domain-1.0-1e-4/step1602"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-50b_math2_anneal-olmo3_math_mix-unf-lm-emb/step966"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-merge-all-olmo3_3x_domain-0.01-1e-4/step16"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-merge-all-math_rl-code_rl-tool_use_sft-0.05-1e-4/step80"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/baileyk/olmo-sft-smoketest/smoke-dolci-drtulu-mix/step1000"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-kevin_med_anneal-10b-general-olmo3_science-biomed-mix/step694"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-olmo2_code_sft-tool_use_sft-safety_sft-0.05-1e-4/step60"
    "/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-50B/step95368"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_code_anneal-olmo3_coding/step310"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_code_anneal-tool-mix-unf-lm-head-embed/step888"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_base-olmo3_tool_use/step422"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_base-olmo3_tool_use-FIXED/step422"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_math_anneal-general-olmo2_math-mix/step1062"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_math_anneal-olmo3_math-mix-4k/step500"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_math_anneal-general-olmo3_math-mix-4k/step966"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_all-0.05-1e-4/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_4-math_rl-0.05-1e-4/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_4-code_rl-0.05-1e-4/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_3-math_code_rl-0.05-1e-4/step66"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_3-olmo3_code_rl-olmo2_math"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-olmo3_sft_3-olmo2_code-olmo3_math"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_3-olmo2_code-olmo3_math-0.05-1e-4/step60"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_3-olmo3_code_rl-olmo2_math-0.05-1e-4/step70"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_3-olmo2_code_math-0.05-1e-4/step62"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4-1-active/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4-2-active/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4-3-active/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4-4-active/step66"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.25-1e-4-3-active/step332"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.25-1e-4-4-active/step332"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-final-sft-only-0.05-1e-4/step66"
    "/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-pmc-10b-8k/step9537"
    "/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-7b-test-anneal-pmc-10b/step19074"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-pmc-10b-general-olmo3_science-biomed-mix/step694"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-pmc-10b-general-olmo3_science-biomed-mix/step694"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_reasoning-mix/step784"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_math_code_tool_use_safety-1.0/step1764"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_tool_use-mix/step888"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_safety-mix/step534"
    "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/olmo2-7B-sft/math_expert_sft_mixed/step1062"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-50b_ol3_code_ann-general-olmo3_code-mix/step782"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-5e-4-4-active/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-5e-4-3-active/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-5e-4-2-active/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-5e-4-1-active/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-final-sft-only-0.05-1e-4/step66"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.25-1e-4-4-active/step332"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.25-1e-4-3-active/step332"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-1.0-1e-4-4-active/step1328"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-1.0-1e-4-3-active/step1328"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains-0.05-1e-4/step66"
MODEL_PATHS=(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-math_rl-code_rl-tool_use_sft-0.05-1e-4/step62"
)
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
            -i $MODEL_PATH \
            -o $MODEL_PATH-hf \
            --skip-validation \
            --max-sequence-length 65536
done


# convert FROM HF
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool-mix-unf-lm-head-embed/step888"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head__1__1771484873_checkpoints/step_500"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-froz-exp1-rt/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-froz-exp1-rt__1__1772677347_checkpoints/step_50"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-froz-exp1-rt-2/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-froz-exp1-rt-2__1__1772683398_checkpoints/step_200"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_200"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_base-olmo3_safety-general-mix/step534-hf"
MODEL_PATHS=(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_math_anneal-olmo3_math-mix-4k/step500-hf/grpo_math_only_flex-2x7b-50b_ol3_ann-ol3_sft_math-6e-7-unf/grpo_math_only_flex-2x7b-50b_ol3_ann-ol3_sft_math-6e-7-unf__1__1773370912_checkpoints/step_200"
)
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
uv run python src/examples/huggingface/convert_checkpoint_from_hf.py \
            -i $MODEL_PATH \
            -o $MODEL_PATH-oc \
            -c /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool-mix-unf-lm-head-embed/step888/config.json \
            --skip-validation 
done

    "${CODE_BASE},${CODE_EXPERT_2},${MATH_EXPERT}|${CKPT_DIR}/FlexOlmo-3x7B-code_base-code_only-math_mixed"
    "${MATH_BASE},${CODE_EXPERT_3},${MATH_EXPERT}|${CKPT_DIR}/FlexOlmo-3x7B-math_base-code_mixed-math_mixed"

MODEL_PATHS=(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_sft-olmo3_code-tool_use"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "Processing: $MODEL_PATH"
    
    mkdir -p "${MODEL_PATH}/model_and_optim" && \
    cp -f "${MODEL_PATH}"/*.distcp "${MODEL_PATH}/model_and_optim/" && \
    cp -f "${MODEL_PATH}/.metadata" "${MODEL_PATH}/model_and_optim/" && \
    python src/examples/huggingface/convert_checkpoint_to_hf.py \
        -i "$MODEL_PATH" \
        -o "${MODEL_PATH}-hf" \
        --skip-validation \
        --max-sequence-length 65536 && \
    cp /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-test-router-hf/chat_template.jinja "${MODEL_PATH}"
    
    if [ $? -ne 0 ]; then
        echo "Failed: $MODEL_PATH"
    fi
done




-----------


uv run python src/examples/huggingface/convert_checkpoint_from_hf.py \
            -i allenai/Olmo-3-1025-7B \
            -o /weka/oe-adapt-default/jacobm/olmo-core-checkpoints/olmo-3-7b-base/ \
            -c /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool-mix-unf-lm-head-embed/step888/config.json \
            --skip-validation 
done