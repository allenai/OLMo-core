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
# SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/code-general-mix
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-50B/step95368
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-frozen-router-5B-1-active/step4769
# SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_code-mix
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_mixed-code_mixed_olmo3_50b_ann_with_sft-math_base_again
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/all
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-anneal-no-expert-bias/step95368
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo3_math-1.0
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-50b_math2_anneal-olmo3_math_mix-unf-lm-emb \
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
    --model_name olmoe-2x7b-unfrozen-lm-head-embed \
    --dataset_path $SFT_DATASET

# MATH:
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-anneal-no-expert-bias/step95368
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-olmo3_math_anneal-50b-8k/step47684
# SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_math-mix
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/olmo3_math/
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-olmo3_50b_math_anneal-olmo3_math-mix-4k \
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
    --workspace ai2/olmo-instruct \
    --model_name olmoe-2x7b \
    --dataset_path $SFT_DATASET

BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-olmo3_reasoning-20b-8k/
SFT_DATASET=

# BIOMED:
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_science-biomed-mix
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-pmc-10b-8k/step9537
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-pmc-10b-general-olmo3_science-biomed-mix \
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
    --workspace ai2/olmo-instruct \
    --model_name olmoe-2x7b-unfrozen-lm-head-embed \
    --dataset_path $SFT_DATASET

# SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_science-biomed-mix
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-7b-test-anneal-pmc-10b/step19074
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-7b-anneal-olmo3_code-50b/step95368
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_code-mix
uv run python src/scripts/train/sft/OLMo-sft.py launch \
    olmo2-7b-50b_ol3_code_ann-general-olmo3_code-mix \
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
    --workspace ai2/olmo-instruct \
    --model_name olmo2-7b \
    --dataset_path $SFT_DATASET

---

# REASONING:
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-olmo3_reasoning-20b-8k/step19074
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-olmo3_reasoning-fixed-20b-8k/step19074
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_reasoning-mix
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-reasoning_anneal-FIXED-general-olmo3_reasoning-mix \
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
    --num_nodes=2 \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --model_name olmoe-2x7b \
    --dataset_path $SFT_DATASET

SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_code-mix
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/weijias/OLMo2-7B-anneal-from-stage1-no-math/step11921/model_and_optim
uv run python src/scripts/train/sft/OLMo-sft.py launch \
    olmo2-7b-BASE-general-olmo3_code-mix \
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
    --workspace ai2/olmo-instruct \
    --model_name olmo2-7b \
    --dataset_path $SFT_DATASET

# ALL MIXED
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-7b-full-mix-150b/step286103/model_and_optim
# BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/weijias/OLMo2-7B-anneal-from-stage1-no-math/step11921/model_and_optim
BASE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex-olmo/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr_step_350/model_and_optim
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo2_math-olmo3_code_tool_use_safety-1.0
uv run python src/scripts/train/sft/OLMo-sft.py launch \
    olmo2-7b-CONTINUED-mixed_all_sft-fixed \
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
    --workspace ai2/olmo-instruct \
    --model_name olmo2-7b \
    --dataset_path $SFT_DATASET

# SAFETY
/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/olmo3_safety


# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-50B/step95368
MIX=olmo3_tool_use
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/$MIX
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-math_base-$MIX-FIXED \
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
    --model_name olmoe-2x7b-unfrozen-lm-head-embed \
    --dataset_path $SFT_DATASET
--------


# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-50B/step95368
# SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/code-general-mix
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-olmo3_math_anneal-5b/step9537
# BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-math-anneal-frozen-router-5b/step9537
BASE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-olmo3_reasoning-20b-8k/step19074
SFT_DATASET=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/mixed/
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-20b_olmo3_math_anneal-math-mixed-sft \
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
    flexolmo-2x7b-no_anneal-UNFROZEN-tool_use_general_mix-2k \
        $BASE_CKPT \
        ai2/jupiter \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=1e-4 \
    --train_module.state_dict_load_opts.flatten_optimizer_state_dict=True \
    --train_module.state_dict_load_opts.strict=False \
    --launch.priority=urgent \
    --seq_len=1024 \
    --launch.num_gpus=8 \
    --num_nodes=8 \
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

-----

# SFT datasets:
# base: 
# each mix:
    # code: 
    # math: 
    # safety: 
    # tool use: 
# router training mix: /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo3_math_code_tool_use_safety-0.05
# all mixed full olmo 2 math: /weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo2_math-olmo3_code_tool_use_safety-1.0


#### 2x7B models:
MATH_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-math_anneal-50b/step47684
MATH_SFT=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/mixed
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-math-sft \
        $MATH_CKPT \
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
    --dataset_path $MATH_SFT

CODE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7B-code_anneal-50b/step47684
CODE_SFT=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_code-mix
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-code-sft \
        $CODE_CKPT \
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
    --dataset_path $CODE_SFT

TOOL_USE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base
TOOL_USE_SFT=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tool-use-general-mix
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-tool-use-sft-unfrozen \
        $TOOL_USE_CKPT \
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
    --model_name olmoe-2x7b-unfrozen-lm-head-embed \
    --dataset_path $TOOL_USE_SFT

SAFETY_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-2x7b-math-base
SAFETY_SFT=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/olmo3_safety-general-mix
uv run python src/scripts/train/sft/FlexOlmo-SFT.py launch \
    flexolmo-2x7b-safety-sft \
        $SAFETY_CKPT \
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
    --dataset_path $SAFETY_SFT

#### 7B baselines:
RETRAIN_MID_SFT=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo2_math-olmo3_code_tool_use_safety-1.0
RETRAIN_MID_TOO=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-7b-full-mix-150b/step286103
uv run python src/scripts/train/sft/OLMo-sft.py launch \
    olmo2-7b-retrain-post-train-sft \
        $RETRAIN_MID_TOO \
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
    --dataset_path $RETRAIN_MID_SFT

RETRAIN_POST_ONLY_SFT=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo2_math-olmo3_code_tool_use_safety-1.0
RETRAIN_POST_ONLY=/weka/oe-training-default/ai2-llm/checkpoints/weijias/OLMo2-7B-anneal-from-stage1-no-math/step11921/model_and_optim
uv run python src/scripts/train/sft/OLMo-sft.py launch \
    olmo2-7b-retrain-post-train-sft \
        $RETRAIN_POST_ONLY \
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
    --model_name olmo2-7b \
    --dataset_path $RETRAIN_POST_ONLY_SFT

CONTINUED_SFT=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/router-training-ablations/general-olmo2_math-olmo3_code_tool_use_safety-1.0
CONTINUED_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex-olmo/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr_step_350/model_and_optim
uv run python src/scripts/train/sft/OLMo-sft.py launch \
    olmo2-7b-continued-post-train-sft \
        $CONTINUED_CKPT \
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
    --model_name olmo2-7b \
    --dataset_path $CONTINUED_SFT

CODE_SFT=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/general-olmo3_code-mix
CODE_CKPT=/weka/oe-training-default/jacobm/flexolmo/checkpoints/flex-7b-anneal-code-50b/step95368
uv run python src/scripts/train/sft/OLMo-sft.py launch \
    olmo2-7b-code-sft \
        $CODE_CKPT \
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
    --model_name olmo2-7b \
    --dataset_path $CODE_SFT

TOOL_USE_SFT=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/tool-use-general-mix
TOOL_USE_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/weijias/OLMo2-7B-anneal-from-stage1-no-math/step11921/model_and_optim
uv run python src/scripts/train/sft/OLMo-sft.py launch \
    olmo2-7b-tool-use-sft \
        $TOOL_USE_CKPT \
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
    --model_name olmo2-7b \
    --dataset_path $TOOL_USE_SFT

SAFETY_CKPT=/weka/oe-training-default/ai2-llm/checkpoints/weijias/OLMo2-7B-anneal-from-stage1-no-math/step11921/model_and_optim
SAFETY_SFT=/weka/oe-training-default/ai2-llm/jacobm/data/flexolmo/sft/olmo3_safety-general-mix
uv run python src/scripts/train/sft/OLMo-sft.py launch \
    olmo2-7b-safety-sft \
        $SAFETY_CKPT \
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
    --model_name olmo2-7b \
    --dataset_path $SAFETY_SFT