# Supervised Finetuning (SFT) using Olmo-core

In our experiments, we observed *8x compute efficiency* when running SFT on an OLMo2 7B model
using Olmo-core compared to open-instruct. This comes from a combination of a more efficient
training codebase and better dataloading via a bin-packing algorithm.

You can follow the instructions here to generate an Olmo-core compatable SFT dataset, launch SFT, and run evaluation with the resulting model.

## Prepping a Dataset

1. Checkout [open-instruct](https://github.com/allenai/open-instruct) and run a command such as:

    ```bash
    gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/jacobm \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list allenai/hardcoded-integration-tests 1.0 \
                jacobmorrison/verifiable-tasks-o3-7500 1.0 \
            --tokenizer_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921-hf \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/sft/usable-tulu-16k/example-tokenized-dataset \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 16384
    ```

    *Until these two PRs: [1](https://github.com/allenai/open-instruct/pull/765) and [2](https://github.com/allenai/open-instruct/pull/749) are merged, you need to check out the branch `tyler/olmocore-tokenization-bug-fix-label-mask`*

    This command will also write tokenizer config files to `<output_dir>/tokenizer` that will be needed later.

    *Be careful with your choice of chat template!* It is highly recommended to use the `olmo` chat template for tokenization. Olmo-core uses `[eos]` tokens to find document boundaries, and the `olmo` chat template uses a single `eos` token to mark the end of a conversation, enabling document packing to work correctly.

    > TIP: Using `uv` you can install gantry on your machine with `uv tool install beaker-gantry`.

## Training

1. Ensure that the beaker workspace you are using has the following secrets configured.
    * `{beaker_username}_BEAKER_TOKEN`
    * `{beaker_username}_AWS_CREDENTIALS`
    * `{beaker_username}_AWS_CONFIG`
    * `{beaker_username}_WANDB_API_KEY`  (optional)
    * `{beaker_username}_COMET_API_KEY`  (optional)

2. Ensure that the dataset and model you want to use are available on the correct storage bucket for the cluster you are using.
    * For example, if you are using `ai2/jupiter-cirrascale-2`, the dataset and model should be available on the `/weka/oe-training-default/ai2-llm` bucket.
    * If you are using ai2/augusta-google-1, the dataset and model should be available on the `gs://ai2-llm` bucket.
    * Data can be copied to/from gcs/weka using the `gsutil` command line tool.
        * From a machine with weka mounted: `gsutil cp -r /weka/<bucket-name>/path/to/dataset gs://ai2-llm/path/to/dataset`

3. Ensure that the tokenizer used when prepping your dataset matches the one you have configured for SFT in Olmo-core.

4. Launch the SFT training script using a command like:

    ```bash
    CKPT="/weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_4K_20B/step33379"
    python src/scripts/train/sft/OLMo2-7B-sft.py launch \
        model_name $CKPT \
            ai2/jupiter-cirrascale-2 \
        --trainer.callbacks.wandb.enabled=True \
        --trainer.max_duration.value=2 \
        --train_module.optim.lr=5e-5 \
        --launch.priority=high \
        --seq_len=16384 \
        --launch.num_gpus=8 \
        --num_nodes=1 \
        --budget ai2/oe-adapt \
        --workspace ai2/<your_workspace> \
        --model_name olmo2-7b \
        --dataset_path /weka/<bucket-name>/path/to/dataset
    ```

    > TIP: The "launch" command automatically creates a Beaker experiment and runs the exact same command remotely with "train" substituted for launch.
    > TIP: Highly recommended: Tokenize and train at the same context length (recommended 16k)
    > TIP: `--model_name`: Loads the correct model config. Currently supported: `olmo2-7b` and `olmo3-7b`
    > TIP: `--dataset_path`: Path to your tokenized dataset. If on Augusta, you must copy it to GCP. Include `gs://`.

## Evaluation

1. Convert the model to a Huggingface model using a command such as:

### OLMo 2
    ```bash
    MODEL_NAME=olmo3_r2_7t-tulu_3_sft-5e_-5-3_epochs
    INPUT_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/$MODEL_NAME/step2226
    gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/<your_workspace> \
            --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras" \
            --weka=oe-adapt-default:/weka/oe-adapt-default \
            --weka=oe-training-default:/weka/oe-training-default \
            --priority high \
            --gpus 1 \
            -- /root/.local/bin/uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
                -i $INPUT_PATH \
                -o /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/usable-tulu/$MODEL_NAME \
                --max-sequence-length 65536
    ```

    > TIP: Look at the logs of your training job to find the path your final checkpoint was saved to.
    > TIP: Recommended to use one GPU. This is currently the only way to "reserve" CPUs for your job, and conversion takes <10 minutes.
    > TIP: If you'll be evaluating your model using `submit_eval_jobs.py` in open-instruct, your converted model must be saved in the `oe-adapt-default` weka bucket.

### OLMo 3

    Currently, the only way to convert an OLMo 3 checkpoint to Huggingface is with the [olmo cookbook](https://github.com/allenai/olmo-cookbook). Follow their installation instructions, and then you can convert your model with a command like:

    ```bash
    olmo-cookbook-eval convert \
        "/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/$MODEL_NAME/$STEP" \
        -t olmo-core-v2 --use-beaker \
        --olmo-core-v2-commit-hash 326b7b01cc77750343510919801316d5a5622d87 \
        --huggingface-transformers-git-url https://github.com/2015aroras/transformers.git \
        --huggingface-transformers-commit-hash  5db7e35d42636e86ee37a43f56a1587daadb7c1b \
        --huggingface-output-dir /oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo3-hparam-search/$MODEL_NAME/ \
        --dtype float32
    ```
    > TIP: If you'll be evaluating your model using `submit_eval_jobs.py` in open-instruct, your converted model must be saved in the `oe-adapt-default` weka bucket.

2. Copy over the tokenizer files to your hf model directory. If you havent made any changes to tokenization, you can copy the files located at `/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft-tokenizer-(olmo|olmo_thinker)-chat-template/`:

    ```bash
    cp /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft-tokenizer-olmo-chat-template/* /weka/oe-adapt-default/path/to/huggingface/model
    ```

    *If training a reasoning model, copy these tokenizer files instead*:
    ```bash
    cp /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft-tokenizer-olmo_thinker-chat-template/* /weka/oe-adapt-default/path/to/reasoning/model
    ```

    > NOTE: Be careful with this step, and it's worth double checking the tokenizer configuration. We plan to automate this in the future to help avoid bugs resulting from manual tokenizer configuration.
    > TIP: If you're copying tokenizer files to make checkpoints saved in the same directory, you can run a command like this instead:

    ```bash
    find /weka/oe-adapt-default/path/where/all/your/models/are/saved/ -maxdepth 1 -type d -name "*regex-for-model-names*" -exec cp /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft-tokenizer-olmo-chat-template/* {} \;
    ```

    This will copy all of the relevant tokenizer files to every 1-level-down subdirectory in your model directory.

3. Launch evaluations using the submit_eval_jobs.sh script in `open-instruct` using a command such as:

    ```bash
    python scripts/submit_eval_jobs.py \
        --model_name olmo2-7b-sft-tulu3mix-fromolmocore \
        --location /weka/oe-adapt-default/tylerr/checkpoints/olmo2-7B-sft/olmo2-7B-sft-tulu3mix/step4143-hf \
        --cluster ai2/saturn-cirrascale ai2/neptune-cirrascale \
        --is_tuned \
        --priority high \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id https://wandb.ai/ai2-llm/tylerr-7B-sft/runs/nitys50e \
        --oe_eval_max_length 4096 \
        --workspace tulu-3-results \
        --skip_oi_evals
    ```

    > TIP: If you're evaluating olmo3 models, you must include the flag `--beaker_image oe-eval-beaker/oe_eval_olmo3_auto`
