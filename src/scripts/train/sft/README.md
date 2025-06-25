# Supervised Finetuning (SFT) using Olmo-core

In our experiments, we observed *8x compute efficiency* when running SFT on an OLMo2 7B model at
4k context length using Olmo-core compared to open-instruct. This comes from a combination of a
more efficient training codebase and better dataloading via a bin-packing algorithm.

You can follow the instructions here to generate an Olmo-core compatable SFT dataset, launch SFT, and run evaluavtion with the resulting model.

## Prepping a Dataset

1. Checkout [open-instruct](https://github.com/allenai/open-instruct) and run a command such as:

    ```bash
    gantry run \
        --cluster ai2/phobos-cirrascale \
        --allow-dirty --timeout -1 -y --budget ai2/oe-adapt \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --add_bos \
            --dataset_mixer_list jacobmorrison/OpenThoughts3-1.2M-no-cot 1.0 \
            --tokenizer_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921-hf \
            --output_dir /weka/oe-training-default/tylerr/data/sft/jacobmorrison-OpenThoughts3-1.2M
    ```

    This command will also write tokenizer config files that will be needed later.

    > TIP: Using `uv` you can install gantry on your machine with `uv tool install beaker-gantry`.

2. Add your dataset to the `sft_datasets.yaml` file in this directory.

## Training

1. Ensure that the beaker workspace you are using (defaults to `ai2/olmo-instruct`) has the following secrets configured.
    * `{beaker_username}_BEAKER_TOKEN`
    * `{beaker_username}_AWS_CREDENTIALS`
    * `{beaker_username}_AWS_CONFIG`
    * `{beaker_username}_WANDB_API_KEY`  (optional)
    * `{beaker_username}_COMET_API_KEY`  (optional)

2. Ensure that the dataset and model you want to use are available on the correct storage bucket for the cluster you are using.
    * For example, if you are using `ai2/jupiter-cirrascale-2`, the dataset and model should be available on the `/weka/oe-training-default/ai2-llm` bucket.
    * If you are using ai2/augusta-google-1, the dataset and model should be available on the `gs://ai2-llm` bucket.
    * Data can be copied to/from gcs/weka using the `gsutil` command line tool. E.g., `gsutil cp -r /path/to/dataset gs://ai2-llm/path/to/dataset`

3. Launch the SFT training script using the following command:

    ```bash
    python src/scripts/train/sft/OLMo2-7B-sft.py launch <run-name> <olmo-core checkpoint> <cluster> \
        --override.option="override"
    ```

    for example:

    ```bash
    python src/scripts/train/sft/OLMo2-7B-sft.py launch \
        olmo2-7B-sft-take2-8gpu /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921 ai2/jupiter-cirrascale-2 \
        --trainer.callbacks.wandb.enabled=True \
        --launch.num_gpus=8
    ```

    > TIP: The "launch" command automatically creates a Beaker experiment and runs the exact same command with "train" substituted for launch.

## Evaluation

1. Convert the model to a Hugging Face model using a command such as:

    ```bash
    gantry run --cluster ai2/phobos-cirrascale --timeout -1 -y --budget ai2/oe-adapt \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras" \
        --weka=oe-adapt-default:/weka/oe-adapt-default \
        --weka=oe-training-default:/weka/oe-training-default \
        -- /root/.local/bin/uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
            -i /weka/oe-training-default/ai2-llm/checkpoints/tylerr/olmo2-7B-sft/olmo2-7B-sft-take2-8gpu/step4143 \
            -o /weka/oe-adapt-default/tylerr/checkpoints/olmo2-7B-sft/olmo2-7B-sft-take2-8gpu/step4143-hf \
            --max-sequence-length 4096
    ```

2. Copy over the tokenizer files to your hf model directory. These files are generated with and located in
    the same directory as the input dataset.

3. Launch evaluations using the submit_eval_jobs.sh script in `open-instruct` using a command such as:

    ```bash
    python scripts/submit_eval_jobs.py \
        --model_name olmo2-7b-sft-fromolmocore \
        --location /weka/oe-adapt-default/tylerr/checkpoints/olmo2-7B-sft/olmo2-7B-sft-take2-8gpu/step4143-hf/ \
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
