# Supervised Finetuning (SFT) using Olmo-core

In our experiments, we observed *8x compute efficiency* when running SFT on an OLMo 7B model
using Olmo-core compared to open-instruct. This comes from a combination of a more efficient
training codebase and better dataloading via a bin-packing algorithm.

You can follow the instructions here to generate an Olmo-core compatable SFT dataset, launch SFT, and run evaluation with the resulting model.

## Prepping a Dataset

1. Check out [open-instruct](https://github.com/allenai/open-instruct) and run a command such as:

    ```bash
    gantry run \
        --cluster ai2/neptune-cirrascale \
        --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/jacobm \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=HF_TOKEN \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list hf-dataset/number_1 1.0 \
                hf-dataset/number_2 1.0 \
            --tokenizer_name_or_path /path/to/hf-style/tokenizer \
            --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/sft/usable-tulu-16k/example-tokenized-dataset \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 32768
    ```

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
    * For example, if you are using `ai2/jupiter`, the dataset and model should be available on the `/weka/oe-training-default/ai2-llm` bucket.
    * If you are using ai2/augusta, the dataset and model should be available on the `gs://ai2-llm` bucket.
    * Both can be copied to/from gcs/weka using the `gsutil` command line tool.
        * From a machine with weka mounted: `gsutil -m cp -r /weka/<bucket-name>/path/to/dataset_or_model gs://ai2-llm/path/to/dataset_or_model`

3. Ensure that the tokenizer used when prepping your dataset matches the one you have configured for SFT in Olmo-core.

4. Launch the SFT training script using a command like:

    ```bash
    BASE_CKPT="/weka/oe-training-default/ai2-llm/path/to-base/checkpoint/step12345/model_and_optim"
    python src/scripts/train/sft/Olmo-3-7B-SFT.py launch \
        MODEL_NAME_HERE $BASE_CKPT \
            ai2/jupiter \
        --trainer.callbacks.wandb.enabled=True \
        --trainer.max_duration.value=2 \
        --train_module.optim.lr=5e-5 \
        --launch.priority=high \
        --seq_len=32768 \
        --launch.num_gpus=8 \
        --num_nodes=1 \
        --budget ai2/BUDGET \
        --workspace ai2/<your_workspace> \
        --dataset_path /weka/<bucket-name>/path/to/dataset
    ```

    * Tips:
        * The "launch" command automatically creates a Beaker experiment and runs the exact same command remotely with "train" substituted for launch.
        * Highly recommended: Tokenize and train at the same context length (recommended 32k)
        * Make sure to use the right script for your model: Currently supported: `Olmo-2-7B`, `Olmo-3-7B`, and `Olmo-3-32B`.
        * `--dataset_path`: Path to your tokenized dataset. If on Augusta, you must copy it to GCP. Include `gs://`.
        * Include `model_and_optim` at the end of your base checkpoint.

## Evaluation

1. Convert the model to a Huggingface model using a command such as:

    ```bash
    gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/<your_workspace> \
            --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras" \
            --weka=oe-adapt-default:/weka/oe-adapt-default \
            --weka=oe-training-default:/weka/oe-training-default \
            --priority high \
            --gpus 1 \
            -- /root/.local/bin/uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
                -i /weka/oe-training-default/$USER/checkpoints/path-to-model/stepFINAL_STEP \
                -o /weka/oe-adapt-default/$USER/checkpoints/path-to-model/stepFINAL_STEP-hf \
                --max-sequence-length 65536
    ```

    * Tips:
        * Look at the logs of your training job to find the path your final checkpoint was saved to.
        * Recommended to use one GPU. This is currently the only way to "reserve" CPUs for your job, and conversion takes <10 minutes.
        * If you'll be evaluating your model using `submit_eval_jobs.py` in open-instruct, your converted model must be saved in the `oe-adapt-default` weka bucket.

2. Launch evaluations using the submit_eval_jobs.sh script in `open-instruct` using a command such as:

    ```bash
    python scripts/submit_eval_jobs.py \
        --model_name MODEL_NAME_HERE \
        --location /weka/oe-adapt-default/$USER/checkpoints/path-to-model/stepFINAL_STEP-hf \
        --cluster ai2/saturn-cirrascale ai2/neptune-cirrascale \
        --is_tuned \
        --priority high \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --oe_eval_max_length 32768 \
        --workspace tulu-3-results \
        --skip_oi_evals \
        --process_output r1_style
    ```