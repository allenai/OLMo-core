# Olmo-3 Model Card

We introduce Olmo 3, a new family of 7B and 32B models featuring a [TODO: insert gain in performance], among other evaluation improvements, compared to the most recent Olmo 2 7B model. These gains come from training on dolma3-mix-1025 and dolma3-dolmino-mix-1025 datasets and staged training approach.

Olmo is a series of Open language models designed to enable the science of language models. These models are trained on the Dolma 3 dataset. We are releasing all code, checkpoints, logs (coming soon), and associated training details.

| Size | Training Tokens | Layers | Hidden Size | Attention Heads | Context Length |
|------|-----------------|--------|-------------|-----------------|----------------|
| [Olmo 3 7B](https://huggingface.co/allenai/Olmo-3-1025-7B) | 5.93 Trillion | 32 | 4096 | 32 | 8192 |
| [Olmo 3 32B](https://huggingface.co/allenai/Olmo-3-1125-32B) | 5.50 Trillion | 64 | 5120 | 40 | 8192 |



The core models released in this batch include the following:

| Stage | [Olmo 3 7B Think] | [Olmo 3 32B Think] | [Olmo 3 7B Instruct] | [Olmo 3 32B Instruct] |
|-------|-------------------|--------------------|----------------------|-----------------------|
| Base Model | [Olmo-3-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) | [Olmo-3-32B](https://huggingface.co/allenai/Olmo-3-1125-32B) |  |  |
| SFT | [Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) | [Olmo-3-32B-Think-SFT](https://huggingface.co/allenai/Olmo-3-32B-Think-SFT) | [Olmo-3-7B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT) | [Olmo-3-32B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-32B-Instruct-SFT) |
| DPO | [Olmo-3-7B-Think-DPO](https://huggingface.co/allenai/Olmo-3-7B-Think-DPO) | [Olmo-3-32B-Think-DPO](https://huggingface.co/allenai/Olmo-3-32B-Think-DPO) | [Olmo-3-7B-Instruct-DPO](https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO) | [Olmo-3-32B-Instruct-DPO](https://huggingface.co/allenai/Olmo-3-32B-Instruct-DPO) |
| Final Models (RLVR) | [Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think) | [Olmo-3-32B-Think](https://huggingface.co/allenai/Olmo-3-32B-Think) | [Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | [Olmo-3-32B-Instruct](https://huggingface.co/allenai/Olmo-3-32B-Instruct) |

## Olmo-3 7B Model Training

OLMo-3 32B pretraining follows a two-stage training procedure.
In the first stage, we train on large amounts of mostly web-based data: [OLMo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124).
In the second stage, we train on a smaller amount of high-quality, targeted data: Dolmino-mix-0324.

| Stage | Model Size | Tokens  | Script | Checkpoint | Monitoring |
|-------|------------|-----------|--------|------------|------------|
| stage 1 (pretraining) | **7B** | 5.93 Trillion | |[stage1-step721901-tokens6056B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage1-step721901-tokens6056B) | [comet.ml/OLMo2-32B](https://www.comet.com/ai2/olmo-2-0325-32b/reports/olmo-2-0325-32b?shareable=WhT37Wy7jqttDoy6ysDBumQzf) |
| stage 2 (midtraining) | **7B** | 100 Billion | | [stage2-ingredient1-step11921-tokens101B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage2-ingredient1-step11921-tokens101B) | [comet.ml/OLMo2-32B](https://www.comet.com/ai2/olmo-2-0325-32b/reports/olmo-2-0325-32b-anneal?shareable=WhT37Wy7jqttDoy6ysDBumQzf) |
| stage 3 (long-context) | **7B** | 50 Billion | |[stage2-ingredient2-step11921-tokens101B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage2-ingredient2-step11921-tokens101B) | [comet.ml/OLMo2-32B](https://www.comet.com/ai2/olmo-2-0325-32b/reports/olmo-2-0325-32b-anneal?shareable=WhT37Wy7jqttDoy6ysDBumQzf)

The table below lists the checkpoints for Stage 1 and Stage 2 of OLMo-2, along with their corresponding Hugging Face format.

| Variant | OLMo Format (Stage 1) | OLMo Format (Stage 2) | Hugging Face Format |
|---------|-----------------------|-----------------------|---------------------|
| **OLMo-2 32B**  | [OLMo-2 32B](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo-2-0325-32B.csv)     | [OLMo-2 32B](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo-2-0325-32B-stage2.csv)      | [Hugging Face for the 32B variant](https://huggingface.co/allenai/OLMo-2-0325-32B)  |

> Note: OLMo-2 7B and 13B models were trained using [the old OLMo trainer](https://github.com/allenai/OLMo). All related checkpoints, configs, and scripts for these models can be found there. While you can train 7B and 13B models with this trainer, please note that the configs and script in the old training codebase are not compatible with this repo.

## Olmo-3 32B