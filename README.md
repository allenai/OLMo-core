<div align="center">
  <!-- <img src="https://github.com/allenai/OLMo/assets/8812459/774ac485-a535-4768-8f7c-db7be20f5cc3" width="300"/> -->
  <img src="https://allenai.org/olmo/olmo-7b-animation.gif" alt="OLMo Logo" width="600" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
  <h1>OLMo-core</h1>
  <h4>Building blocks for OLMo modeling and training</h4>
</div>
<p align="center">
  <a href="https://github.com/allenai/OLMo/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://github.com/allenai/OLMo/releases">
    <img alt="Docs" src="https://img.shields.io/badge/OLMocore-docs-red">
  </a>
  <a href="https://arxiv.org/pdf/2501.00656.pdf">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2402.00838-blue">
  </a>
  <a href="https://playground.allenai.org">
    <img alt="Playground" src="https://img.shields.io/badge/Ai2-Playground-F0529C">
  </a>
  <a href="https://discord.gg/sZq3jTNVNG">
    <img alt="Discord" src="https://img.shields.io/badge/Discord%20-%20blue?style=flat&logo=discord&label=Ai2&color=%235B65E9">
  </a>
</p>

## Installation

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system and hardware.

For development, we recommend installing from source:

```bash
git clone https://github.com/allenai/OLMo-core.git
cd OLMo-core
pip install -e .[all]
```
Or you can install from PyPI with:

```bash
pip install ai2-olmo-core
```

There are a number of optional dependencies that must be installed to use certain functionality as well, including:
- [flash-attn](https://github.com/Dao-AILab/flash-attention) for flash attention and certain other fused operations.
- [torchao](https://github.com/pytorch/ao) for float8 training.
- [megablocks](https://github.com/databricks/megablocks) for mixture-of-experts (MoE) models.

The published [Docker images](https://github.com/orgs/allenai/packages?repo_name=OLMo-core) contain all core and optional dependencies, and are regularly tested on our in-house H100 clusters.
But there are several things to keep in mind if you intend to use these images:
- They do not come with the OLMo-core package installed, only its dependencies, to accommodate for regular code changes.
- They may not work on your own cluster if you have different hardware or driver/CUDA versions.

If the published images do not work for your use-case for any of the above reasons, you could adapt our [Dockerfile](https://github.com/allenai/OLMo-core/blob/main/src/Dockerfile) to build your own images.

### Steps to reproduce

To reproduce any of the training processes, for distributed training across multiple nodes:

```bash
python src/scripts/train/OLMo2-32B.py launch run_name --launch.num_nodes=N
```
To resume training from a checkpoint:
```bash
python src/scripts/train/OLMo2-32B.py launch continued_training --launch.num_nodes=N --trainer.load_path="path/to/checkpoint" --trainer.load_strategy=if_available
```
To train on a single GPU:
```bash
python src/scripts/train/OLMo2-13B.py train_single {training_name}
```


##### OLMo 7B and 13B models were trained using our previous training infrastructure. All related checkpoints, configs, and scripts for these models (training/fine-tuning) can be found in the [OLMo](https://github.com/allenai/OLMo) repository. Our new 32B model was trained using our updated training infrastructure. While you can also train 7B and 13B models on this new trainer, please note that the released checkpoints and configs for those models use a different format than the new 32B model.

## Official training scripts

Official training scripts for various model sizes can be found in [`src/scripts/train/`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/train).
To see the exact usage for each script, run the script without any arguments.

Throughput numbers from these scripts with various different configuration settings are reported below, measured on a cluster with NVIDIA H100 GPUs.

| Model&nbsp;size | Model&nbsp;arch.&nbsp;&nbsp; | Context&nbsp;length | Precision | Throughput[^1] | Training&nbsp;&nbsp;&nbsp;script | Commandline&nbsp;overrides&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
| :--------: | :--------: | :------------: | :-------: | -----------: | :----------- | :-------- |
| **1B**  | OLMo-1124 | 4096 | BF16 | 55,000 TPS | `OLMo2-1B.py` | |
| | | 4096 | BF16/FP8[^2] | 65,000 TPS | `OLMo2-1B.py` | `--train_module.float8_config.enabled=true` |
| **7B**  | OLMo-1124 | 4096 | BF16 | 10,000 TPS | `OLMo2-7B.py` | |
| | | 4096 | BF16/FP8 | 13,000 TPS | `OLMo2-7B.py` | `--train_module.float8_config.enabled=true` |
| **8B**  | Llama | 4096 | BF16 | 9,500 TPS | `Llama3-8B.py` | |
| | | 4096 | BF16/FP8 | 12,500 TPS | `Llama3-8B.py` | `--train_module.float8_config.enabled=true` |
| **13B** | OLMo-1124 | 4096 | BF16 | 4,600 TPS | `OLMo2-13B.py` | |
| | | 4096 | BF16/FP8 | 5,500 TPS | `OLMo2-13B.py` | `--train_module.float8_config.enabled=true` |
| **32B** | OLMo-1124 | 4096 | BF16 | 1,500 TPS | `OLMo2-32B.py` | |
| | | 4096 | BF16/FP8 | 1,800 TPS | `OLMo2-32B.py` | `--train_module.float8_config.enabled=true` |

[^1]: Throughput reported in tokens per second per device.
[^2]: In this setup most matrix multiplications are computed in `float8`, everything else is in `bfloat16`.

You can find list of all the checkpoints of 32B in [`configs/`](https://github.com/allenai/OLMo-core/tree/main/configs).

## OLMo-2 Model Training

Below is a comprehensive table showing the Stage 2 training details for all OLMo 2 model sizes (7B, 13B, and 32B).

| Model Size | Training | Checkpoint | Training Config | Monitoring |
|------------|----------|------------|-----------------|------------|
| **7B** | random seed 42, 50B tokens | [stage2-ingredient1-step11931-tokens50B](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/stage2-ingredient1-step11931-tokens50B) | [OLMo2-7B-stage2-seed42.yaml](configs/official-1124/OLMo2-7B-stage2-seed42.yaml) | [wandb.ai/OLMo2-7B](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/reports/) |
|  | random seed 42069, 50B tokens | [stage2-ingredient2-step11931-tokens50B](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/stage2-ingredient2-step11931-tokens50B) | [OLMo2-7B-stage2-seed42069.yaml](configs/official-1124/OLMo2-7B-stage2-seed42069.yaml) | [wandb.ai/OLMo2-7B](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/reports/) |
| | random seed 666, 50B tokens | [stage2-ingredient3-step11931-tokens50B](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/stage2-ingredient3-step11931-tokens50B) | [OLMo2-7B-stage2-seed666.yaml](configs/official-1124/OLMo2-7B-stage2-seed666.yaml) | [wandb.ai/OLMo2-7B](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/reports/) |
| | **Final Souped Model** | [main](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/main) | No config, weights averaged in Python | - |
| **13B** | random seed 1110, 100B tokens | [stage2-ingredient1-step11931-tokens100B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient1-step11931-tokens100B) | [OLMo2-13B-stage2-seed1110-100B.yaml](configs/official-1124/OLMo2-13B-stage2-seed1110-100B.yaml) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |
|  | random seed 2662, 100B tokens | [stage2-ingredient2-step11931-tokens100B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient2-step11931-tokens100B) | [OLMo2-13B-stage2-seed2662-100B.yaml](configs/official-1124/OLMo2-13B-stage2-seed2662-100B.yaml) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |
|  | random seed 6209, 100B tokens | [stage2-ingredient3-step11921-tokens100B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient3-step11921-tokens100B) | [OLMo2-13B-stage2-seed6209-100B.yaml](configs/official-1124/OLMo2-13B-stage2-seed6209-100B.yaml) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |
|  | random seed 2662, 300B tokens | [stage2-ingredient4-step11931-tokens300B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient4-step35773-tokens300B) | [OLMo2-13B-stage2-seed2662-300B.yaml](configs/official-1124/OLMo2-13B-stage2-seed2662-300B.yaml) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |
|  | **Final Souped Model** | [main](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/main) | No config, weights averaged in Python | - |
| **32B** | random seed 1110, 100B tokens | [stage2-ingredient1-step11921-tokens100B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage2-ingredient1-step11921-tokens101B) |  | coming soon |
|  | random seed 2662, 100B tokens | [stage2-ingredient2-step11921-tokens100B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage2-ingredient2-step11921-tokens101B) |  | coming soon |
|  | random seed 6209, 100B tokens | [stage2-ingredient3-step11921-tokens100B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient3-step11931-tokens100B) |  | coming soon |
|  | **Final Souped Model** | [main](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/main) | No config, weights averaged in Python | - |

All training configs are set up to download the latest checkpoint after stage 1 and start training from there.

## Inference

You can use our Hugging Face integration to run inference on the OLMo Transformers checkpoints:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0325-32B")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0325-32B")
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# inputs = {k: v.to('cuda') for k,v in inputs.items()} # optional verifying cuda
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

Alternatively, with the Hugging Face pipeline abstraction:

```python
from transformers import pipeline
olmo_pipe = pipeline("text-generation", model="allenai/OLMo-2-0325-32B")
print(olmo_pipe("Language modeling is"))
```
### Quantization

```python
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0325-32B", torch_dtype=torch.float16, load_in_8bit=True)  # requires bitsandbytes
```

## Evaluation
Additional tools for evaluating OLMo models are available at the [OLMo Eval](https://github.com/allenai/OLMo-eval) and [olmes](https://github.com/allenai/olmes) repositories.

## API stability

Even though this library is under rapid development we are trying hard to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with every release except for features that are explicitly marked as beta features. Those features will be tagged like this in the [API docs](https://olmo-core.readthedocs.io/en/latest/):

![image](https://github.com/user-attachments/assets/c666686d-3ae6-4c88-8381-befd698d3fd0)


## Citing

```bibtex
@misc{olmo20242olmo2furious,
      title={2 OLMo 2 Furious}, 
      author={Team OLMo and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Michal Guerquin and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James V. Miranda and Jacob Morrison and Tyler Murray and Crystal Nam and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Christopher Wilhelm and Michael Wilson and Luke Zettlemoyer and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2501.00656},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.00656}, 
}
```
