<div align="center">
  <!-- <img src="https://github.com/allenai/OLMo/assets/8812459/774ac485-a535-4768-8f7c-db7be20f5cc3" width="300"/> -->
  <img src="https://allenai.org/olmo/olmo-7b-animation.gif" alt="OLMo Logo" width="600" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
  <h1>OLMo-core</h1>
  <h4>Building blocks for OLMo modeling and training</h4>
</div>
<p align="center">
  <a href="https://github.com/allenai/OLMo-core/tree/main/src/examples">
    <img alt="Examples" src="https://img.shields.io/badge/Train-Examples-994B00">
  </a>
  <a href="https://github.com/allenai/OLMo-core/releases/tag/v1.9.0">
    <img alt="Pypi" src="https://img.shields.io/pypi/v/ai2-olmo-core.svg">
  </a>  
  <a href="https://github.com/allenai/OLMo-core/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://olmo-core.readthedocs.io/en/latest/">
    <img alt="Docs" src="https://img.shields.io/badge/OLMocore-docs-red">
  </a>
  <a href="https://arxiv.org/pdf/2501.00656.pdf">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2402.00838-orange">
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
- [flash-attn](https://github.com/Dao-AILab/flash-attention) and [ring-flash-attn](https://github.com/zhuzilin/ring-flash-attention) for intra-document masking and context parallelism.
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) for a low-memory "fused-linear" loss implementation.
- [torchao](https://github.com/pytorch/ao) for float8 training.
- [grouped_gemm](https://github.com/tgale96/grouped_gemm) for dropless mixture-of-experts (MoE) models. You may need to compile from source until [PR #21](https://github.com/tgale96/grouped_gemm/pull/21) is released (post v0.1.6).

The published [Docker images](https://github.com/orgs/allenai/packages?repo_name=OLMo-core) contain all core and optional dependencies, and are regularly tested on our in-house H100 clusters.
But there are several things to keep in mind if you intend to use these images:
- They do not come with the OLMo-core package installed, only its dependencies, to accommodate for regular code changes.
- They may not work on your own cluster if you have different hardware or driver/CUDA versions.

If the published images do not work for your use-case for any of the above reasons, you could adapt our [Dockerfile](https://github.com/allenai/OLMo-core/blob/main/src/Dockerfile) to build your own images.

## Official training scripts

Official training scripts for released models can be found in [`src/scripts/official/`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official).
These scripts are meant to be launched with ``torchrun``. For example:

```bash
torchrun --nproc-per-node=8 ./src/scripts/official/OLMo2-0325-32B-train.py run01
```

You can override most configuration options from the command-line. For example, to override the learning rate you could launch the script like this:

```bash
torchrun --nproc-per-node=8 ./src/scripts/train/OLMo2-0325-32B-train.py run01 --train_module.optim.lr=6e-3
```

You can find list of all the checkpoints of 32B in [`src/scripts/train/official-0325/`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/train/official-0325).

#### OLMo 7B and 13B models were trained using our previous training infrastructure. All related checkpoints, configs, and scripts for these models (training/fine-tuning) can be found in the [OLMo](https://github.com/allenai/OLMo) repository. Our new 32B model was trained using our updated training infrastructure. While you can also train 7B and 13B models on this new trainer, please note that the released checkpoints and configs for those models use a different format than the new 32B model.

## OLMo-2 Model Training

Below is a comprehensive table showing the Stage 2 training details for all OLMo 2 32B.

| Model Size | Training | Checkpoint | Monitoring |
|------------|----------|------------|------------|
| **32B** | random seed 1110, 100B tokens | [stage2-ingredient1-step11921-tokens100B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage2-ingredient1-step11921-tokens101B) | coming soon |
|  | random seed 2662, 100B tokens | [stage2-ingredient2-step11921-tokens100B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage2-ingredient2-step11921-tokens101B) | coming soon |
|  | random seed 6209, 100B tokens | [stage2-ingredient3-step11921-tokens100B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient3-step11931-tokens100B) | coming soon |
|  | **Final Souped Model** | [main](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/main) | No config, weights averaged in Python | - |

Note: You can find all the configs and checkpoints for 7B and 13B in the [OLMo](https://github.com/allenai/OLMo) repository.
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
