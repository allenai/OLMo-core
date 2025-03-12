<div align="center">
  <h1>OLMo-core</h1>
  <p>Building blocks for OLMo modeling and training</p>
</div>
<p align="center">
  <a href="https://github.com/allenai/OLMo-core/tree/main/src/examples">Examples</a> ||
  <a href="https://olmo-core.readthedocs.io/en/latest/">Docs</a> ||
  <a href="https://pypi.org/project/ai2-olmo-core/">PyPI</a> ||
  <a href="https://github.com/orgs/allenai/packages?repo_name=OLMo-core">Docker Images</a> ||
  <a href="https://beaker.org/ws/ai2/OLMo-core/images">Beaker Images</a> ||
  <a href="https://github.com/allenai/OLMo-core/blob/main/LICENSE">License</a> ||
  <a href="https://github.com/allenai/OLMo-core/blob/main/CHANGELOG.md">Changelog</a>
</p>

## Installation

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system and hardware. Then you can install from PyPI with:

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

## API stability

Even though this library is under rapid development we are trying hard to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with every release except for features that are explicitly marked as beta features. Those features will be tagged like this in the [API docs](https://olmo-core.readthedocs.io/en/latest/):

![image](https://github.com/user-attachments/assets/c666686d-3ae6-4c88-8381-befd698d3fd0)

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

## Development

After cloning OLMo-core and setting up a Python virtual environment, install the codebase from source with:

```bash
pip install -e .[all]
```

The Python library source code is located in `src/olmo_core`. The corresponding tests are located in `src/test`. The library docs are located in `docs`. You can build the docs locally with `make docs`.

Code checks:
- We use `pytest` to run tests. You can run all tests with `pytest -v src/test`. You can also point `pytest` at a specific test file to run it individually.
- We use `isort` and `black` for code formatting. Ideally you should integrate these into your editor, but you can also run them manually or configure them with a pre-commit hook. To validate that all files are formatted correctly, run `make style-check`.
- We use `ruff` as our primary linter. You can run it with `make lint-check`.
- We use `mypy` as our type checker. You can run it with `make type-check`.

## Citing

```bibtex
@article{OLMo2,
  title={2 OLMo 2 Furious},
  author={Team OLMo and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Michal Guerquin and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James Validad Miranda and Jacob Daniel Morrison and Tyler C. Murray and Crystal Nam and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Chris Wilhelm and Michael Wilson and Luke S. Zettlemoyer and Ali Farhadi and Noah A. Smith and Hanna Hajishirzi},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:275213098},
  journal={arXiv preprint},
}
```
