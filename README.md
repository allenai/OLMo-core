<div align="center">
  <h1>OLMo-core</h1>
  <p>Building blocks for OLMo modeling and training</p>
</div>
<p align="center">
  <a href="https://github.com/allenai/OLMo-core/tree/main/src/examples">Examples</a> ||
  <a href="https://olmo-core.readthedocs.io/en/latest/">Docs</a> ||
  <a href="https://pypi.org/project/ai2-olmo-core/">PyPI</a> ||
  <a href="https://beaker.org/ws/ai2/OLMo-core/images">Beaker Images</a> ||
  <a href="https://github.com/allenai/OLMo-core/blob/main/LICENSE">License</a> ||
  <a href="https://github.com/allenai/OLMo-core/blob/main/CHANGELOG.md">Changelog</a>
</p>

## Installation

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system. Then you can install from PyPI with:

```bash
pip install ai2-olmo-core
```

## Official training scripts

Official training scripts for various model sizes can be found in [`src/scripts/train/`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/train).
To see the exact usage for each script, run the script without any arguments.

Throughput numbers from these scripts with various different configuration settings are reported below, measured on a cluster with NVIDIA H100 GPUs.

| Model&nbsp;size | Context&nbsp;length | Precision | Throughput[^1] | Training&nbsp;script | Commandline&nbsp;overrides&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
| :--------: | :------------: | :-------: | -----------: | :----------- | :-------- |
| **1B**  | 4096 | BF16 | 44,000 TPS | `OLMo-1B.py` | |
| | 256-8192[^2] | BF16 | 49,000 TPS | `OLMo-1B.py` | `--dataset.name=vsl` |
| | 4096 | FP8 | 51,000 TPS | `OLMo-1B.py` | `--model.float8_config.enabled=true` |
| **7B**  | 4096 | BF16 | 10,000 TPS | `OLMo-7B.py` | |
| | | FP8 | 13,000 TPS | `OLMo-7B.py` | `--model.float8_config.enabled=true` |
| **13B** | 4096 | BF16 | 4,600 TPS | `OLMo-13B.py` | |

[^1]: Throughput reported in tokens per second per device.
[^2]: Denotes variable sequence length (VSL) with the Grow-P2 curriculum from [Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum](https://arxiv.org/abs/2405.13226).

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
@article{OLMo,
  title={OLMo: Accelerating the Science of Language Models},
  author={Dirk Groeneveld and Iz Beltagy and Pete Walsh and Akshita Bhagia and Rodney Kinney and Oyvind Tafjord and A. Jha and Hamish Ivison and Ian Magnusson and Yizhong Wang and Shane Arora and David Atkinson and Russell Authur and Khyathi Raghavi Chandu and Arman Cohan and Jennifer Dumas and Yanai Elazar and Yuling Gu and Jack Hessel and Tushar Khot and William Merrill and Jacob Daniel Morrison and Niklas Muennighoff and Aakanksha Naik and Crystal Nam and Matthew E. Peters and Valentina Pyatkin and Abhilasha Ravichander and Dustin Schwenk and Saurabh Shah and Will Smith and Emma Strubell and Nishant Subramani and Mitchell Wortsman and Pradeep Dasigi and Nathan Lambert and Kyle Richardson and Luke Zettlemoyer and Jesse Dodge and Kyle Lo and Luca Soldaini and Noah A. Smith and Hanna Hajishirzi},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267365485},
  journal={arXiv preprint},
}
```
