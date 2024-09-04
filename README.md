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

Official training scripts for various model sizes can be found in [`src/scripts/train/`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/train). Throughput numbers are reported below.

| Model size | Context Length | Script | Throughput[^1] | MFU |
| :--------: | :------------: | ------ | -------------: | --- |
| 1B  | 4K | [`OLMo-1B.py`](https://github.com/allenai/OLMo-core/blob/main/src/scripts/train/OLMo-1B.py) | 45-46K TPS | 39-40% |
| 7B  | 4K | [`OLMo-7B.py`](https://github.com/allenai/OLMo-core/blob/main/src/scripts/train/OLMo-7B.py) | 9.7-10K TPS | 47-48% |
| 13B | 4K | [`OLMo-13B.py`](https://github.com/allenai/OLMo-core/blob/main/src/scripts/train/OLMo-13B.py) | 4.4-4.6K TPS | 41-42% |

[^1]: Throughput numbers reported in tokens per second per device, measured on a cluster of H100 GPUs.

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
