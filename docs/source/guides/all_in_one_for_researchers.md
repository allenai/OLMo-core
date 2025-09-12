# All-in-one for researchers

This guide is intended for researchers who are new to OLMo-core and would like to experiment with OLMo models or ablate new architectures or data recipes.
We will show you:

- How to launch your first experiment with a small transformer model on Beaker, or locally with `torchrun`.
- How to customize different components of the training loop, such as the model, data loader, optimizer, etc.
- How to troubleshoot common issues.

## Setup

For rapid experimentation we recommend forking OLMo-core for your project instead of installing it as a dependency.
So start by [creating a fork](https://github.com/allenai/OLMo-core/fork) if you haven't already, and then cloning your fork to the computer where you'll be doing the development.

Next you should create or activate a Python virtual environment with a Python version of at least 3.10.
We recommend using [uv](https://docs.astral.sh/uv/) for that, but any other virtual environment system will suffice as well, including conda.

Now once you've `cd`-ed into the root directory of your clone of OLMo-core *and* activated your virtual environment, install [PyTorch](https://pytorch.org) according the directions for your environment and hardware (a CPU-only distribution is fine for development).
