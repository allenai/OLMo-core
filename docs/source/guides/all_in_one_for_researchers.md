# All-in-one for researchers

This guide is intended for researchers who are new to OLMo-core and would like to experiment with OLMo models or ablate new architectures or data recipes.
We will show you:

- How to launch your first experiment with a small transformer model on Beaker, or locally with `torchrun`.
  This example will be a good foundation to build your own projects on top of.
- How to customize different components of the training loop, such as the model, data loader, optimizer, etc.
- How to troubleshoot common issues.
- How to scale up to larger models while maintaining high MFU.
- Additional topics, such as how to fine-tune from weights HuggingFace.

```{attention}
If you run into any issues with tutorial, don't hesitate to [open an issue on GitHub](https://github.com/allenai/OLMo-core/issues/new/choose) or reach out on Slack in the [#olmo-core-users](https://allenai.slack.com/archives/C08AU86NMCM) channel.
```

## Setup

### Fork, clone, install

For rapid experimentation we recommend forking OLMo-core for your project instead of installing it as a dependency.
So start by [creating a fork](https://github.com/allenai/OLMo-core/fork) if you haven't already, and then cloning your fork to the computer where you'll be doing the development.

Next you should create or activate a Python virtual environment with a Python version of at least 3.10.
We recommend using [uv](https://docs.astral.sh/uv/) for that, but any other virtual environment system will suffice as well, including conda.

Now once you've `cd`-ed into the root directory of your clone of OLMo-core *and* activated your virtual environment, install [PyTorch](https://pytorch.org) according the directions specific to your operating system and hardware (a CPU-only distribution is fine for local development).
And finally, install OLMo-core in editable mode by running

```
pip install -e '.[all]'
```

or an equivalent `uv` command, such as `uv pip install -e '.[all]'` or `uv sync --all-extras`.

### Beaker

If you'll be using [Beaker](https://beaker.allen.ai) to run experiments you should also [install and configure the Beaker CLI](https://beaker-docs.apps.allenai.org/start/install.html),
which will allow the OLMo-core launch module ({mod}`olmo_core.launch.beaker`) to authenticate with Beaker on your behalf.

It's also a good idea to create a dedicated Beaker workspace at this point for your project.
You can do that by running:

```
beaker workspace create --org=ai2 WORKSPACE_NAME
```

You should then set the budget account for that workspace if you know the one you should be using:

```
beaker workspace set-budget ai2/WORKSPACE_NAME ai2/BUDGET_NAME
```

We also recommend making this new workspace your default for now to avoid accidentally using a different one:

```
beaker config set default_workspace ai2/WORKSPACE_NAME
```

We have official Beaker images for OLMo-core that include all dependencies.
The most up-to-date versions are defined in the {class}`~olmo_core.launch.beaker.OLMoCoreBeakerImage` enum,
and a complete list can be found in the [OLMo-core workspace](https://beaker.allen.ai/orgs/ai2/workspaces/OLMo-core/images).

If you need to build a custom image, see the instructions below.

### Docker

We maintain a [Dockerfile](https://github.com/allenai/OLMo-core/blob/main/src/Dockerfile) for building official images with all of OLMo-core's dependencies.
You can build one yourself by running `make docker-image` from the repository root.
See the [Makefile](https://github.com/allenai/OLMo-core/blob/main/Makefile) for all the different build arguments that you can modify.

## Run your first experiment

We'll start by launching a short language model pretraining run with a small transformer (271M params) on a subset of c4.
This will only take a few minutes on as little as 2 NVIDIA 40GB A100s.

We'll be using the script [`src/examples/llama/train.py`](https://github.com/allenai/OLMo-core/blob/main/src/examples/llama/train.py).
But before we actually launch the training run, let's look at how the key components/hyperparameters of the training run are defined.

### Defining a config

Near the top of the script you'll find a custom config dataclass:

```{literalinclude} ../../../src/examples/llama/train.py
:language: py
:start-after: '# docs: start-define-config'
:end-before: '    # docs: end-define-config'
```

The structure of the config class is arbitrary, and creating one isn't strictly necessary to use OLMo-core, but it has several benefits:
1. First, it gives us a good way to keep track of all the hyperparameters of each experiment. Since the config inherits from OLMo-core's {class}`~olmo_core.config.Config` baseclass, it comes with useful methods to serialize it to JSON which, for example, could be uploaded to Weights & Biases or saved to the run's checkpoint directory.
2. Second, due to this line in the script:

   ```{literalinclude} ../../../src/examples/llama/train.py
   :language: py
   :start-after: '    # docs: start-config-merge'
   :end-before: '    # docs: end-config-merge'
   ```

   we can override fields in the config at runtime via command-line options in dot-notation.
   For example, you can add the option `--data_loader.prefetch_factor=4` to update the `prefetch_factor` field within the `data_loader` part of the config.

   ```{tip}
   The script includes a dry-run mode that you can use to validate that your overrides are applied correctly.
   Just add the overrides option `--dry-run` and the script will print out the config and exit without actually training.
   If you pass an invalid overrides option (wrong type or invalid field name) the script will raise an exception.
   ```

### Launching the run

Now that we know how to change settings on the fly we're ready to launch the run.
And in order to get it running as fast as possible we're going to turn off a few features from the command-line that we'd normally want on, such as checkpointing and in-loop evals.
We're also going to tell the trainer to stop at step 100.
So the overrides we'll pass in are:

- `--trainer.callbacks.lm_evaluator.enabled=false` to disable the in-loop perplexity evaluator.
- `--trainer.callbacks.downstream_evaluator.enabled=false` to disable the in-loop downstream task evaluator.
- `--trainer.no_checkpoints` to disable checkpointing.
- `--trainer.hard_stop='{value: 100, unit: steps}'` to stop at step 100.

```{tip}
Notice the value we set for `--trainer.hard_stop` is a JSON/YAML mapping. This will get deserialized into a {class}`~olmo_core.train.Duration` instance.
```

#### Launching on Beaker

For Beaker users, you'll find a separate launch script in the same directory as the training script: [`src/examples/llama/train_launch.py`](https://github.com/allenai/OLMo-core/blob/main/src/examples/llama/train_launch.py).
This is a just a thin CLI wrapper around the `olmo_core.launch.beaker` functionality with some settings specific to the corresponding training script.
This script takes a few of its own command-line options (such as `--gpus`, `--preemptible/--not-preemptible`, `--priority`, and `--weka`) and the rest will be passed on as overrides to the training script.

So, to launch the job on Beaker, run this command from the root of your repository:

```fish
python src/examples/llama/train_launch.py tutorial-run-01 \
  --gpus=2 \
  --not-preemptible \
  --priority=normal \
  --weka=oe-training-default \
  --trainer.callbacks.lm_evaluator.enabled=false \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --trainer.no_checkpoints \
  --trainer.hard_stop='{value: 100, unit: steps}'
```

If the launch is successful it will print a link to the Beaker workload and then stream the logs to your terminal for the duration of the run.

#### Launching locally with torchrun

For non-Beaker users, the script can also be run locally via `torchrun`. Assuming you're using 2 local GPUs, the command would be:

```fish
torchrun --nproc-per-node=2 src/examples/llama/train.py tutorial-run-01 \
  --trainer.callbacks.lm_evaluator.enabled=false \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --trainer.no_checkpoints \
  --trainer.hard_stop='{value: 100, unit: steps}'
```

## Customizing components

Now that you've run your first experiment and have a way to test changes, let's look at how to customize different components of the training loop.

The first thing you should know is that the {class}`~olmo_core.train.Trainer` class is actually a general-purpose trainer, i.e. it can be adapted to pretty much any deep learning task by providing a custom {class}`~olmo_core.train.train_module.TrainModule` as the {data}`~olmo_core.train.Trainer.train_module` argument.

### TrainModule (model and optimizer)

A `TrainModule` abstracts away the model, optimizer, and checkpointing details from the trainer.

The example script we used above made use of the {class}`~olmo_core.train.train_module.TransformerTrainModule` implementation that's designed specifically for training any {class}`olmo_core.nn.transformer.Transformer` type model on text data.
So if that sounds like your use-case, the `TransformerTrainModule` will probably work just fine out-of-the-box for you.
Otherwise you should look at the source code for the {class}`~olmo_core.train.train_module.BasicTrainModule`, as that's a good starting point for writing your own.

### Callbacks

The behavior of the training loop can also be customized through the trainer's rich callback API.
A callback is just a subclass of the base {class}`~olmo_core.train.callbacks.Callback` class, and you can add any number of callbacks to the trainer via the {data}`~olmo_core.train.Trainer.callbacks` argument (a mapping of callback name to callback instance), or by using the {meth}`~olmo_core.train.Trainer.add_callback()` trainer method.

There's a number of helpful callbacks that come with OLMo-core, which you can find in the {mod}`olmo_core.train.callbacks` module, such as the {class}`~olmo_core.train.callbacks.WandBCallback` for logging training metrics to Weights & Biases.

Several of these callbacks are considered mandatory and are automatically added to the trainer unless you provide them on your own. These include:
- a {class}`~olmo_core.train.callbacks.ConsoleLoggerCallback` for logging progress to the terminal,
- a {class}`~olmo_core.train.callbacks.SpeedMonitorCallback` for recording throughput metrics,
- a {class}`~olmo_core.train.callbacks.GarbageCollectorCallback` for manually managing Python garbage collection, and
- a {class}`~olmo_core.train.callbacks.CheckpointerCallback` for periodically writing checkpoints.

### Data loader

Data loading can be customized by providing a custom {class}`~olmo_core.data.data_loader.DataLoaderBase` implementation as the {data}`~olmo_core.train.Trainer.data_loader` argument to the trainer.
The API supports both mapped- (known length) and iterable-style (unknown length) datasets.
It's the user's responsibility to ensure that the data loader is compatible with distributed training if using more than one GPU.
See the [data loading guide](./data_loading.rst) for more details.

## Troubleshooting

Sooner or later you're likely to run into issues with your training runs, especially when adding custom components, so here are a few tips to help you troubleshoot them.

### Low-level kernel errors

Due to the [asynchronous execution](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution) of CUDA kernels, the stack traces that are reported at the Python level when a kernel fails are often misleading.
To get a more informative stacktrace you can force synchronous kernel execution by setting the environment variable `CUDA_LAUNCH_BLOCKING=1`.
The launch script in the example above comes with a flag (`--debug`) that will automatically set this for you in the Beaker job.

### CUDA OOM errors

When you run into CUDA out-of-memory (OOM) errors, the first thing you should try to do is reduce the training micro-batch size (see the {data}`~olmo_core.train.train_module.TransformerTrainModule.rank_microbatch_size` argument to the `TransformerTrainModule` for example).
If that's already as small as it can be, consider other options such as activation checkpointing.
See the [scaling](#scaling) section below for more ideas.

### Poor throughput or MFU

When you're trying to pinpoint a bottleneck in your training loop, it's a good idea to first look at the time spent loading each batch from your data loader,
which is a metric that's logged by the {class}`~olmo_core.train.callbacks.SpeedMonitorCallback`.

If data loading is not the issue, consider using the {class}`~olmo_core.train.callbacks.ProfilerCallback` to get a trace which can be viewed with [Perfetto UI](https://ui.perfetto.dev/).
And see the [scaling](#scaling) section below for more ideas.

### Other bugs

For other bugs unrelated to CUDA, it's always a good idea to try to isolate the code that causes the issue.
If you can reproduce the issue in a small standalone script, it will be much easier to debug and fix.
And if that bug can be reproduced from a single process you could run it with a debugging like [pdb](https://docs.python.org/3/library/pdb.html).

Issues that only manifest in a distributed setting can be harder to debug.
Consider writing a distributed unit test with the {func}`~olmo_core.testing.run_distributed_test` helper function.

## Scaling

Scaling transformers is a complex topic.
A complete guide on the matter is well beyond the scope of this document, so this section will be focused on settings that can be applied to the {class}`~olmo_core.train.train_module.TransformerTrainModule` specifically, as we anticipate most readers will be using that.

But for more in-depth information on the topic we recommend checking out the [Scaling Book](https://jax-ml.github.io/scaling-book).

At the time of writing, the `TransformerTrainModule` supports 3 dimensions of parallelism for dense models, namely data parallelism through FSDP or DDP, tensor parallelism (TP), and context parallelism (CP), as well as expert parallelism (EP) for MoEs.
There's also experimental support for pipeline parallelism (PP) in the {class}`~olmo_core.train.train_module.TransformerPipelineTrainModule`,
but if you follow these general guidelines you should be able to train up to 70B parameter dense models at a reasonable MFU without pipelining.

### Guidelines

- For models with 1B or more parameters you should use FSDP instead of DDP.
  This can be configured by setting the `dp_config` option as follows:
  ```python
  TransformerTrainModule(
      dp_config=TransformerDataParallelConfig(name="fsdp", param_dtype="bfloat16"),
      ...
  )
  ```
  Equivalently you can set the `dp_config` via command-line overrides like this:
  ```
  --train_module.dp_config='{name: fsdp, param_dtype: bfloat16}'
  ```
  Depending on the size of your model, the number of nodes you're training on, and the data center bandwidth, you may also want to try HSDP instead of FSDP:
  ```python
  TransformerTrainModule(
      dp_config=TransformerDataParallelConfig(name="hsdp", param_dtype="bfloat16"),
      ...
  )
  ```
- Don't use TP, CP, or activation checkpointing (AC) unless you get OOMs with a rank micro-batch size of a single instance, and always use the biggest micro-batch size you can fit.
- When you can't fit a single-instance micro-batch, try enabling a minimal amount of activation checkpointing first. A good strategy is to set the `ac_config` option to
  ```python
  TransformerTrainModule(
      ac_config=TransformerActivationCheckpointingConfig(mode="budget", activation_memory_budget=0.90),
      ...
  )
  ```
  and find the highest budget that will fit without crashing or producing memory allocation warnings.
  Equivalently you can set the `ac_config` via command-line overrides like this:
  ```
  --train_module.ac_config='{mode: budget, activation_memory_budget: 0.90}'
  ```
- Always use `torch.compile` (set `compile_model=True`, or `--train_module.compile_model=true` from the command-line). Not only will this make your model run faster, but it typically reduces peak CUDA memory usage as well.

## Additional topics and resources

### Reproducing official OLMo runs

When new OLMo models are published we provide public versions of the training scripts in [`src/scripts/official`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official) which can be launched with `torchrun`. If you have access to Beaker you could also use any of the internal scripts in [`src/scripts/train`](https://github.com/allenai/OLMo-core/tree/main/src/scripts/train), which are updated more often and generally have very good default settings for optimal throughput on Ai2's Beaker clusters.

### Fine-tuning from HuggingFace weights

OLMo-core's `Trainer` can be used for fine-tuning just as well as pretraining.
The only additional steps needed are to convert to the pretrained weights into a format that the `Trainer` expects and then to tell the `Trainer` to load those weights at the beginning of your run.
For an example of the former with HuggingFace models, see [this HF conversion guide](../examples/huggingface.rst).
Add for the latter, you just need to add something like this to your training script prior to the call to `Trainer.fit()`:

```{literalinclude} ../../../src/examples/llama/train.py
:language: py
:start-after: '    # docs: start-load-path'
:end-before: '    # docs: end-load-path'
```
