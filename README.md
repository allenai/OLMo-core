<div align="center">
  <h1>OLMo-core</h1>
</div>
<p align="center">
  <a href="XXX">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="XXX">
    <img alt="GitHub release" src="https://img.shields.io/github/release/allenai/OLMo.svg">
  </a>
  <a href="XXX">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2402.00838-blue">
  </a>
</p>

`OLMo-core` is a repository for training and using OLMo3, AI2's state-of-the-art open language model. It is designed by scientists, for scientists.

This is a one-page recipe to launch a pretraining experiment from scratch, from tokenization to downstream evaluations.

## Installation
Create or activate a Python virtual environment with a Python version ≥ 3.10, then install [PyTorch](https://pytorch.org/).

For development, we recommend installing `OLMo-core` from source:
```bash
git clone https://github.com/allenai/OLMo-core.git
cd OLMo-core
pip install -e .[all]
```

Or you can install `OLMo-core` from PyPI with:
```bash
pip install ai2-olmo-core
```

## Pretraining

We will use the script `src/examples/llm/train.py` to launch our first pretraining run. Official training scripts for released models can be found in `src/scripts/official/`.

### Data download

```bash
aria2c -i links.sh -x 16 -s 16 -j 8 -c --auto-file-renaming=false
```

### Defining a config
Near the top of the script we'll find the config dataclass.

```python
@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    """Model config."""
    dataset: NumpyDatasetConfig
    """Dataset config."""
    data_loader: NumpyDataLoaderConfig
    """Data loader config."""
    trainer: TrainerConfig
    """Trainer config."""
    train_module: TransformerTrainModuleConfig
    """Train module config. Contains settings for optimizer."""
    init_seed: int = 12536
    """Random seed to initialize model weights."""
    load_path: Optional[str] = None
    """Path to load checkpoint from if no checkpoint is found in the save folder.
    Mainly used when you want to fine-tune from a pretrained model."""
    load_trainer_state: bool = False
    """Whether to load the trainer state (including data loader state) when loading from `load_path`.
    This only makes sense when trainer state is available in the checkpoint and you're resuming
    on the same dataset."""
```

To override any fields in the config at runtime, we can simply pass them in as command-line options. For instance, adding `--data_loader.prefetch_factor=4` would update the `prefetch_factor` field within the `data_loader` part of the config. To validate that our overrides are applied correctly, we can print the config without actually launching training using the `--dry-run` flag.

```bash
python src/examples/llm/train.py tutorial-run-01 --dry-run
```
Note that the single positional argument provided, here `tutorial-run-01`, is the name of the run. Now we can try overriding a few config options and verify that the corresponding fields in the printed config have changed.
```bash
python src/examples/llm/train.py tutorial-run-01 --dry-run \
  --data_loader.prefetch_factor=4 \
  --trainer.callbacks.wandb.enabled=true
```

Finally, we can change the model architecture via the `--model-factory` argument. The options for this argument are the various classmethods of [TransformerConfig](https://olmo-core.readthedocs.io/en/latest/nn/transformer.html#olmo_core.nn.transformer.TransformerConfig), which define preset model configurations. By default, `model-factory` is set to `llama2_271M`, which constructs a small transformer with 271M params. Alternatively, you can hardcode the desired config by replacing the following lines

```python
    try:
        factory = getattr(TransformerConfig, opts.model_factory)
    except AttributeError:
        raise ValueError(f"Unknown model factory: {opts.model_factory}")
    model_config = factory(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    )
```

with a particular `TransformerConfig` instance, such as an OLMo2 1B model, as follows.

```python
    model_config = TransformerConfig.olmo2_1B(
        vocab_size=tokenizer_config.padded_vocab_size()
    )
```

To specify a new model config, we recommend creating a new classmethod under `TransformerConfig`. Keep in mind that as you change the model size and architecture you’ll likely want to adjust hyperparameters and performance settings such as the learning rate and micro-batch size (`--train_module.rank_microbatch_size`).

### Launching the run
Now that we know how to change settings on the fly, we're ready to launch the run. For the first run, we'll use overrides to disable the in-loop perplexity evaluator, in-loop downstream task evaluator, checkpoint, and terminate the training at step 100. Assuming you have two GPUs available, the command would be
```bash
torchrun --nproc-per-node=2 src/examples/llm/train.py \
  tutorial-run-01 \
  --save-folder=/tmp/tutorial-run-01 \
  --work-dir=/tmp/dataset-cache \
  --trainer.callbacks.lm_evaluator.enabled=false \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --trainer.no_checkpoints \
  --trainer.hard_stop='{value: 100, unit: steps}'
```
This should take only a few minutes on two NVIDIA 40GB A100s.

### Finetuning pretrained models

This script can be used for finetuning pretrained models as well. To tell `Trainer` to load pretrained weights at the beginning of the run, use the `--load-path` option. You may also need to convert your model into a format that the `Trainer` expects. See this [HF conversion guide](You need to convert the pretrained weights into a format that the Trainer expects. See this HF conversion guide for an example of converting weights from HuggingFace into the right format.) for an example of converting weights from HuggingFace into the right format.

## Providing your own data
To provide our own dataset for pretraining, we will want to use the Dolma Toolkit housed at [allenai/dolma](https://github.com/allenai/dolma), which can be installed with `pip install dolma`. Then, we can access the `dolma tokens` command for tokenizing documents. The tool can be used with any HuggingFace-compatible tokenizer.

### Raw data format

The raw data should in `.jsonl` format, ideally with `gz` or `zst` compression. Each line of the JSONL file is a JSON object representing a single document, with the following format

```
{
    "id": "...",             # MANDATORY: source-specific identifier
    "text": "foo",           # MANDATORY: content of the document
    "source": "...",         # MANDATORY: source of the data, such as peS2o, common-crawl, etc.
    "added": "...",          # OPTIONAL: timestamp ai2 acquired this data
    "created": "..."         # OPTIONAL: timestamp when orig document was created (best-guess if not available)
    "metadata": {...}        # OPTIONAL: source-specific metadata
}
```

Note that the `id` field only needs to be unique within the `source`. For example, having the two documents `{"source": "c4", "id": "123"}` and `{"source": "github", "id": "123"}` would be acceptable.

### Tokenization
Now, we can tokenize our raw data with the following command. Please use `dolma tokens --help` for the full list of parameters accepted by `dolma tokens`.

```bash
dolma tokens \
    --documents "/path/to/documents/*" \
    --destination "/path/to/destination" \
    --tokenizer.name_or_path allenai/dolma2-tokenizer \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --tokenizer.encode_special_tokens \
    --processes $(python3 -c "import multiprocessing; print(multiprocessing.cpu_count())") \
    --dtype uint32
```

### Tokenized data format

The output is in the form of `.npy` files containing concatenated tokenized documents, and a `.csv.gz` file containing all the metadata. The metadata has the following columns and one row for each document:
- `start` (int): The start index of the document/chunk in the `.npy` tokenized file (0-indexed)
- `end` (int): The end index of the document/chunk in the `.npy` tokenized file (0-indexed, exclusive)
- `id` (str): The unique identifier of the original document
- `src` (str): The source file path where the original document came from
- `loc` (int): The line number/location of the document in the original source file (1-indexed)

Now that we have the output data paths, we can pass them into our script by enumerating a list as in `--dataset.paths='["/path/to/data1.npy", "/path/to/data2.npy"]'`, or, by using arbitrary wildcards, as in `--dataset.paths='["/path/to/data/*.npy"]'`.
