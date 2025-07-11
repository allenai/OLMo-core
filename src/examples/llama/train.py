"""
Example of how to train a Llama transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llama/train.py run_name [OVERRIDES...]
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, cast, Union, Sequence, Dict, Any, Optional
import glob
import torch
import traceback
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetBase,
    NumpyDatasetType,
    TokenizerConfig,
    DataCollator,
)
from olmo_core.nn.blt import utils as blt_utils
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

SEQUENCE_LENGTH = 1024

# This will read stream data from the public endpoints by default, but that might be a lot slower
# than reading data locally.
DATA_PATTERN = "/weka/oe-training-default/ai2-llm/preprocessed/dclm/baseline_type_topic_classified_20pct/allenai/dolma2-tokenizer/**/**/part-0*-00000.npy"
DATA_PATHS = sorted(glob.glob(DATA_PATTERN, recursive=True))
DATA_WORK_DIR = "/tmp/dataset-cache"
TOKENIZER = {
    "config": TokenizerConfig.dolma2(),
    "hf_tokenizer": AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B"),
}


# class ByteDatasetWrapper:
#     def __init__(self, dataset: NumpyDatasetBase):
#         self.dataset = dataset

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         item = self.dataset[idx]
#         # Convert input_ids to byte sequences
#         item["input_ids"] = blt_utils.chars_to_bytes(item["input_ids"])
#         return item


# class ByteDatasetConfigWrapper:
#     def __init__(self, dataset_config: NumpyDatasetConfig):
#         self.dataset_config = dataset_config

#     def build(self):
#         return ByteDatasetWrapper(
#             self.dataset_config.build()
#         )

#     def __getattr__(self, name):
#         return getattr(self.dataset_config, name)



# @dataclass
# class ByteTokenizerConfig(TokenizerConfig):
#     special_tokens: list[str] = field(default_factory=lambda: [])

#     @classmethod
#     def from_tokenizer(cls, tokenizer):
#         # all_special_tokens does not contain added special tokens (e.g. <|endofprompt|> for OLMo-2)
#         # this is an attempt to include all of them, but it may not be exhaustive.
#         special_tokens = sorted(set(
#             tokenizer.all_special_tokens
#             + list(tokenizer.get_added_vocab().keys())  # type: ignore
#         ))

#         return cls(
#             vocab_size=tokenizer.vocab_size,
#             special_tokens=special_tokens,
#             # convention: 256 bytes first, then special tokens
#             pad_token_id=256 + special_tokens.index(tokenizer.pad_token),
#             eos_token_id=256 + special_tokens.index(tokenizer.eos_token),
#         )


@dataclass
class DetokenizingDataCollator(DataCollator):
    tokenizer: PreTrainedTokenizerBase = None  # type: ignore
    byte_tokenizer_config: ByteTokenizerConfig = None  # type: ignore

    def __post_init__(self):
        # precompute the byte sequences corresponding to all tokens in the vocabulary
        self.byte_sequences = {}

        for key, value in self.tokenizer.get_vocab().items():
            if key in self.byte_tokenizer_config.special_tokens:
                byte_sequence = [256 + self.byte_tokenizer_config.special_tokens.index(key)]
            else:
                byte_sequence = blt_utils.chars_to_bytes(key)

            assert self.byte_sequences.get(value) is None
            self.byte_sequences[value] = byte_sequence

    def __call__(
        self, items: Union[Sequence[Dict[str, Any]], Sequence[torch.Tensor]]
    ) -> Dict[str, Any]:
        batch_size = len(items)
        max_len = max((len(x["input_ids"] if isinstance(x, dict) else x) for x in items))

        byte_lengths_tensor = torch.zeros((batch_size, max_len), dtype=torch.int32)
        byte_tokens = []

        for example_idx, item in enumerate(items):
            example_byte_tokens = []
            for token_idx, token in enumerate(item["input_ids"] if isinstance(item, dict) else item):
                token_byte_tokens = self.byte_sequences[int(token)]
                byte_lengths_tensor[example_idx, token_idx] = len(token_byte_tokens)
                example_byte_tokens.extend(token_byte_tokens)
            byte_tokens.append(example_byte_tokens)

        max_byte_length = max(len(x) for x in byte_tokens)
        byte_tokens_tensor = torch.full(
            (batch_size, max_byte_length),
            fill_value=int(self.byte_tokenizer_config.pad_token_id),
            dtype=torch.int32
        )
        for example_idx, example_byte_tokens in enumerate(byte_tokens):
            byte_tokens_tensor[example_idx, :len(example_byte_tokens)] = torch.tensor(
                example_byte_tokens, dtype=torch.int32
            )

        attention_mask_tensor = byte_tokens_tensor != self.byte_tokenizer_config.pad_token_id

        return {
            "input_ids": byte_tokens_tensor,
            "byte_lengths": byte_lengths_tensor,
            "attention_mask": attention_mask_tensor,
        }


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    byte_tokenizer_config = ByteTokenizerConfig.from_tokenizer(TOKENIZER["hf_tokenizer"])

    model_config = TransformerConfig.llama2_271M(
        vocab_size=byte_tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    )

    dataset_config = NumpyDatasetConfig(
        paths=DATA_PATHS,
        name=NumpyDatasetType.fsl,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=8192,
        tokenizer=byte_tokenizer_config,
        work_dir=DATA_WORK_DIR,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=16 * SEQUENCE_LENGTH, # DEBUG
        seed=0,
        num_workers=0, # DEBUG
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=16 * SEQUENCE_LENGTH,
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=AdamWConfig(
            lr=1e-3,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=False, # DEBUG
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=100),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"/tmp/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=100,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                project="benjaminm-tok",
                entity="ai2-llm",
                cancel_check_interval=10,
                enabled=True,  # change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=["hellaswag"],
                #  FIXME: should be byte_tokenizer_config but does not work in eval
                #  replace AutoTokenizer with olmo_eval.HFTokenizer?
                tokenizer=TOKENIZER["config"],
                eval_interval=250,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


def main(run_name: str, overrides: List[str]):
    config = build_config(run_name, overrides)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset,
        collator=DetokenizingDataCollator(
            pad_token_id=dataset.pad_token_id,
            tokenizer=TOKENIZER["hf_tokenizer"],
            byte_tokenizer_config=ByteTokenizerConfig.from_tokenizer(TOKENIZER["hf_tokenizer"]),
        ),
        dp_process_group=train_module.dp_process_group
    )
    trainer = config.trainer.build(train_module, data_loader)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    try:
        main(run_name, overrides=overrides)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()
        import ipdb; ipdb.post_mortem()
    finally:
        teardown_training_environment()
