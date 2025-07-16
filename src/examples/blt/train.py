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
import math
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from torch.nn import functional as F

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    ByteTokenizerConfig,
    TokenizerConfig,
    DataCollator,
    PaddingDirection,
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
from olmo_core.train.common import LoadStrategy
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

SEQUENCE_LENGTH = 1024
BYTE_EXPANSION_FACTOR = 8  # how much longer the byte sequence is compared to subword sequence

# DEBUG: replaced 0* with 00
DATA_PATTERN = "/weka/oe-training-default/ai2-llm/preprocessed/dclm/baseline_type_topic_classified_20pct/allenai/dolma2-tokenizer/**/**/part-00-00000.npy"
DATA_PATHS = sorted(glob.glob(DATA_PATTERN, recursive=True))
DATA_WORK_DIR = "/tmp/dataset-cache"


@dataclass
class ByteDataCollator(DataCollator):
    def __call__(
        self, items: Union[Sequence[Dict[str, Any]], Sequence[torch.Tensor]]
    ) -> Dict[str, Any]:
        batch = super().__call__(items)

        all_original_input_ids = []
        all_patch_lengths = []

        max_original_len = max(
            len(x["original_input_ids"]) if isinstance(x, dict) and "original_input_ids" in x else math.nan for x in items
        )

        for x in items:
            original_input_ids = x.get("original_input_ids") if isinstance(x, dict) else None
            patch_lengths = x.get("patch_lens") if isinstance(x, dict) else None

            if original_input_ids is not None:
                # both or neither
                assert patch_lengths is not None

                max_original_len = int(max_original_len)

                pad_shape = (
                    (max_original_len - len(original_input_ids), 0)
                    if self.pad_direction == PaddingDirection.left
                    else (0, max_original_len - len(original_input_ids))
                )

                # Pad input IDs.
                all_original_input_ids.append(
                    F.pad(
                        original_input_ids.to(dtype=torch.long),
                        pad_shape,
                        value=self.pad_token_id,
                    )
                )
                # Pad byte lengths.
                all_patch_lengths.append(
                    F.pad(
                        patch_lengths.to(dtype=torch.long),
                        pad_shape,
                        value=0,
                    )
                )

        if all_original_input_ids:
            batch["original_input_ids"] = torch.stack(all_original_input_ids, dim=0)
        if all_patch_lengths:
            batch["patch_lens"] = torch.stack(all_patch_lengths, dim=0)

        return batch


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    tokenizer_config = ByteTokenizerConfig.blt()

    model_config = TransformerConfig.blt_1b(
        vocab_size=260#tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    )
    # save on hash embeddings for now to reduce gpu memory
    model_config = model_config.replace(
        local_encoder=model_config.local_encoder.replace(  # type: ignore
            hash_byte_group_size=[3],
            hash_byte_group_nb_functions=1,
        )
    )

    dataset_config = NumpyDatasetConfig(
        paths=DATA_PATHS,
        name=NumpyDatasetType.byte_fsl,
        sequence_length=SEQUENCE_LENGTH, # subword sequence length
        max_sequence_length=SEQUENCE_LENGTH * BYTE_EXPANSION_FACTOR, # max. length of the byte sequence
        tokenizer=tokenizer_config,
        work_dir=DATA_WORK_DIR,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=4 * SEQUENCE_LENGTH * BYTE_EXPANSION_FACTOR, # DEBUG (bs was 256)
        seed=0,
        num_workers=0, # DEBUG
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=1 * SEQUENCE_LENGTH * BYTE_EXPANSION_FACTOR,
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=AdamWConfig(
            lr=1e-3,
            group_overrides=[
                OptimGroupOverride(params=["local_encoder.embedding.weight", "local_encoder.hash_embeddings.*.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
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
            load_strategy=LoadStrategy.never,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(10), # DEBUG
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                pre_train_checkpoint=False,
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
        #  FIXME: make byte tokenizer work for eval
        # .with_callback(
        #     "downstream_evaluator",
        #     DownstreamEvaluatorCallbackConfig(
        #         tasks=["hellaswag"],
        #         tokenizer=tokenizer_config,
        #         eval_interval=250,
        #     ),
        # )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,  # type: ignore
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
        collator=ByteDataCollator(pad_token_id=dataset.pad_token_id),
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
