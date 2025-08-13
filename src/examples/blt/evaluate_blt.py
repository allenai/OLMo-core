import logging
import os
import sys
from dataclasses import dataclass
from typing import List, cast
from types import SimpleNamespace
import glob
import traceback
from pathlib import Path
import copy
from tqdm.auto import tqdm
import json

import torch

from olmo_core.train.callbacks.evaluator_callback import DownstreamEvaluator
from olmo_core.distributed.parallel import build_world_mesh
from olmo_core.utils import get_default_device
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.blt_train_module import TransformerBLTTrainModule
from olmo_core.train.train_module import EvalBatchSpec
from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    NumpyByteFSLDataset,
    ByteTokenizerConfig,
    TokenizerConfig,
    ByteDataCollator,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerType,
    TransformerBlockConfig,
    TransformerBlockType,
)
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.mamba import MambaConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.blt.config import LocalEncoderConfig, LocalDecoderConfig
from olmo_core.optim import AdamWConfig, LinearWithWarmup, OptimGroupOverride
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
from olmo_core.nn.blt.config import BLTConfig
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

NUM_WORKERS = 16
BYTE_EXPANSION_FACTOR = int(os.environ.get("BYTE_EXPANSION_FACTOR", "6"))
SEQUENCE_LENGTH = 1024
EVAL_BATCH_SIZE = 4
EVAL_TASKS = [
    "arc_challenge_test_rc_5shot",
    "arc_easy_test_rc_5shot",
    "hellaswag_rc_5shot",  # 1K subset of HellaSwag
    "winogrande_val_rc_5shot",  # Helpful after 750M-5xC scale
    "csqa_val_rc_5shot",
    "piqa_val_rc_5shot",
    "mmlu_stem_test_rc_5shot",
    "mmlu_humanities_test_rc_5shot",
    "mmlu_social_sciences_test_rc_5shot",
    "mmlu_other_test_rc_5shot",
    "basic_skills_string_operations_rc_5shot",
    "basic_skills_pattern_rc_5shot",
    "basic_skills_logical_reasoning_rc_5shot",
    "basic_skills_common_knowledge_rc_5shot",
    "basic_skills_coding_rc_5shot",
    "basic_skills_arithmetic_rc_5shot",
]
# BLT_ARCH = "blt_7b"
# BLT_CKPT_PATH = "/weka/oe-training-default/benjaminm/checkpoints/blt_7b/model_and_optim"
# BLT_CONFIG_PATH = "../blt/hf-weights/blt_7b/params.json"
BLT_ARCH = "blt_1b"
BLT_CKPT_PATH = "/weka/oe-training-default/benjaminm/checkpoints/blt_1b/model_and_optim"
BLT_CONFIG_PATH = "../blt/hf-weights/blt_1b/params.json"
BLT_ENTROPY_CKPT_PATH = "../blt/hf-weights/entropy_model"
OUTPUT = "../blt/hf-weights/blt_1b/metrics.json"

log = logging.getLogger(__name__)

def _load_patcher():
    from bytelatent.data.file_util import get_fs
    from bytelatent.args import TrainArgs

    fs = get_fs(BLT_CONFIG_PATH)
    train_args = TrainArgs.model_validate_json(fs.read_text(BLT_CONFIG_PATH))
    patcher_args = train_args.data.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = True
    patcher_args.entropy_model_checkpoint_dir = BLT_ENTROPY_CKPT_PATH

    return patcher_args.build()


# prepare batch similar to TransformerBLTTrainModule._prepare_batch
def _prepare_batch(batch, tokenizer, patcher):
    train_module_duck = cast(TransformerBLTTrainModule, SimpleNamespace(tokenizer=tokenizer, blt_config=None))
    input_ids, labels, batch = TransformerBLTTrainModule._prepare_batch(train_module_duck, copy.deepcopy(batch))
    patch_lens, _ = patcher.patch(input_ids)
    batch["patch_lens"] = patch_lens

    return input_ids, labels, batch

def main(run_name: str, overrides: List[str]):
    seed_all(0)

    tokenizer = ByteTokenizerConfig.blt().build()
    model = getattr(TransformerConfig, BLT_ARCH)().build(init_device="meta")
    model = parallelize_model(
        model,
        world_mesh=build_world_mesh(),
        device=get_default_device(),
        max_sequence_length=SEQUENCE_LENGTH * BYTE_EXPANSION_FACTOR,
        rank_microbatch_size=EVAL_BATCH_SIZE,
        compile_model=True,
    )

    incompatible_keys = load_model_and_optim_state(BLT_CKPT_PATH, model)
    log.info(f"Incompatible keys after loading BLT checkpoint: {incompatible_keys}")

    model.eval()

    patcher = _load_patcher()
    all_metrics = {}

    for eval_task in EVAL_TASKS:
        evaluator = DownstreamEvaluator(
            name="downstream",
            task=eval_task,
            batch_spec=EvalBatchSpec(
                rank_batch_size=EVAL_BATCH_SIZE * SEQUENCE_LENGTH,
            ),
            tokenizer=tokenizer.hf_tokenizer,
            device=model.device,
        )

        i = 0

        for batch in tqdm(iter(evaluator), total=evaluator.total_batches):
            input_ids, labels, batch = _prepare_batch(batch, tokenizer, patcher)
            with torch.no_grad():
                logits, _, ce_loss, _ = model(input_ids, labels=labels, **batch)

            device_batch = {k: v.to(model.device) if v is not None else None for k, v in batch.items()}
            evaluator.update_metrics(device_batch, ce_loss, logits)

            i += 1

        metrics = evaluator.compute_metrics()
        print(metrics)
        all_metrics.update({k: v.item() for k, v in metrics.items()})

    json.dump(all_metrics, open(OUTPUT, "w"), indent=2)    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    try:
        main(run_name, overrides=overrides)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        traceback.print_exc()
        if get_rank() == 0:
            import ipdb; ipdb.post_mortem()
    finally:
        teardown_training_environment()
