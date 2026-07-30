"""olmo-core full fine-tune of Qwen3-4B on NQ-k50 retrieval (8k context).

Loads the converted Qwen3-4B-Base olmo-core checkpoint, SFTs on the pre-tokenized NQ data
(NumpyPaddedFSLDataset + label mask = loss only on the 'Relevant Document: [id]' completion),
and saves a checkpoint to export->HF for eval.

ARM (env or --arm) selects precision:
  baseline : fp32 master + fp32 optimizer moments (bf16 compute)  -- the tested OLMo recipe
  bf16     : bf16 master (model dtype=bf16) + bf16 optimizer moments  -- "full bf16", no SR

Launch with torchrun (see jobs/olmo_nq_sft.sh).
"""

import argparse
import json
import os

import rich

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.optim import CosWithWarmup, SkipStepAdamWConfig

from corpus_reasoning.lib.olmo_models import build_transformer_config, resolve_olmo_model
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

def build(opts):
    bf16 = opts.arm == "bf16"

    spec = resolve_olmo_model(opts.base_model)
    # load_path MUST point at the model_and_optim subdir, NOT the parent. The trainer's
    # checkpoint gate (Checkpointer.dir_is_checkpoint) only recognizes a dir as a "model-only"
    # checkpoint if `<dir>/.metadata` exists at its root. The converter writes that marker inside
    # `<ckpt>/model_and_optim/.metadata`, so pointing load_path at the parent makes the gate
    # return False and the trainer silently trains FROM SCRATCH (12.5 == ln(vocab) loss).
    converted_ckpt = opts.converted_ckpt or f"/scratch/users/prasann/olmo_ckpts/{spec.builder}_olmo"
    load_path = f"{converted_ckpt}/model_and_optim"
    slug = spec.builder.lower()  # e.g. qwen3_0_6b
    if opts.save_folder is None:
        opts.save_folder = f"/scratch/users/prasann/olmo_ckpts/nq_sft_{slug}_{opts.arm}"

    model_config = build_transformer_config(spec, dtype=DType.bfloat16 if bf16 else None)
    # Fused linear cross-entropy (liger): never materialize the full [tokens, vocab] logits
    # tensor — that was the big OOM at microbatch>1 with Qwen3's large vocab.
    model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    # Use the eos/pad ids the data was actually written with (sidecar from the tokenizer).
    # bos_token_id=None -> documents are segmented purely on eos (matches our EOS-terminated docs).
    meta = json.load(open(opts.tokens.replace("_tokens.npy", "_meta.json")))
    tok_cfg = TokenizerConfig(
        vocab_size=spec.vocab_size,
        eos_token_id=meta["eos_token_id"],
        pad_token_id=meta["pad_token_id"],
        bos_token_id=None,
        identifier=opts.base_model,
    )

    dataset_config = NumpyPaddedFSLDatasetConfig(
        paths=[opts.tokens],
        label_mask_paths=[opts.label_mask],
        sequence_length=opts.seq_len,
        tokenizer=tok_cfg,
        dtype=NumpyDatasetDType.uint32,   # Qwen vocab > 65535
        work_dir=opts.work_dir,
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.seq_len * opts.batch_examples,  # tokens
        seed=0, num_workers=2,
    )

    optim = SkipStepAdamWConfig(lr=opts.lr, **({"dtype": DType.bfloat16} if bf16 else {}))

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.seq_len * opts.microbatch_examples,  # examples per fwd per rank
        max_sequence_length=opts.seq_len,
        optim=optim,
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=(None if bf16 else DType.bfloat16),  # bf16 arm: model is already bf16
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=opts.warmup),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            load_path=load_path,             # pretrained Qwen3 weights (model_and_optim subdir)
            load_trainer_state=False,        # weights only -- fresh optimizer/data state
            load_optim_state=False,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.epochs(opts.epochs),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                enabled=True,
                project="corpus-reasoning",
                name=f"olmo_nq_k50_{slug}_{opts.arm}",
                tags=["olmo-core", slug, "nq", "8k", opts.arm],
            ),
        )
        # pre_train_checkpoint=False: skip the wasteful ~48GB step-0 save (it fires because
        # load_path doesn't set checkpoint_loaded). The final checkpoint is still saved by
        # post_train() at the end -- that's what we export for eval.
        .with_callback(
            "checkpointer",
            CheckpointerCallback(save_interval=100000, pre_train_checkpoint=False, save_async=False),
        )
    )
    return model_config, dataset_config, data_loader_config, train_module_config, trainer_config


def main():
    raise NotImplementedError(
        "olmo_nq_sft.py is DEPRECATED — superseded by the config-driven general path. "
        "Use: python scripts/train/train.py configs/<task>.yml  (set `backend: olmo-core`), "
        "or sbatch jobs/olmo_train.sh configs/<task>.yml. The general trainer is "
        "scripts/train/olmo_train.py. This script is kept only for reference."
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "bf16"], default=os.environ.get("ARM", "baseline"))
    ap.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B-Base"))
    ap.add_argument("--converted-ckpt", default=None, help="olmo ckpt dir; default /scratch/.../<builder>_olmo")
    ap.add_argument("--tokens", default="data/olmo_nq/nq_k50_cot_train_tokens.npy")
    ap.add_argument("--label-mask", default="data/olmo_nq/nq_k50_cot_train_label_mask.npy")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--batch-examples", type=int, default=2)        # global examples per optimizer step
    ap.add_argument("--microbatch-examples", type=int, default=1)   # examples per forward per GPU (grad accum = batch/(micro*dp))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--save-folder", default=None)
    ap.add_argument("--work-dir", default="/tmp/olmo_nq_cache")
    ap.add_argument("--dry-run", action="store_true")
    opts = ap.parse_args()

    mc, dc, dlc, tmc, tc = build(opts)
    if get_rank() == 0:
        rich.print({"arm": opts.arm, "epochs": opts.epochs, "lr": opts.lr,
                    "batch_examples": opts.batch_examples, "save_folder": opts.save_folder})

    if opts.dry_run:
        ds = dc.build()
        ds.prepare()  # builds the instance-index cache (the data loader does this at train time)
        if get_rank() == 0:
            print(f"[dry-run] dataset built OK: {len(ds)} examples, seq_len={opts.seq_len}")
        return

    prepare_training_environment()
    try:
        seed_all(12536)
        model = mc.build(init_device="meta")
        train_module = tmc.build(model)
        dataset = dc.build()
        data_loader = dlc.build(dataset, dp_process_group=train_module.dp_process_group)
        trainer = tc.build(train_module, data_loader)
        trainer.fit()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
