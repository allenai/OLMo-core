"""
LOCAL (torchrun, no Beaker/weka) DENSE variant of the Qwen3-0.6B contradiction-n20 SFT.

Sibling of ``Qwen3-0.6B-fast-landmark-contradiction-n20-SFT-local.py`` but with standard dense
attention (no landmark tokens): the model is ``qwen3_0_6B`` with ``fast_landmark=False`` and the data
is fed straight through ``PadToLengthInstanceSource`` (no ``LandmarkInstanceSource`` wrapper -- the
raw n20 npy shards are landmark-agnostic; landmark tokens are only inserted at load time by the
landmark source). Because dense attention never touches the landmark token embedding, the RAW
Qwen3-0.6B base is a valid init here (unlike the landmark variants, which want a landmark-pretrained
base). This makes it a quick, honest "can a 0.6B learn contradiction n20 no-cot at all" baseline.

Run::

    PYTHONPATH=.../OLMo-core/src torchrun --nproc_per_node=8 \\
      src/scripts/train/attn_explore/Qwen3-0.6B-dense-contradiction-n20-SFT-local.py \\
      --run-name q06b-dense-contra-n20-sft-local
"""

import argparse
import json
from dataclasses import replace
from datetime import datetime

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    PadToLengthInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
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
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

SEQUENCE_LENGTH = 2048

# ---- LOCAL paths (shared NFS) ----
DATA_PATH = "/scratch/users/prasann/longctx_sft_qwen/contradiction_n20"
# RAW Qwen3-0.6B base distcp (dense is landmark-agnostic, so this is a valid init). Lives on mooney
# node-local /data -> this launcher is mooney-pinned. Override with --base-checkpoint.
BASE_CHECKPOINT = "/data/prasann/olmo_ckpts/qwen3_0_6B_base_olmo/model_and_optim"
SAVE_ROOT = "/scratch/users/prasann/olmo_ckpts"
WORK_DIR = "/scratch/users/prasann/longctx_sft_qwen/dataset-cache"

GLOBAL_BATCH_SIZE = SEQUENCE_LENGTH * 8
LR = 5e-5
NUM_EPOCHS = 3


def build_and_fit(opts: argparse.Namespace) -> None:
    run_name = opts.run_name
    run_name_with_ts = f"{run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    save_folder = opts.save_folder or f"{SAVE_ROOT}/{run_name}"

    tokenizer_config = TokenizerConfig.qwen3()
    # Shards are separated by single EOS tokens; qwen3 ties bos==eos, so drop BOS for segmentation.
    doc_tokenizer_config = replace(tokenizer_config, bos_token_id=None)

    seq_len = opts.seq_len
    global_batch_size = seq_len * 8
    model_factory = {
        "0.6B": TransformerConfig.qwen3_0_6B,
        "4B": TransformerConfig.qwen3_4B,
    }[opts.model_size]
    model_config = model_factory(
        vocab_size=tokenizer_config.padded_vocab_size(),
        # qwen3 factories default backend=None -> TorchAttentionBackend, which can't KV-cache during
        # eval decode (eval_lc_native.py generates with use_cache=True). Pin flash_2 so the saved
        # config.json supports KV-cached generation (matches the docchunk-mix script).
        attn_backend=AttentionBackendName.flash_2,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=seq_len,
        max_sequence_length=seq_len,
        optim=SkipStepAdamWConfig(
            lr=opts.lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=LinearWithWarmup(warmup_fraction=0.03, alpha_f=0.0),
        compile_model=opts.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            shard_degree=1,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )

    instance_source_config = PadToLengthInstanceSourceConfig.from_npy(
        f"{opts.data_path}/token_ids_part_*.npy",
        tokenizer=doc_tokenizer_config,
        sequence_length=seq_len,
        label_mask_paths=[f"{opts.data_path}/labels_mask_*.npy"],
        expand_glob=True,
    )

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir,
        global_batch_size=global_batch_size,
        seed=34521,
        num_workers=opts.num_workers,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            load_path=opts.base_checkpoint,
            load_strategy=LoadStrategy.always,
            load_trainer_state=False,
            load_optim_state=False,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.epochs(opts.epochs),
            hard_stop=Duration.steps(opts.max_steps) if opts.max_steps else None,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=250,
                max_checkpoints=3,
                save_async=True,
            ),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
    )
    if opts.wandb:
        trainer_config = trainer_config.with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=run_name,
                entity=opts.wandb_entity,
                project="memory-networks",
                enabled=True,
                cancel_check_interval=10,
            ),
        )

    seed_all(12536)
    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)
    source = instance_source_config.build(data_loader_config.work_dir)
    data_loader = data_loader_config.build(
        source, dp_process_group=train_module.dp_process_group
    )
    trainer = trainer_config.build(train_module, data_loader)
    trainer.fit()

    if opts.save_checkpoint:
        from olmo_core.distributed.checkpoint import save_model_and_optim_state

        model_dir = f"{save_folder}/model_and_optim"
        save_model_and_optim_state(model_dir, train_module.model, save_overwrite=True)
        if get_rank() == 0:
            experiment = {
                "model": model_config.as_config_dict(),
                "dataset": {"tokenizer": tokenizer_config.as_config_dict()},
            }
            with open(f"{save_folder}/config.json", "w") as f:
                json.dump(experiment, f)
            print(f"[dense-contra] saved model-only checkpoint -> {save_folder}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="q06b-dense-contra-n20-sft-local")
    ap.add_argument("--save-folder", default=None, help=f"default {SAVE_ROOT}/<run-name>")
    ap.add_argument("--base-checkpoint", default=BASE_CHECKPOINT,
                    help="olmo distcp model_and_optim subdir to init from")
    ap.add_argument("--data-path", default=DATA_PATH,
                    help="dir with token_ids_part_*.npy + labels_mask_*.npy shards")
    ap.add_argument("--work-dir", default=WORK_DIR,
                    help="dataset-cache dir; use a FRESH node-local dir to avoid stale NFS locks")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="data loader workers (0 = load in-process; simplest for tiny smoke data)")
    ap.add_argument("--model-size", choices=["0.6B", "4B"], default="0.6B",
                    help="qwen3_0_6B vs qwen3_4B factory (same docchunk-agnostic dense path)")
    ap.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH,
                    help="pad/truncate length; global batch = seq_len * 8 (1 instance/GPU on 8 GPUs)")
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="after fit, save model-only checkpoint (config.json + model_and_optim) for eval")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full; smoke)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-entity", default=None)
    opts = ap.parse_args()

    # Debug aid for silent stalls (NFS locks / hub timeouts / collective hangs): periodically dump
    # every thread's Python stack to stderr so the blocking frame shows up in the slurm log.
    import faulthandler

    faulthandler.dump_traceback_later(300, repeat=True)

    prepare_training_environment()
    try:
        build_and_fit(opts)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
