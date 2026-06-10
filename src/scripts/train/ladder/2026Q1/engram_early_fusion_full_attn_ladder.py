"""
Full-attention ladder with Engram-style learned early fusion.

This is the learned-memory baseline for KN early fusion. It uses the same
Dolma2 data mix and the same embedding-boundary injection point, but the
default mode hashes raw 2/3-token suffixes into a fixed-size learned memory
budget instead of reading a Kneser-Ney-smoothed continuation distribution.

The older ``low_rank_vocab`` diagnostic remains available explicitly. That
mode uses a raw context-key table, but it must not read KN top-k token IDs, log
probabilities, continuation counts, or backoff weights.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
from pathlib import Path

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import *  # noqa: F401,F403
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    DeviceMeshSpec,
    ModelLadder,
    Olmo3ModelConfigurator,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)


_POE_PATH = Path(__file__).with_name("ngram_poe_full_attn_ladder.py")
_POE_SPEC = importlib.util.spec_from_file_location("_ngram_poe_full_attn_ladder", _POE_PATH)
assert _POE_SPEC is not None and _POE_SPEC.loader is not None
_poe = importlib.util.module_from_spec(_POE_SPEC)
_POE_SPEC.loader.exec_module(_poe)

DOLMA2_BASELINE_PATHS = _poe.DOLMA2_BASELINE_PATHS
DEFAULT_ENGRAM_TABLE_DIR = (
    "/weka/oe-training-default/ai2-llm/ngram-tables/pilots/"
    "pilot-2026-06-03-fraction1e-2-n5-prune0-1-3-10-20-v1"
)
DEFAULT_ENGRAM_N_MAX = 5
DEFAULT_ENGRAM_CODE_DIM = 16
DEFAULT_ENGRAM_TOP_M = 32
DEFAULT_ENGRAM_ALPHA_INIT = 5.0
DEFAULT_ENGRAM_VOCAB_CHUNK_SIZE = 4096
DEFAULT_ENGRAM_MODE = "hashed_memory"
DEFAULT_ENGRAM_NGRAM_ORDERS = (2, 3)
DEFAULT_ENGRAM_HEADS_PER_ORDER = 4
DEFAULT_ENGRAM_HEAD_DIM = 4
DEFAULT_ENGRAM_SLOTS_PER_TABLE = None
DEFAULT_ENGRAM_HASH_SEED = 17


@dataclasses.dataclass(kw_only=True, eq=True)
class EngramEarlyFusionConfigurator(Olmo3ModelConfigurator):
    """Olmo3 configurator that enables learned Engram early fusion."""

    engram_alpha_init: float = DEFAULT_ENGRAM_ALPHA_INIT
    engram_alpha_lr: float | None = None
    engram_mode: str = DEFAULT_ENGRAM_MODE
    engram_table_dir: str | None = DEFAULT_ENGRAM_TABLE_DIR
    engram_n_max: int = DEFAULT_ENGRAM_N_MAX
    engram_code_dim: int = DEFAULT_ENGRAM_CODE_DIM
    engram_top_m: int = DEFAULT_ENGRAM_TOP_M
    engram_vocab_chunk_size: int = DEFAULT_ENGRAM_VOCAB_CHUNK_SIZE
    engram_ngram_orders: tuple[int, ...] = DEFAULT_ENGRAM_NGRAM_ORDERS
    engram_heads_per_order: int = DEFAULT_ENGRAM_HEADS_PER_ORDER
    engram_head_dim: int = DEFAULT_ENGRAM_HEAD_DIM
    engram_slots_per_table: int | None = DEFAULT_ENGRAM_SLOTS_PER_TABLE
    engram_hash_seed: int = DEFAULT_ENGRAM_HASH_SEED
    compile_model: bool = True
    smoke_1gpu: bool = False

    def configure_minimal_device_mesh_spec(
        self,
        *,
        size_spec,
        sequence_length,
        device_type,
    ) -> DeviceMeshSpec:
        if self.smoke_1gpu:
            return DeviceMeshSpec(world_size=1, dp_world_size=1)
        return super().configure_minimal_device_mesh_spec(
            size_spec=size_spec,
            sequence_length=sequence_length,
            device_type=device_type,
        )

    def build_train_module(
        self,
        *,
        size_spec,
        sequence_length,
        rank_microbatch_size,
        model_config,
        optim_config,
        scheduler,
        device_type,
    ) -> TransformerTrainModule:
        device_type = device_type.lower()
        assert (
            "h100" in device_type
            or "b200" in device_type
            or "a100" in device_type
            or "l40" in device_type
        )
        assert sequence_length in {2048, 4096, 8192}
        size_spec = TransformerSize(size_spec)

        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        )

        train_module_config = TransformerTrainModuleConfig(
            rank_microbatch_size=rank_microbatch_size,
            max_sequence_length=sequence_length,
            optim=optim_config,
            compile_model=self.compile_model,
            dp_config=dp_config,
            z_loss_multiplier=1e-5,
            early_fusion_engram=True,
            early_fusion_engram_alpha_init=self.engram_alpha_init,
            early_fusion_engram_alpha_lr=self.engram_alpha_lr,
            early_fusion_engram_mode=self.engram_mode,
            early_fusion_engram_table_dir=(
                self.engram_table_dir if self.engram_mode == "low_rank_vocab" else None
            ),
            early_fusion_engram_N_max=self.engram_n_max,
            early_fusion_engram_code_dim=self.engram_code_dim,
            early_fusion_engram_top_m=self.engram_top_m,
            early_fusion_engram_vocab_chunk_size=self.engram_vocab_chunk_size,
            early_fusion_engram_ngram_orders=self.engram_ngram_orders,
            early_fusion_engram_heads_per_order=self.engram_heads_per_order,
            early_fusion_engram_head_dim=self.engram_head_dim,
            early_fusion_engram_slots_per_table=self.engram_slots_per_table,
            early_fusion_engram_hash_seed=self.engram_hash_seed,
            max_grad_norm=1.0,
            scheduler=scheduler,
        )

        model = model_config.build(init_device="meta")
        train_module = train_module_config.build(model)
        assert isinstance(train_module, TransformerTrainModule)
        return train_module


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()

    base_source = ConcatAndChunkInstanceSourceConfig(  # noqa: F405
        sources=[
            NumpyDocumentSourceConfig(  # noqa: F405
                source_paths=DOLMA2_BASELINE_PATHS,
                tokenizer=tokenizer,
            ),
        ],
        sequence_length=args.sequence_length,
    )
    engram_mode = getattr(args, "engram_mode", DEFAULT_ENGRAM_MODE)
    if engram_mode == "low_rank_vocab":
        instance_source = NgramContextInstanceSourceConfig(  # noqa: F405
            source=base_source,
            table_dir=getattr(args, "engram_table_dir", DEFAULT_ENGRAM_TABLE_DIR),
            N_max=getattr(args, "engram_n_max", DEFAULT_ENGRAM_N_MAX),
        )
    else:
        _reject_low_rank_only_args_for_hashed_mode(args)
        instance_source = base_source

    smoke_1gpu = getattr(args, "smoke_1gpu", False)
    smoke_run = getattr(args, "smoke_run", False) or smoke_1gpu
    max_devices = 1 if smoke_1gpu else args.max_gpus
    model_construction_kwargs = {"sliding_window": None}
    if getattr(args, "attn_backend", None) is not None:
        model_construction_kwargs["attn_backend"] = args.attn_backend

    return ModelLadder(
        name=args.name,
        dir=str(
            _poe.io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)
        ),
        sizes=[s for s in TransformerSize if s.approx_num_params <= 1e9],
        max_devices=max_devices,
        device_type=get_gpu_type(args.cluster),
        model_configurator=EngramEarlyFusionConfigurator(
            model_construction_kwargs=model_construction_kwargs,
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
            engram_alpha_init=getattr(
                args, "engram_alpha_init", DEFAULT_ENGRAM_ALPHA_INIT
            ),
            engram_alpha_lr=getattr(args, "engram_alpha_lr", None),
            engram_mode=engram_mode,
            engram_table_dir=getattr(args, "engram_table_dir", DEFAULT_ENGRAM_TABLE_DIR),
            engram_n_max=getattr(args, "engram_n_max", DEFAULT_ENGRAM_N_MAX),
            engram_code_dim=getattr(args, "engram_code_dim", DEFAULT_ENGRAM_CODE_DIM),
            engram_top_m=getattr(args, "engram_top_m", DEFAULT_ENGRAM_TOP_M),
            engram_vocab_chunk_size=getattr(
                args, "engram_vocab_chunk_size", DEFAULT_ENGRAM_VOCAB_CHUNK_SIZE
            ),
            engram_ngram_orders=tuple(
                getattr(args, "engram_ngram_orders", DEFAULT_ENGRAM_NGRAM_ORDERS)
            ),
            engram_heads_per_order=getattr(
                args, "engram_heads_per_order", DEFAULT_ENGRAM_HEADS_PER_ORDER
            ),
            engram_head_dim=getattr(args, "engram_head_dim", DEFAULT_ENGRAM_HEAD_DIM),
            engram_slots_per_table=getattr(
                args, "engram_slots_per_table", DEFAULT_ENGRAM_SLOTS_PER_TABLE
            ),
            engram_hash_seed=getattr(
                args, "engram_hash_seed", DEFAULT_ENGRAM_HASH_SEED
            ),
            compile_model=getattr(args, "compile_model", True),
            smoke_1gpu=smoke_1gpu,
        ),
        run_configurator=(
            _poe._WSDSChinchillaSmoke(chinchilla_multiple=args.chinchilla_multiple)
            if smoke_run
            else WSDSChinchillaRunConfigurator(
                chinchilla_multiple=args.chinchilla_multiple
            )
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=[instance_source],
        data_loader=ComposableDataLoaderConfig(  # noqa: F405
            num_workers=16, instance_filter_config=InstanceFilterConfig()  # noqa: F405
        ),
    )


def _reject_low_rank_only_args_for_hashed_mode(args: argparse.Namespace) -> None:
    low_rank_only_args = (
        ("--engram-table-dir", "engram_table_dir", DEFAULT_ENGRAM_TABLE_DIR),
        ("--engram-n-max", "engram_n_max", DEFAULT_ENGRAM_N_MAX),
        ("--engram-code-dim", "engram_code_dim", DEFAULT_ENGRAM_CODE_DIM),
        ("--engram-top-m", "engram_top_m", DEFAULT_ENGRAM_TOP_M),
        (
            "--engram-vocab-chunk-size",
            "engram_vocab_chunk_size",
            DEFAULT_ENGRAM_VOCAB_CHUNK_SIZE,
        ),
    )
    overridden = [
        flag
        for flag, attr, default in low_rank_only_args
        if getattr(args, attr, default) != default
    ]
    if overridden:
        raise ValueError(
            "These options only apply with --engram-mode low_rank_vocab and would "
            f"be ignored in hashed_memory mode: {', '.join(overridden)}"
        )


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engram-mode",
        choices=("hashed_memory", "low_rank_vocab"),
        default=DEFAULT_ENGRAM_MODE,
        help=(
            "Engram baseline mode. hashed_memory hashes raw token suffixes into "
            "a fixed V*d_model budget; low_rank_vocab keeps the older context-code "
            "plus vocab-decoder diagnostic."
        ),
    )
    parser.add_argument(
        "--engram-table-dir",
        type=str,
        default=DEFAULT_ENGRAM_TABLE_DIR,
        help="Directory containing raw forward_index.bin for low_rank_vocab mode.",
    )
    parser.add_argument(
        "--engram-n-max",
        type=int,
        default=DEFAULT_ENGRAM_N_MAX,
        help="Maximum ngram order for context-key lookup.",
    )
    parser.add_argument(
        "--engram-code-dim",
        type=int,
        default=DEFAULT_ENGRAM_CODE_DIM,
        help="Learned context-code dimension.",
    )
    parser.add_argument(
        "--engram-top-m",
        type=int,
        default=DEFAULT_ENGRAM_TOP_M,
        help="Number of learned vocabulary candidates to select per context.",
    )
    parser.add_argument(
        "--engram-alpha-init",
        type=float,
        default=DEFAULT_ENGRAM_ALPHA_INIT,
        help="Positive initial scale for the learned Engram prior.",
    )
    parser.add_argument(
        "--engram-alpha-lr",
        type=float,
        default=None,
        help="Optional optimizer learning rate override for learned alpha.",
    )
    parser.add_argument(
        "--engram-vocab-chunk-size",
        type=int,
        default=DEFAULT_ENGRAM_VOCAB_CHUNK_SIZE,
        help="Vocabulary chunk size for exact sparse top-M selection.",
    )
    parser.add_argument(
        "--engram-ngram-orders",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENGRAM_NGRAM_ORDERS),
        help="Ngram suffix orders to hash in hashed_memory mode.",
    )
    parser.add_argument(
        "--engram-heads-per-order",
        type=int,
        default=DEFAULT_ENGRAM_HEADS_PER_ORDER,
        help="Number of independent hash heads per ngram order in hashed_memory mode.",
    )
    parser.add_argument(
        "--engram-head-dim",
        type=int,
        default=DEFAULT_ENGRAM_HEAD_DIM,
        help="Embedding width per hash hit in hashed_memory mode.",
    )
    parser.add_argument(
        "--engram-slots-per-table",
        type=int,
        default=DEFAULT_ENGRAM_SLOTS_PER_TABLE,
        help=(
            "Slots per order/head hash table. Defaults to the largest value "
            "that fits the V*d_model parameter budget."
        ),
    )
    parser.add_argument(
        "--engram-hash-seed",
        type=int,
        default=DEFAULT_ENGRAM_HASH_SEED,
        help="Deterministic hash seed for hashed_memory mode.",
    )
    parser.add_argument(
        "--no-compile-model",
        dest="compile_model",
        action="store_false",
        help="Disable torch.compile for Engram smoke/debug runs.",
    )
    parser.set_defaults(compile_model=True)
    parser.add_argument(
        "--attn-backend",
        choices=("torch", "flash_2", "flash_3", "flash_4", "te"),
        default=None,
        help="Override the automatically selected attention backend.",
    )
    parser.add_argument(
        "--smoke-run",
        action="store_true",
        help=(
            "Use the short smoke-test schedule without forcing the run onto a "
            "single GPU."
        ),
    )
    parser.add_argument(
        "--smoke-1gpu",
        action="store_true",
        help=(
            "Run on a single GPU for fast end-to-end smoke testing. Pair with "
            "--chinchilla-multiple ~0.001."
        ),
    )


if __name__ == "__main__":
    main(configure_ladder, add_additional_args=add_additional_args)
