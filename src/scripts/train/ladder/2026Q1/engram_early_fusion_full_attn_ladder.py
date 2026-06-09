"""
Full-attention ladder with Engram-style learned early fusion.

This is the learned-memory baseline for KN early fusion. It uses the same
Dolma2 data mix and the same embedding-boundary injection point, but the
per-position ngram signal is a learned static memory row keyed by the observed
context instead of a Kneser-Ney-smoothed continuation distribution.

For context h, the model learns a code c_h and a shared token decoder q_v:

    s_h(v) = dot(c_h, q_v)
    e_h = sum_{v in top-M(s_h)} softmax(s_h)(v) W_out[v]
    h_0(t) = h_token(t) + alpha * e_h

The context-key table may come from the same ngram build used by KN early
fusion, but this baseline must not read KN top-k token IDs, log probabilities,
continuation counts, or backoff weights.
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


@dataclasses.dataclass(kw_only=True, eq=True)
class EngramEarlyFusionConfigurator(Olmo3ModelConfigurator):
    """Olmo3 configurator that enables learned Engram early fusion."""

    engram_alpha_init: float = DEFAULT_ENGRAM_ALPHA_INIT
    engram_alpha_lr: float | None = None
    engram_table_dir: str = DEFAULT_ENGRAM_TABLE_DIR
    engram_n_max: int = DEFAULT_ENGRAM_N_MAX
    engram_code_dim: int = DEFAULT_ENGRAM_CODE_DIM
    engram_top_m: int = DEFAULT_ENGRAM_TOP_M
    engram_vocab_chunk_size: int = DEFAULT_ENGRAM_VOCAB_CHUNK_SIZE
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
            early_fusion_engram_table_dir=self.engram_table_dir,
            early_fusion_engram_N_max=self.engram_n_max,
            early_fusion_engram_code_dim=self.engram_code_dim,
            early_fusion_engram_top_m=self.engram_top_m,
            early_fusion_engram_vocab_chunk_size=self.engram_vocab_chunk_size,
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
    wrapped_source = NgramContextInstanceSourceConfig(  # noqa: F405
        source=base_source,
        table_dir=getattr(args, "engram_table_dir", DEFAULT_ENGRAM_TABLE_DIR),
        N_max=getattr(args, "engram_n_max", DEFAULT_ENGRAM_N_MAX),
    )

    smoke_1gpu = getattr(args, "smoke_1gpu", False)
    smoke_run = getattr(args, "smoke_run", False) or smoke_1gpu
    max_devices = 1 if smoke_1gpu else args.max_gpus
    model_construction_kwargs = {"sliding_window": None}
    if getattr(args, "attn_backend", None) is not None:
        model_construction_kwargs["attn_backend"] = args.attn_backend

    return ModelLadder(
        name=args.name,
        dir=str(_poe.io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=[s for s in TransformerSize if s.approx_num_params <= 1e9],
        max_devices=max_devices,
        device_type=get_gpu_type(args.cluster),
        model_configurator=EngramEarlyFusionConfigurator(
            model_construction_kwargs=model_construction_kwargs,
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
            engram_alpha_init=getattr(args, "engram_alpha_init", DEFAULT_ENGRAM_ALPHA_INIT),
            engram_alpha_lr=getattr(args, "engram_alpha_lr", None),
            engram_table_dir=getattr(args, "engram_table_dir", DEFAULT_ENGRAM_TABLE_DIR),
            engram_n_max=getattr(args, "engram_n_max", DEFAULT_ENGRAM_N_MAX),
            engram_code_dim=getattr(args, "engram_code_dim", DEFAULT_ENGRAM_CODE_DIM),
            engram_top_m=getattr(args, "engram_top_m", DEFAULT_ENGRAM_TOP_M),
            engram_vocab_chunk_size=getattr(
                args, "engram_vocab_chunk_size", DEFAULT_ENGRAM_VOCAB_CHUNK_SIZE
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
        instance_sources=[wrapped_source],
        data_loader=ComposableDataLoaderConfig(  # noqa: F405
            num_workers=16, instance_filter_config=InstanceFilterConfig()  # noqa: F405
        ),
    )


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engram-table-dir",
        type=str,
        default=DEFAULT_ENGRAM_TABLE_DIR,
        help="Directory containing raw forward_index.bin.",
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
