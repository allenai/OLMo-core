"""
Full-attention ladder with early fusion from Kneser-Ney top-k ngram priors.

This is the early-fusion counterpart to ``ngram_poe_full_attn_ladder.py``.
It uses the same Dolma2 data mix and the same ``NgramTopKInstanceSource``
fields, but trains ordinary hard-label CE. The ngram signal is injected at
the embedding boundary as a learned-scale weighted sum of LM-head
unembedding rows:

    h_0(t) = h_token(t) + alpha * Σ_v p_ngram(v | ctx_t) W_out[v]

The Kneser-Ney probabilities are used as raw full-vocabulary probability
mass, not renormalized over the K candidates. Kneser-Ney smoothing discounts
observed ngram counts and redistributes removed mass through shorter
histories, so this prior comes from the same smoothed continuation set as
the late-fusion PoE runs.
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
DEFAULT_NGRAM_TABLE_DIR = _poe.DEFAULT_NGRAM_TABLE_DIR
DEFAULT_NGRAM_K = _poe.DEFAULT_NGRAM_K
DEFAULT_NGRAM_N_MAX = _poe.DEFAULT_NGRAM_N_MAX
DEFAULT_EARLY_FUSION_ALPHA_INIT = 0.1


@dataclasses.dataclass(kw_only=True, eq=True)
class NgramEarlyFusionConfigurator(Olmo3ModelConfigurator):
    """Olmo3 configurator that enables early ngram fusion, not late PoE."""

    early_fusion_alpha_init: float = DEFAULT_EARLY_FUSION_ALPHA_INIT
    early_fusion_alpha_lr: float | None = None
    ngram_table_dir: str = DEFAULT_NGRAM_TABLE_DIR
    ngram_k: int = DEFAULT_NGRAM_K
    ngram_n_max: int = DEFAULT_NGRAM_N_MAX
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
            compile_model=True,
            dp_config=dp_config,
            z_loss_multiplier=1e-5,
            early_fusion_ngram=True,
            early_fusion_alpha_init=self.early_fusion_alpha_init,
            early_fusion_alpha_lr=self.early_fusion_alpha_lr,
            early_fusion_ngram_table_dir=self.ngram_table_dir,
            early_fusion_ngram_K=self.ngram_k,
            early_fusion_ngram_N_max=self.ngram_n_max,
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
    wrapped_source = NgramTopKInstanceSourceConfig(  # noqa: F405
        source=base_source,
        table_dir=getattr(args, "ngram_table_dir", DEFAULT_NGRAM_TABLE_DIR),
        K=getattr(args, "ngram_k", DEFAULT_NGRAM_K),
        N_max=getattr(args, "ngram_n_max", DEFAULT_NGRAM_N_MAX),
    )

    smoke_1gpu = getattr(args, "smoke_1gpu", False)
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
        model_configurator=NgramEarlyFusionConfigurator(
            model_construction_kwargs=model_construction_kwargs,
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
            early_fusion_alpha_init=getattr(
                args, "early_fusion_alpha_init", DEFAULT_EARLY_FUSION_ALPHA_INIT
            ),
            early_fusion_alpha_lr=getattr(args, "early_fusion_alpha_lr", None),
            ngram_table_dir=getattr(args, "ngram_table_dir", DEFAULT_NGRAM_TABLE_DIR),
            ngram_k=getattr(args, "ngram_k", DEFAULT_NGRAM_K),
            ngram_n_max=getattr(args, "ngram_n_max", DEFAULT_NGRAM_N_MAX),
            smoke_1gpu=smoke_1gpu,
        ),
        run_configurator=(
            _poe._WSDSChinchillaSmoke(chinchilla_multiple=args.chinchilla_multiple)
            if smoke_1gpu
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
        "--ngram-table-dir",
        type=str,
        default=DEFAULT_NGRAM_TABLE_DIR,
        help="Directory containing forward_index_topk.bin.",
    )
    parser.add_argument(
        "--ngram-k",
        type=int,
        default=DEFAULT_NGRAM_K,
        help="Top-K size for the ngram early-fusion prior.",
    )
    parser.add_argument(
        "--ngram-n-max",
        type=int,
        default=DEFAULT_NGRAM_N_MAX,
        help="Maximum ngram order for lookup.",
    )
    parser.add_argument(
        "--early-fusion-alpha-init",
        type=float,
        default=DEFAULT_EARLY_FUSION_ALPHA_INIT,
        help="Positive initial scale for the early-fusion ngram prior.",
    )
    parser.add_argument(
        "--early-fusion-alpha-lr",
        type=float,
        default=None,
        help="Optional optimizer learning rate override for learned alpha.",
    )
    parser.add_argument(
        "--attn-backend",
        choices=("torch", "flash_2", "flash_3", "flash_4", "te"),
        default=None,
        help="Override the automatically selected attention backend.",
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
