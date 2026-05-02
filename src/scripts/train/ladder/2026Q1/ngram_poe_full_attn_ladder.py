"""
Full-attention ladder with n-gram product-of-experts (PoE) logit bias.

Data pipeline mirrors ``baseline_full_attn_ladder.py`` (same six Dolma2 source
paths, same tokenizer, same ConcatAndChunk chunking). The data wrapper is
``NgramSoftTargetInstanceSource`` with ``output_log_probs=True`` so per-position
top-K log-probabilities are passed through the batch, ready to be added to
the LM's log-probabilities at the K candidate positions.

Train-time loss:

    log p_final(w | h) = log p_lm(w | h) + λ · log p_ngram(w | h) − log Z(h)
    L = − log p_final(label | h)

— in words, the joint log-probability for any token equals the model's
log-probability plus λ times the ngram's log-probability for that token,
minus a per-position normalizer; the loss is the negative log-likelihood of
the joint at the gold next-token. λ is a constant mixing weight (no schedule).

Eval-time consistency: in-loop evaluators don't go through the InstanceSource
wrapper, so ``TransformerTrainModule.eval_batch`` instantiates its own ngram
source and applies the same scatter-add bias before computing CE loss. The
deployed inference distribution and the training-time joint are therefore
the same.

Everything else (z_loss_multiplier=1e-5, fsdp, grad clip, compile, etc.)
matches the baseline exactly.
"""

import argparse
import dataclasses
import logging

import olmo_core.io as io
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

log = logging.getLogger(__name__)

# Matches ``baseline_v02_full_attn_ladder.py`` exactly so the PoE arm is an
# apples-to-apples comparison against the v02 baseline isoparam.
DOLMA2_BASELINE_PATHS = [
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2-fixed/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.2/allenai/dolma2-tokenizer/s2pdf/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/**/*.npy",
]

# Default ngram pilot directory on Weka. Must contain
# ``forward_index_topk.bin`` produced by
# ``data_gen/build_topk_forward_index.py``. Override via ``--ngram-table-dir``
# for smoke tests against a different pilot.
DEFAULT_NGRAM_TABLE_DIR = (
    "/weka/oe-training-default/ai2-llm/ngram-tables/pilots/"
    "pilot-2026-04-22-fraction1e-3-n5"
)

# Top-K size and max ngram order. Must match the precomputed file.
DEFAULT_SOFT_TARGET_K = 16
DEFAULT_SOFT_TARGET_N_MAX = 5

# Constant ngram mixing weight. λ=1.0 is "full PoE." Tunable via
# ``--poe-lambda`` for sweeps.
DEFAULT_POE_LAMBDA = 1.0


@dataclasses.dataclass(kw_only=True)
class _WSDSChinchillaSmoke(WSDSChinchillaRunConfigurator):
    """Smoke variant of WSDS Chinchilla. Mirrors the one in the soft-target
    ladder. Three changes vs production: (a) relaxes the chinchilla_multiple
    >= 0.5 floor and the power-of-2 check; (b) single anneal at the
    requested multiple instead of one per power of 2 from 2^-1; (c) tiny
    fixed batch size + minimal warmup so a 1-GPU smoke runs in minutes
    rather than hours.
    """

    def __post_init__(self):
        from olmo_core.exceptions import OLMoConfigurationError

        if self.chinchilla_multiple <= 0:
            raise OLMoConfigurationError("'chinchilla_multiple' must be positive")
        if not (0 < self.decay_fraction < 0.5):
            raise OLMoConfigurationError(
                "'decay_fraction' must be greater than 0.0 and less than 0.5"
            )

    def configure_chinchilla_periods(self, num_params: int) -> tuple[int, list[float]]:
        return 16384, [self.chinchilla_multiple]

    def configure_target_batch_size(self, num_params: int) -> int:
        return 16384


@dataclasses.dataclass(kw_only=True, eq=True)
class NgramPoEConfigurator(Olmo3ModelConfigurator):
    """
    Olmo3 configurator that plumbs the PoE auxiliary fields into
    :class:`TransformerTrainModuleConfig`. Identical to the parent's
    ``build_train_module`` except for the PoE knobs.

    When ``smoke_1gpu`` is true, override the parent's hardcoded 8-GPU
    minimum device-mesh spec to allow a 1-GPU smoke. At 190M, microbatch
    sizing transfers between 1-GPU and 8-GPU because the binding memory
    constraint is activation memory (not sharded by FSDP).
    """

    poe_lambda: float = DEFAULT_POE_LAMBDA
    ngram_table_dir: str = DEFAULT_NGRAM_TABLE_DIR
    soft_target_k: int = DEFAULT_SOFT_TARGET_K
    soft_target_n_max: int = DEFAULT_SOFT_TARGET_N_MAX
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
        assert "h100" in device_type or "b200" in device_type or "a100" in device_type
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
            poe_lambda=self.poe_lambda,
            poe_ngram_table_dir=self.ngram_table_dir,
            poe_ngram_K=self.soft_target_k,
            poe_ngram_N_max=self.soft_target_n_max,
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
    wrapped_source = NgramSoftTargetInstanceSourceConfig(  # noqa: F405
        source=base_source,
        table_dir=getattr(args, "ngram_table_dir", DEFAULT_NGRAM_TABLE_DIR),
        K=getattr(args, "soft_target_k", DEFAULT_SOFT_TARGET_K),
        N_max=getattr(args, "soft_target_n_max", DEFAULT_SOFT_TARGET_N_MAX),
        # PoE wants raw natural-log kenlm log-probs, not renormalized-over-K
        # linear probs. The downstream train_module's PoE branch reads these
        # under the ``soft_target_log_probs`` key.
        output_log_probs=True,
    )

    instance_sources: list[InstanceSourceConfig] = [wrapped_source]  # noqa: F405

    smoke_1gpu = getattr(args, "smoke_1gpu", False)
    max_devices = 1 if smoke_1gpu else args.max_gpus
    ladder = ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=[s for s in TransformerSize if s.approx_num_params <= 1e9],
        max_devices=max_devices,
        device_type=get_gpu_type(args.cluster),
        model_configurator=NgramPoEConfigurator(
            model_construction_kwargs={"sliding_window": None},
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
            poe_lambda=getattr(args, "poe_lambda", DEFAULT_POE_LAMBDA),
            ngram_table_dir=getattr(args, "ngram_table_dir", DEFAULT_NGRAM_TABLE_DIR),
            soft_target_k=getattr(args, "soft_target_k", DEFAULT_SOFT_TARGET_K),
            soft_target_n_max=getattr(args, "soft_target_n_max", DEFAULT_SOFT_TARGET_N_MAX),
            smoke_1gpu=smoke_1gpu,
        ),
        run_configurator=(
            _WSDSChinchillaSmoke(chinchilla_multiple=args.chinchilla_multiple)
            if smoke_1gpu
            else WSDSChinchillaRunConfigurator(
                chinchilla_multiple=args.chinchilla_multiple
            )
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(  # noqa: F405
            num_workers=16, instance_filter_config=InstanceFilterConfig()  # noqa: F405
        ),
    )
    return ladder


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    """Hook called by ``internal.ladder.main`` to register custom CLI args."""
    parser.add_argument(
        "--ngram-table-dir",
        type=str,
        default=DEFAULT_NGRAM_TABLE_DIR,
        help=(
            "Directory containing forward_index_topk.bin. Defaults to the "
            "pilot 1e-3 n=5 tables on Weka."
        ),
    )
    parser.add_argument(
        "--poe-lambda",
        type=float,
        default=DEFAULT_POE_LAMBDA,
        help="Constant ngram mixing weight in the PoE joint log-prob.",
    )
    parser.add_argument(
        "--soft-target-k",
        type=int,
        default=DEFAULT_SOFT_TARGET_K,
        help="Top-K size for the ngram log-prob bias.",
    )
    parser.add_argument(
        "--smoke-1gpu",
        action="store_true",
        help=(
            "Run on a single GPU for fast end-to-end smoke testing. "
            "Same semantics as the soft-target ladder's flag — overrides "
            "the configurator's 8-GPU minimum, forces max_devices=1, "
            "and swaps in a smoke run-configurator that allows tiny "
            "chinchilla_multiple values with minimal warmup so a smoke "
            "completes in minutes. Pair with --chinchilla-multiple ~0.001."
        ),
    )


if __name__ == "__main__":
    main(configure_ladder, add_additional_args=add_additional_args)
