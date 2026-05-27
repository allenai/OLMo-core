"""Full-attention ladder with **stupid-backoff** n-gram product-of-experts
(PoE) logit bias.

Parallel of :mod:`ngram_poe_full_attn_ladder` (which uses the KN-smoothed
top-K forward index). The differences:

- Data wrapper is :class:`NgramStupidBackoffInstanceSource` instead of
  :class:`NgramSoftTargetInstanceSource`. The instance source emits
  per-position ragged ``sb_override_*`` arrays — variable-K overrides
  per position, not the fixed-K dense slots the KN-smoothed path uses.
- ``TransformerTrainModuleConfig`` is configured with ``poe_sb_table_dir``
  / ``poe_sb_alpha`` / ``poe_sb_N_max`` (mutually exclusive with the
  KN-smoothed ``poe_ngram_table_dir``). λ is shared via ``poe_lambda``.

Train-time loss::

    log p_final(w | h) = log p_lm(w | h) + λ · sb_score(w | h_t) − log Z(h)
    L = − log p_final(label | h)

where ``sb_score(w | h_t)`` is the highest-order stupid-backoff score for
``w`` given history ``h_t`` (with universal length-V unigram floor as the
fallback when ``w`` isn't observed at any higher order). The training
step adds the bias on top of the LM's logits via
:func:`olmo_core.data.sb_bias.apply_sb_bias_inplace`, then standard
cross-entropy at the gold label.

Eval-time consistency: in-loop evaluators don't go through the
InstanceSource wrapper, so :meth:`TransformerTrainModule._apply_poe_eval_bias_sb`
synchronously recomputes the per-instance overrides from the same SB
reader and applies the matching bias before computing CE — the inference
distribution and the training-time joint are therefore equivalent.

Everything else (z_loss_multiplier=1e-5, fsdp, grad clip, compile, etc.)
matches the baseline / KN-smoothed PoE ladders exactly.
"""

import argparse
import dataclasses
import json
import logging
import math
import os
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

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
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)

DOLMA2_BASELINE_PATHS = [
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2-fixed/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/arxiv/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/finemath-3plus/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.2/allenai/dolma2-tokenizer/s2pdf/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/stack-edu/**/*.npy",
    "/weka/oe-training-default/ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/wikipedia/**/*.npy",
]

# Default SB counts_index pilot directory on Weka. Must contain
# ``counts_index/`` built at format_version=2 by
# ``data_gen/build_counts_index.py``.
DEFAULT_SB_TABLE_DIR = (
    "/weka/oe-training-default/ai2-llm/ngram-tables/pilots/"
    "pilot-2026-04-22-fraction1e-3-n5-counts"
)

DEFAULT_SB_N_MAX = 5
DEFAULT_SB_ALPHA = 0.4         # Brants et al 2007.
DEFAULT_POE_LAMBDA = 0.1       # Same default as the KN-smoothed PoE 190M sweep winner.
DEFAULT_DOLMA2_VOCAB_SIZE = 100352
DEFAULT_SB_MAX_ORDER2_CONTINUATIONS = None
DEFAULT_SB_LOOKUP_THREADS = 1
DEFAULT_DATA_LOADER_WORKERS = 4
DEFAULT_DATA_LOADER_PREFETCH_FACTOR = None
DEFAULT_SMOKE_SOURCE_INSTANCES = None


def _parse_order_caps(raw_caps: list[str] | None) -> dict[int, int] | None:
    if not raw_caps:
        return None
    caps: dict[int, int] = {}
    for raw in raw_caps:
        if "=" not in raw:
            raise ValueError(f"expected order cap as ORDER=CAP, got {raw!r}")
        order_s, cap_s = raw.split("=", 1)
        order = int(order_s)
        cap = int(cap_s)
        if order < 2:
            raise ValueError(f"order cap must target order >= 2, got {order}")
        if cap <= 0:
            raise ValueError(f"order cap must be positive, got {cap}")
        caps[order] = cap
    return caps


def _parse_order_int_map(raw_values: list[str] | None, *, label: str) -> dict[int, int] | None:
    if not raw_values:
        return None
    values: dict[int, int] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"expected {label} as ORDER=VALUE, got {raw!r}")
        order_s, value_s = raw.split("=", 1)
        order = int(order_s)
        value = int(value_s)
        if order < 2:
            raise ValueError(f"{label} must target order >= 2, got {order}")
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
        values[order] = value
    return values


def _parse_order_float_map(raw_values: list[str] | None, *, label: str) -> dict[int, float] | None:
    if not raw_values:
        return None
    values: dict[int, float] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"expected {label} as ORDER=VALUE, got {raw!r}")
        order_s, value_s = raw.split("=", 1)
        order = int(order_s)
        value = float(value_s)
        if order < 2:
            raise ValueError(f"{label} must target order >= 2, got {order}")
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
        values[order] = value
    return values


def _resolve_counts_index_dir(table_dir: str) -> Path:
    path = Path(table_dir)
    if (path / "counts_index").is_dir():
        return path / "counts_index"
    if (path / "meta.json").is_file():
        return path
    raise ValueError(
        f"SB table dir must be a pilot dir containing counts_index/ or a counts_index/ dir: {table_dir}"
    )


def _mean_count_for_order(index_dir: Path, order: int, n_pairs: int, *, chunk_entries: int) -> float:
    counts = np.memmap(index_dir / f"order{order}.counts.bin", dtype=np.uint64, mode="r")
    total = 0
    for start in range(0, n_pairs, chunk_entries):
        end = min(start + chunk_entries, n_pairs)
        chunk = np.asarray(counts[start:end], dtype=np.uint64)
        total += int(chunk.sum(dtype=np.uint64))
    return total / n_pairs


def _median_count_for_order(index_dir: Path, order: int, n_pairs: int, *, max_samples: int) -> float:
    counts = np.memmap(index_dir / f"order{order}.counts.bin", dtype=np.uint64, mode="r")
    if n_pairs <= max_samples:
        sample = np.asarray(counts[:], dtype=np.uint64)
    else:
        stride = max(1, math.ceil(n_pairs / max_samples))
        sample = np.asarray(counts[::stride], dtype=np.uint64)
    return float(np.median(sample))


def _derive_min_order_counts_from_ratios(
    table_dir: str,
    ratios: dict[int, float] | None,
    *,
    stat: str,
    chunk_entries: int,
    median_samples: int,
) -> dict[int, int] | None:
    if not ratios:
        return None
    if stat not in {"mean", "median"}:
        raise ValueError(f"stat must be 'mean' or 'median', got {stat!r}")
    if chunk_entries <= 0:
        raise ValueError("--sb-min-order-count-ratio-chunk-entries must be positive")
    if median_samples <= 0:
        raise ValueError("--sb-min-order-count-ratio-median-samples must be positive")

    index_dir = _resolve_counts_index_dir(table_dir)
    with open(index_dir / "meta.json") as f:
        meta = json.load(f)

    derived: dict[int, int] = {}
    for order, ratio in sorted(ratios.items()):
        info = meta["per_order"].get(str(order))
        if info is None:
            raise ValueError(f"order {order} is not present in {index_dir / 'meta.json'}")
        n_pairs = int(info["n_pairs"])
        if n_pairs <= 0:
            raise ValueError(f"order {order} has no continuation rows")
        if stat == "mean":
            base = _mean_count_for_order(index_dir, order, n_pairs, chunk_entries=chunk_entries)
        else:
            base = _median_count_for_order(
                index_dir,
                order,
                n_pairs,
                max_samples=median_samples,
            )
        threshold = max(1, int(math.ceil(base * ratio)))
        derived[order] = threshold
        log.info(
            "Derived SB min-count threshold order=%s stat=%s base=%.4f ratio=%.4f threshold=%s",
            order,
            stat,
            base,
            ratio,
            threshold,
        )
    return derived


class HeadInstanceSource(InstanceSource):  # noqa: F405
    """Cheap first-N wrapper for smoke tests.

    This avoids building a full-source global order file when a smoke run only
    needs a small number of batches.
    """

    def __init__(self, source: InstanceSource, max_instances: int, *, work_dir):  # noqa: F405
        super().__init__(
            work_dir=work_dir,
            sequence_length=source.sequence_length,
            max_sequence_length=source.max_sequence_length,
            label=source.label,
        )
        self.source = source
        self.max_instances = min(int(max_instances), len(source))
        if self.max_instances <= 0:
            raise ValueError(f"max_instances must be positive, got {max_instances}")

    @property
    def num_instances(self) -> int:
        return self.max_instances

    @property
    def fingerprint(self) -> str:
        return f"head-{self.max_instances}-{self.source.fingerprint}"

    def __len__(self) -> int:
        return self.max_instances

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.source[self.validate_index(idx)]

    def children(self):
        return [self.source]


@dataclasses.dataclass
class HeadInstanceSourceConfig(InstanceSourceConfig):  # noqa: F405
    source: InstanceSourceConfig  # noqa: F405
    max_instances: int

    def build(self, work_dir) -> HeadInstanceSource:
        return HeadInstanceSource(
            self.source.build(work_dir),
            self.max_instances,
            work_dir=work_dir,
        )


@dataclasses.dataclass(kw_only=True)
class _WSDSChinchillaSmoke(WSDSChinchillaRunConfigurator):
    """Smoke variant of WSDS Chinchilla, copied from the KN-smoothed
    ladder so a 1-GPU smoke completes in minutes rather than hours."""

    target_batch_size: int = 16384

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
        return self.target_batch_size


@dataclasses.dataclass(kw_only=True, eq=True)
class NgramSBPoEConfigurator(Olmo3ModelConfigurator):
    """Olmo3 configurator that wires SB-PoE knobs into the
    :class:`TransformerTrainModuleConfig`."""

    poe_lambda: float = DEFAULT_POE_LAMBDA
    learn_poe_lambda: bool = False
    poe_lambda_lr: float | None = None
    sb_table_dir: str = DEFAULT_SB_TABLE_DIR
    sb_alpha: float = DEFAULT_SB_ALPHA
    sb_n_max: int = DEFAULT_SB_N_MAX
    sb_max_order2_continuations: int | None = DEFAULT_SB_MAX_ORDER2_CONTINUATIONS
    sb_max_order_continuations: dict[int, int] | None = None
    sb_min_order_counts: dict[int, int] | None = None
    sb_index_access: str = "mmap"
    sb_mirror_to_shm: bool = True
    sb_lookup_threads: int = DEFAULT_SB_LOOKUP_THREADS
    sb_eval_lookup_threads: int | None = None
    sb_topk_uniform_residual_k: int | None = None
    eval_rank_microbatch_size: int | None = None
    dolma2_vocab_size: int = DEFAULT_DOLMA2_VOCAB_SIZE
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
            eval_rank_microbatch_size=self.eval_rank_microbatch_size,
            max_sequence_length=sequence_length,
            optim=optim_config,
            compile_model=True,
            dp_config=dp_config,
            z_loss_multiplier=1e-5,
            poe_lambda=self.poe_lambda,
            poe_lambda_learnable=self.learn_poe_lambda,
            poe_lambda_lr=self.poe_lambda_lr,
            poe_sb_table_dir=self.sb_table_dir,
            poe_sb_alpha=self.sb_alpha,
            poe_sb_N_max=self.sb_n_max,
            poe_sb_dolma2_vocab_size=self.dolma2_vocab_size,
            poe_sb_max_order2_continuations=self.sb_max_order2_continuations,
            poe_sb_max_order_continuations=self.sb_max_order_continuations,
            poe_sb_min_order_counts=self.sb_min_order_counts,
            poe_sb_index_access=self.sb_index_access,
            poe_sb_mirror_to_shm=self.sb_mirror_to_shm,
            poe_sb_lookup_threads=self.sb_lookup_threads,
            poe_sb_eval_lookup_threads=self.sb_eval_lookup_threads,
            poe_sb_topk_uniform_residual_k=self.sb_topk_uniform_residual_k,
            max_grad_norm=1.0,
            scheduler=scheduler,
        )

        model = model_config.build(init_device="meta")
        train_module = train_module_config.build(model)
        assert isinstance(train_module, TransformerTrainModule)
        return train_module


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    if getattr(args, "sb_eval_timing", False):
        os.environ["OLMO_SB_EVAL_TIMING"] = "1"

    tokenizer = TokenizerConfig.dolma2()
    sb_order_caps = _parse_order_caps(getattr(args, "sb_max_order_continuations", None))
    sb_min_order_count_ratios = _parse_order_float_map(
        getattr(args, "sb_min_order_count_ratio", None),
        label="minimum order count ratio",
    )
    sb_ratio_counts = _derive_min_order_counts_from_ratios(
        getattr(args, "sb_table_dir", DEFAULT_SB_TABLE_DIR),
        sb_min_order_count_ratios,
        stat=getattr(args, "sb_min_order_count_ratio_stat", "mean"),
        chunk_entries=getattr(args, "sb_min_order_count_ratio_chunk_entries", 10_000_000),
        median_samples=getattr(args, "sb_min_order_count_ratio_median_samples", 2_000_000),
    )
    sb_min_order_counts = _parse_order_int_map(
        getattr(args, "sb_min_order_count", None),
        label="minimum order count",
    )
    if sb_ratio_counts:
        merged_counts = dict(sb_ratio_counts)
        if sb_min_order_counts:
            overlap = sorted(set(merged_counts).intersection(sb_min_order_counts))
            if overlap:
                log.info(
                    "Explicit --sb-min-order-count overrides ratio-derived thresholds for orders %s",
                    overlap,
                )
            merged_counts.update(sb_min_order_counts)
        sb_min_order_counts = merged_counts
    if sb_min_order_counts:
        log.info("Using SB minimum raw-count thresholds by order: %s", sb_min_order_counts)

    base_source = ConcatAndChunkInstanceSourceConfig(  # noqa: F405
        sources=[
            NumpyDocumentSourceConfig(  # noqa: F405
                source_paths=DOLMA2_BASELINE_PATHS,
                tokenizer=tokenizer,
            ),
        ],
        sequence_length=args.sequence_length,
    )
    smoke_source_instances = getattr(
        args, "smoke_source_instances", DEFAULT_SMOKE_SOURCE_INSTANCES
    )
    if smoke_source_instances is not None:
        base_source = HeadInstanceSourceConfig(
            source=base_source,
            max_instances=int(smoke_source_instances),
        )
    wrapped_source = NgramStupidBackoffInstanceSourceConfig(  # noqa: F405
        source=base_source,
        table_dir=getattr(args, "sb_table_dir", DEFAULT_SB_TABLE_DIR),
        dolma2_vocab_size=int(tokenizer.padded_vocab_size()),
        N_max=getattr(args, "sb_n_max", DEFAULT_SB_N_MAX),
        alpha=getattr(args, "sb_alpha", DEFAULT_SB_ALPHA),
        max_order2_continuations=getattr(
            args, "sb_max_order2_continuations", DEFAULT_SB_MAX_ORDER2_CONTINUATIONS
        ),
        max_order_continuations=sb_order_caps,
        min_order_counts=sb_min_order_counts,
        index_access=getattr(args, "sb_index_access", "mmap"),
        mirror_to_shm=getattr(args, "sb_mirror_to_shm", True),
        lookup_threads=getattr(args, "sb_lookup_threads", DEFAULT_SB_LOOKUP_THREADS),
        topk_uniform_residual_k=getattr(args, "sb_topk_uniform_residual_k", None),
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
        model_configurator=NgramSBPoEConfigurator(
            model_construction_kwargs={"sliding_window": None},
            rank_microbatch_size=None
            if args.rank_mbz is None
            else args.rank_mbz * args.sequence_length,
            eval_rank_microbatch_size=None
            if getattr(args, "eval_rank_mbz", None) is None
            else args.eval_rank_mbz * args.sequence_length,
            poe_lambda=getattr(args, "poe_lambda", DEFAULT_POE_LAMBDA),
            learn_poe_lambda=getattr(args, "learn_poe_lambda", False),
            poe_lambda_lr=getattr(args, "poe_lambda_lr", None),
            sb_table_dir=getattr(args, "sb_table_dir", DEFAULT_SB_TABLE_DIR),
            sb_alpha=getattr(args, "sb_alpha", DEFAULT_SB_ALPHA),
            sb_n_max=getattr(args, "sb_n_max", DEFAULT_SB_N_MAX),
            sb_max_order2_continuations=getattr(
                args,
                "sb_max_order2_continuations",
                DEFAULT_SB_MAX_ORDER2_CONTINUATIONS,
            ),
            sb_max_order_continuations=sb_order_caps,
            sb_min_order_counts=sb_min_order_counts,
            sb_index_access=getattr(args, "sb_index_access", "mmap"),
            sb_mirror_to_shm=getattr(args, "sb_mirror_to_shm", True),
            sb_lookup_threads=getattr(
                args,
                "sb_lookup_threads",
                DEFAULT_SB_LOOKUP_THREADS,
            ),
            sb_eval_lookup_threads=getattr(args, "sb_eval_lookup_threads", None),
            sb_topk_uniform_residual_k=getattr(
                args, "sb_topk_uniform_residual_k", None
            ),
            dolma2_vocab_size=int(tokenizer.padded_vocab_size()),
            smoke_1gpu=smoke_1gpu,
        ),
        run_configurator=(
            _WSDSChinchillaSmoke(
                chinchilla_multiple=args.chinchilla_multiple,
                target_batch_size=getattr(args, "smoke_target_batch_size", 16384),
            )
            if smoke_1gpu
            else WSDSChinchillaRunConfigurator(
                chinchilla_multiple=args.chinchilla_multiple
            )
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(  # noqa: F405
            # SB override construction is CPU/data-loader heavy. The default
            # stays conservative for provenance, but the CLI exposes worker and
            # prefetch controls so throughput sweeps can raise workers without
            # also letting every worker queue multiple million-row examples.
            num_workers=getattr(args, "data_loader_workers", DEFAULT_DATA_LOADER_WORKERS),
            prefetch_factor=getattr(
                args,
                "data_loader_prefetch_factor",
                DEFAULT_DATA_LOADER_PREFETCH_FACTOR,
            ),
            instance_filter_config=InstanceFilterConfig(),  # noqa: F405
        ),
    )
    load_path = getattr(args, "load_path", None)
    if (
        load_path is not None
        or getattr(args, "eval_only", False)
        or getattr(args, "lm_eval_only", False)
        or getattr(args, "disable_evals", False)
        or getattr(args, "final_eval_only", False)
        or getattr(args, "eval_max_batches", None) is not None
    ):
        original_configure_trainer = ladder._configure_trainer

        def _configure_trainer_with_eval_options(self, size_spec, for_benchmarking=False):
            trainer_config = original_configure_trainer(size_spec, for_benchmarking=for_benchmarking)
            callbacks = dict(trainer_config.callbacks)
            if getattr(args, "lm_eval_only", False):
                callbacks.pop("downstream_evaluator", None)
            if getattr(args, "disable_evals", False):
                callbacks.pop("lm_evaluator", None)
                callbacks.pop("downstream_evaluator", None)
            else:
                if getattr(args, "final_eval_only", False):
                    for callback_name in ("lm_evaluator", "downstream_evaluator"):
                        callback = callbacks.get(callback_name)
                        if callback is not None:
                            callbacks[callback_name] = dataclasses.replace(
                                callback,
                                eval_interval=None,
                                fixed_steps=None,
                                eval_on_finish=True,
                            )
                eval_max_batches = getattr(args, "eval_max_batches", None)
                if eval_max_batches is not None:
                    for callback_name in ("lm_evaluator", "downstream_evaluator"):
                        callback = callbacks.get(callback_name)
                        if callback is not None:
                            callbacks[callback_name] = dataclasses.replace(
                                callback,
                                eval_duration=Duration.steps(int(eval_max_batches)),
                            )
            if getattr(args, "eval_only", False):
                for callback_name in ("lm_evaluator", "downstream_evaluator"):
                    callback = callbacks.get(callback_name)
                    if callback is not None:
                        callbacks[callback_name] = dataclasses.replace(
                            callback,
                            eval_on_startup=True,
                            cancel_after_first_eval=True,
                        )
            return dataclasses.replace(
                trainer_config,
                callbacks=callbacks,
                load_path=load_path,
                load_trainer_state=(
                    False if getattr(args, "eval_only", False) else trainer_config.load_trainer_state
                ),
                load_optim_state=(
                    False if getattr(args, "eval_only", False) else trainer_config.load_optim_state
                ),
                load_strategy=LoadStrategy.always
                if load_path is not None
                else trainer_config.load_strategy,
            )

        ladder._configure_trainer = MethodType(_configure_trainer_with_eval_options, ladder)
    return ladder


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    """Hook called by ``internal.ladder.main`` to register custom CLI args."""
    parser.add_argument(
        "--sb-table-dir",
        type=str,
        default=DEFAULT_SB_TABLE_DIR,
        help=(
            "Pilot directory containing counts_index/ produced by "
            "data_gen/build_counts_index.py at format_version=2. "
            "Defaults to the 1e-3 pilot."
        ),
    )
    parser.add_argument(
        "--poe-lambda",
        type=float,
        default=DEFAULT_POE_LAMBDA,
        help=(
            "Ngram mixing weight in the PoE joint log-prob, or the initialization "
            "when --learn-poe-lambda is set."
        ),
    )
    parser.add_argument(
        "--learn-poe-lambda",
        action="store_true",
        help=(
            "Learn a positive SB PoE mixing weight initialized from --poe-lambda. "
            "The train module optimizes log(lambda), so the effective lambda stays positive."
        ),
    )
    parser.add_argument(
        "--poe-lambda-lr",
        type=float,
        default=None,
        help=(
            "Optional learning-rate override for learned lambda. If unset, lambda uses "
            "the model LR schedule in a zero-weight-decay optimizer group."
        ),
    )
    parser.add_argument(
        "--sb-alpha",
        type=float,
        default=DEFAULT_SB_ALPHA,
        help="Stupid-backoff discount α per Brants et al 2007.",
    )
    parser.add_argument(
        "--sb-n-max",
        type=int,
        default=DEFAULT_SB_N_MAX,
        help="Highest ngram order to consult. Must equal the order the counts_index was built with.",
    )
    parser.add_argument(
        "--sb-max-order2-continuations",
        type=int,
        default=DEFAULT_SB_MAX_ORDER2_CONTINUATIONS,
        help=(
            "Optional hard cap on order-2 continuations per one-token history. "
            "Keeps the highest-count continuations and lets omitted tokens fall "
            "through to the unigram floor. Orders 3+ remain exact."
        ),
    )
    parser.add_argument(
        "--sb-max-order-continuations",
        action="append",
        default=None,
        metavar="ORDER=CAP",
        help=(
            "Optional hard cap for any SB order, repeatable, for example "
            "'--sb-max-order-continuations 2=128 --sb-max-order-continuations 3=128'. "
            "The legacy --sb-max-order2-continuations flag still sets the order-2 cap."
        ),
    )
    parser.add_argument(
        "--sb-topk-uniform-residual-k",
        type=int,
        default=None,
        help=(
            "Use a KN-top-K-style SB PoE bias: for each position, select the "
            "highest matching SB history row, keep its top-K continuations by "
            "count, and emit logit deltas relative to a uniform residual over "
            "the rest of the vocab. This bypasses the dense unigram floor."
        ),
    )
    parser.add_argument(
        "--sb-min-order-count",
        action="append",
        default=None,
        metavar="ORDER=COUNT",
        help=(
            "Optional minimum raw-count threshold for any SB order, repeatable, "
            "for example '--sb-min-order-count 2=5 --sb-min-order-count 3=2'. "
            "Continuations below the threshold are omitted and fall through to "
            "lower orders or the unigram floor. This is a post-hoc pruning "
            "approximation for threshold sweeps before rebuilding counts."
        ),
    )
    parser.add_argument(
        "--sb-min-order-count-ratio",
        action="append",
        default=None,
        metavar="ORDER=RATIO",
        help=(
            "Derive a minimum raw-count threshold from the selected order's "
            "count statistic, repeatable, for example "
            "'--sb-min-order-count-ratio 2=3 --sb-min-order-count-ratio 3=4'. "
            "The concrete threshold is ceil(RATIO * statistic). This makes "
            "pruning scale with the n-gram fitting sample size instead of "
            "hard-coding one absolute count for every index."
        ),
    )
    parser.add_argument(
        "--sb-min-order-count-ratio-stat",
        choices=("mean", "median"),
        default="mean",
        help=(
            "Statistic used by --sb-min-order-count-ratio. 'mean' is an exact "
            "sequential scan over counts. 'median' uses an evenly-spaced sample "
            "when an order has more rows than --sb-min-order-count-ratio-median-samples."
        ),
    )
    parser.add_argument(
        "--sb-min-order-count-ratio-chunk-entries",
        type=int,
        default=10_000_000,
        help="Count entries to read per chunk when computing exact per-order means.",
    )
    parser.add_argument(
        "--sb-min-order-count-ratio-median-samples",
        type=int,
        default=2_000_000,
        help=(
            "Maximum evenly-spaced count samples used for median-derived "
            "thresholds. Ignored for mean-derived thresholds."
        ),
    )
    parser.add_argument(
        "--sb-index-access",
        choices=("mmap", "pread"),
        default="mmap",
        help=(
            "SB counts-index access backend. 'mmap' is the existing memmap path. "
            "'pread' keeps histories mmap-backed for vectorized search, but uses "
            "explicit os.pread calls for offsets, continuation, count, and "
            "history-total arrays."
        ),
    )
    parser.add_argument(
        "--no-sb-mirror-to-shm",
        dest="sb_mirror_to_shm",
        action="store_false",
        help=(
            "Open SB index files directly from Weka instead of first copying "
            "them to /dev/shm. Required for larger indexes that do not fit in "
            "the node shared-memory allocation."
        ),
    )
    parser.set_defaults(sb_mirror_to_shm=True)
    parser.add_argument(
        "--sb-lookup-threads",
        type=int,
        default=DEFAULT_SB_LOOKUP_THREADS,
        help=(
            "In-process threads per SB reader for splitting one long sequence "
            "lookup into chunks. This complements --data-loader-workers: "
            "workers prepare future batches, while lookup threads reduce the "
            "latency of one training-data batch. Eval uses the same value "
            "unless --sb-eval-lookup-threads is set."
        ),
    )
    parser.add_argument(
        "--sb-eval-lookup-threads",
        type=int,
        default=None,
        help=(
            "Eval-only SB lookup thread count. Use this to keep dataloader "
            "workers at a lower thread count while letting inline LM eval use "
            "more threads for high-override batches."
        ),
    )
    parser.add_argument(
        "--data-loader-workers",
        type=int,
        default=DEFAULT_DATA_LOADER_WORKERS,
        help=(
            "Number of PyTorch data-loader worker processes per rank. SB PoE "
            "override construction is CPU-heavy; worker sweeps on the 1e-4 "
            "index showed useful scaling well above the old default of 4."
        ),
    )
    parser.add_argument(
        "--data-loader-prefetch-factor",
        type=int,
        default=DEFAULT_DATA_LOADER_PREFETCH_FACTOR,
        help=(
            "Number of batches prefetched per data-loader worker. Use 1 for "
            "high-worker SB PoE tests to limit queued million-row override "
            "arrays. Leave unset for PyTorch's default behavior."
        ),
    )
    parser.add_argument(
        "--smoke-source-instances",
        type=int,
        default=DEFAULT_SMOKE_SOURCE_INSTANCES,
        help=(
            "Limit the training source to the first N sequence instances. "
            "Use only for smoke tests; it avoids writing a full pretraining "
            "epoch order file before a short run."
        ),
    )
    parser.add_argument(
        "--smoke-target-batch-size",
        type=int,
        default=16384,
        help=(
            "Global target batch size in tokens for --smoke-1gpu training. "
            "Use --eval-rank-mbz to change LM eval batch size separately."
        ),
    )
    parser.add_argument(
        "--eval-rank-mbz",
        type=int,
        default=None,
        help=(
            "Eval microbatch size in full sequences per data-parallel rank. "
            "Defaults to --rank-mbz. Larger values reduce LM eval batches but "
            "increase per-GPU eval memory."
        ),
    )
    parser.add_argument(
        "--smoke-1gpu",
        action="store_true",
        help=(
            "Run on a single GPU for fast end-to-end smoke testing. "
            "Pair with --chinchilla-multiple ~0.001."
        ),
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help=(
            "Optional checkpoint path or checkpoint folder to load before running. "
            "Useful with --eval-only to benchmark eval from an existing smoke checkpoint."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Run startup LM + downstream evals from the loaded checkpoint and then cancel "
            "before training. Intended for checkpoint-based eval-path smoke tests."
        ),
    )
    parser.add_argument(
        "--lm-eval-only",
        action="store_true",
        help=(
            "Disable downstream in-loop evals while keeping LM perplexity evals. "
            "Useful when downstream eval is too slow for SB PoE training runs."
        ),
    )
    parser.add_argument(
        "--disable-evals",
        action="store_true",
        help="Disable all in-loop eval callbacks. Useful for isolating train throughput.",
    )
    parser.add_argument(
        "--final-eval-only",
        action="store_true",
        help=(
            "Disable fixed-step evals and run evals only at training finish. "
            "Useful for short smoke runs where checkpoint fixed steps would duplicate eval."
        ),
    )
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=None,
        help=(
            "Limit each evaluator to this many batches. Useful for eval throughput "
            "benchmarks from a checkpoint."
        ),
    )
    parser.add_argument(
        "--sb-eval-timing",
        action="store_true",
        help="Enable per-call SB eval timing logs via OLMO_SB_EVAL_TIMING=1.",
    )


if __name__ == "__main__":
    main(configure_ladder, add_additional_args=add_additional_args)
