"""Launch StateBench direct-training experiments on the hybrid-small-suite 275M architecture.

Each run trains one model variant on one StateBench training distribution
(``integer-code--r-trivial``, ``integer-code--aperiodic``, or ``integer-code--periodic``),
repeating the distribution as necessary to fill the size-specific Chinchilla budget
(Cx1 by default).

Model variants share the hybrid-small-suite 275M backbone (d_model=640, 10 layers,
peri-norm blocks, gated attention with head QK-norm, embed scale + embedding norm) and
differ only in their sequence mixers:

- ``transformer-rope``: full attention on every layer, RoPE.
- ``transformer-nope``: full attention on every layer, no positional embeddings.
- ``hybrid``: GDN on every layer except each 5th, which is global NoPE attention
  (exactly the hybrid-small-suite 275M model).
- ``gdn-sdp``: GDN on every layer with ``allow_neg_eigval=False``.
- ``gdn-full``: GDN on every layer with ``allow_neg_eigval=True``.

Typical usage:

    uv run src/scripts/train/ladder/state_bench.py launch \\
      --size 275M --model-type hybrid --distribution r-trivial \\
      --init-seed 0 --max-gpus 8

Omitting ``--model-type`` and/or ``--distribution`` with ``launch`` expands the
omitted axes into the full suite.
"""

import argparse
import math
import sys
from dataclasses import dataclass

from olmo_core.config import DType, StrEnum
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    InstanceFilterConfig,
    NumpyDocumentSourceConfig,
    SamplingInstanceSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import get_gpu_type
from olmo_core.internal.ladder import _launch_run, configure_launcher, get_requested_sizes, main
from olmo_core.io import join_path
from olmo_core.model_ladder import (
    DeviceMeshSpec,
    ModelLadder,
    TransformerModelConfigurator,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GatedDeltaNetConfig,
    GateGranularity,
)
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.rope import RoPEConfig
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerBlockType, TransformerConfig
from olmo_core.optim import OptimConfig, Scheduler
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)

WEKA_ROOT = "/weka/oe-training-default/ai2-llm"

STATE_BENCH_DATA_ROOT = (
    "/weka/oe-training-default/jacksonp/state-bench/data/"
    "state-tracking-long-context-v1/rendered-tokenized/tokens"
)
STATE_BENCH_DISTRIBUTION_TOKENS = {
    "integer-code--r-trivial": 17_176_160_694,
    "integer-code--aperiodic": 17_176_160_694,
    "integer-code--periodic": 15_431_502_989,
}
STATE_BENCH_DISTRIBUTION_ALIASES = {
    "r-trivial": "integer-code--r-trivial",
    "aperiodic": "integer-code--aperiodic",
    "periodic": "integer-code--periodic",
}

MAX_WANDB_TAG_LENGTH = 64


class StateBenchSize(StrEnum):
    size_275M = "275M"


class StateBenchModelType(StrEnum):
    transformer_rope = "transformer-rope"
    transformer_nope = "transformer-nope"
    hybrid = "hybrid"
    gdn_sdp = "gdn-sdp"
    gdn_full = "gdn-full"


def _format_chinchilla_multiple(chinchilla_multiple: float) -> str:
    return f"{chinchilla_multiple:g}"


def _wandb_tags(*tags: str) -> list[str]:
    return [tag[:MAX_WANDB_TAG_LENGTH] for tag in tags]


def _root_dir(cluster: str) -> str:
    if cluster.startswith("ai2/"):
        return WEKA_ROOT
    return "gs://ai2-llm"


@dataclass(kw_only=True, eq=True)
class StateBenchModelConfigurator(TransformerModelConfigurator):
    """
    Configure the hybrid-small-suite 275M architecture in one of its StateBench variants.

    See the module docstring for a description of the variants. Everything except the
    sequence mixers is shared across variants and matches
    ``src/scripts/train/hybrid-small-suite/arch.py``.
    """

    model_type: str
    init_seed: int = 0

    def configure_model(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        tokenizer: TokenizerConfig,
        device_type: str,
    ) -> TransformerConfig:
        size_spec = StateBenchSize(size_spec)
        assert size_spec == StateBenchSize.size_275M
        model_type = StateBenchModelType(self.model_type)

        d_model = 640
        hidden_size = 640 * 8
        n_layers = 10
        n_heads = 8
        n_kv_heads = 8
        head_dim = 128
        global_layer_interval = 5
        dtype = DType.float32
        expand_v = 2.0

        layer_norm = LayerNormConfig(
            name=LayerNormType.rms,
            eps=1e-6,
            bias=False,
            dtype=dtype,
        )
        feed_forward = FeedForwardConfig(
            hidden_size=hidden_size,
            bias=False,
            dtype=dtype,
            activation=ActivationFunction.silu,
        )

        def attention_block(rope: RoPEConfig | None) -> TransformerBlockConfig:
            return TransformerBlockConfig(
                name=TransformerBlockType.peri_norm,
                sequence_mixer=AttentionConfig(
                    name=AttentionType.default,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    bias=False,
                    rope=rope,
                    gate=GateConfig(
                        granularity=GateGranularity.elementwise,
                        full_precision=True,
                    ),
                    qk_norm=layer_norm,
                    use_head_qk_norm=True,
                    backend=self._attn_backend(device_type),
                    dtype=dtype,
                ),
                feed_forward=feed_forward,
                layer_norm=layer_norm,
            )

        def gdn_block(allow_neg_eigval: bool) -> TransformerBlockConfig:
            return TransformerBlockConfig(
                name=TransformerBlockType.peri_norm,
                sequence_mixer=GatedDeltaNetConfig(
                    n_heads=n_heads,
                    n_v_heads=n_heads,
                    head_dim=head_dim,
                    expand_v=expand_v,
                    allow_neg_eigval=allow_neg_eigval,
                    dtype=dtype,
                ),
                feed_forward=feed_forward,
                layer_norm=layer_norm,
            )

        block: TransformerBlockConfig
        block_overrides: dict[int, TransformerBlockConfig] | None = None
        if model_type == StateBenchModelType.transformer_rope:
            block = attention_block(RoPEConfig())
        elif model_type == StateBenchModelType.transformer_nope:
            block = attention_block(None)
        elif model_type == StateBenchModelType.hybrid:
            block = gdn_block(allow_neg_eigval=True)
            block_overrides = {
                layer_idx: attention_block(None)
                for layer_idx in range(n_layers)
                if layer_idx % global_layer_interval == (global_layer_interval - 1)
            }
        elif model_type == StateBenchModelType.gdn_sdp:
            block = gdn_block(allow_neg_eigval=False)
        elif model_type == StateBenchModelType.gdn_full:
            block = gdn_block(allow_neg_eigval=True)
        else:
            raise OLMoConfigurationError(f"Unknown model type '{model_type}'")

        return TransformerConfig(
            d_model=d_model,
            vocab_size=tokenizer.padded_vocab_size(),
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(
                loss_implementation=LMLossImplementation.default,
                layer_norm=layer_norm,
                bias=False,
                dtype=dtype,
            ),
            dtype=dtype,
            block_overrides=block_overrides,
            embed_scale=math.sqrt(d_model),
            embedding_norm=LayerNormConfig(
                name=LayerNormType.rms,
                eps=1e-6,
                bias=False,
            ),
            init_seed=self.init_seed,
        )

    def _attn_backend(self, device_type: str) -> AttentionBackendName:
        if "h100" in device_type.lower():
            try:
                AttentionBackendName.flash_3.assert_supported()
                return AttentionBackendName.flash_3
            except RuntimeError:
                pass
        elif "b200" in device_type.lower():
            try:
                AttentionBackendName.flash_4.assert_supported()
                return AttentionBackendName.flash_4
            except RuntimeError:
                pass
        return AttentionBackendName.torch

    def configure_rank_microbatch_size(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        device_type: str,
    ) -> int:
        if self.rank_microbatch_size is not None:
            assert self.rank_microbatch_size > 0
            assert self.rank_microbatch_size % sequence_length == 0
            return self.rank_microbatch_size
        # 2 sequences per rank keeps the realized global batch size within ~2% of the WSDS
        # target for the 275M class (5 sequences, the hybrid-small-suite throughput setting,
        # overshoots the target by ~28% since the batch must be a multiple of mbz x dp).
        return 2 * sequence_length

    def configure_minimal_device_mesh_spec(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        device_type: str,
    ) -> DeviceMeshSpec:
        return DeviceMeshSpec(world_size=8, dp_world_size=None)

    def build_train_module(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        rank_microbatch_size: int,
        model_config: TransformerConfig,
        optim_config: OptimConfig,
        scheduler: Scheduler,
        device_type: str,
    ) -> TransformerTrainModule:
        # Same as TransformerModelConfigurator.build_train_module, minus the coercion of
        # size_spec to TransformerSize (which doesn't contain our sizes).
        train_module_config = TransformerTrainModuleConfig(
            rank_microbatch_size=rank_microbatch_size,
            max_sequence_length=sequence_length,
            optim=optim_config,
            compile_model=True,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.fsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            ),
            z_loss_multiplier=1e-5,
            max_grad_norm=1.0,
            scheduler=scheduler,
        )
        model = model_config.build(init_device="meta")
        train_module = train_module_config.build(model)
        assert isinstance(train_module, TransformerTrainModule)
        return train_module


def _source_paths(args: argparse.Namespace, distribution: str) -> list[str]:
    """Return the token shard glob for one StateBench training distribution."""
    return [str(join_path(args.state_bench_data_root, distribution, "train", "*.npy"))]


def _state_bench_source(
    args: argparse.Namespace, tokenizer: TokenizerConfig, distribution: str
) -> ConcatAndChunkInstanceSourceConfig:
    """Configure one independently sampled StateBench training distribution."""
    return ConcatAndChunkInstanceSourceConfig(
        sources=[
            NumpyDocumentSourceConfig(
                source_paths=_source_paths(args, distribution),
                tokenizer=tokenizer,
                expand_glob=True,
                source_group_size=-1,
                label=distribution,
            )
        ],
        sequence_length=args.sequence_length,
        label=distribution,
    )


def _model_configurator(args: argparse.Namespace) -> StateBenchModelConfigurator:
    return StateBenchModelConfigurator(
        model_type=str(args.model_type),
        init_seed=args.init_seed,
        rank_microbatch_size=None
        if args.rank_mbz is None
        else args.rank_mbz * args.sequence_length,
    )


@dataclass(kw_only=True)
class StateBenchLadder(ModelLadder):
    """Ladder recipe for StateBench direct-training experiments."""

    model_type: str
    distribution: str
    state_bench_tokens: int
    training_tokens: int
    chinchilla_multiple: float
    init_seed: int

    def get_save_folder(self, size_spec: str) -> str:
        return str(
            join_path(
                self.dir,
                size_spec,
                self.model_type,
                self.distribution,
                f"Cx{_format_chinchilla_multiple(self.chinchilla_multiple)}",
                f"init_seed{self.init_seed}",
            )
        )

    def _configure_trainer(self, size_spec: str, for_benchmarking: bool = False):
        config = super()._configure_trainer(size_spec, for_benchmarking=for_benchmarking)
        run_name = (
            f"{size_spec}/{self.model_type}/{self.distribution}/"
            f"Cx{_format_chinchilla_multiple(self.chinchilla_multiple)}/"
            f"init_seed{self.init_seed}"
        )
        if "wandb" in config.callbacks:
            config.callbacks["wandb"].name = run_name  # type: ignore[attr-defined]
            config.callbacks["wandb"].project = self.project or self.name  # type: ignore[attr-defined]
            config.callbacks["wandb"].group = f"{self.name}/{size_spec}/{self.model_type}"  # type: ignore[attr-defined]
            config.callbacks["wandb"].tags = _wandb_tags(  # type: ignore[attr-defined]
                f"size:{size_spec}",
                f"model_type:{self.model_type}",
                "data:state-bench-only",
                f"state_bench_distribution:{self.distribution}",
                f"state_bench_distribution_tokens:{self.state_bench_tokens}",
                f"chinchilla_multiple:{_format_chinchilla_multiple(self.chinchilla_multiple)}",
                f"init_seed:{self.init_seed}",
                f"training_tokens:{self.training_tokens}",
            )
        if "slack_notifier" in config.callbacks:
            config.callbacks["slack_notifier"].name = run_name  # type: ignore[attr-defined]
        return config


def add_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        cluster="ai2/jupiter",
        workspace="ai2/beyond-state",
        budget="ai2/oe-other",
        priority="urgent",
        chinchilla_multiple=1.0,
        init_seed=0,
    )
    if cmd == "launch":
        parser.set_defaults(func=launch_state_bench)
    parser.add_argument(
        "--model-type",
        choices=list(StateBenchModelType),
        default=None,
        help="Model variant for this condition. Omit with `launch` to launch all variants.",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=0,
        help="Random seed used for model parameter initialization.",
    )
    parser.add_argument(
        "--distribution",
        choices=sorted(STATE_BENCH_DISTRIBUTION_ALIASES),
        default=None,
        help="StateBench training distribution. Omit with `launch` to launch all distributions.",
    )
    parser.add_argument(
        "--state-bench-data-root",
        type=str,
        default=STATE_BENCH_DATA_ROOT,
        help="Directory containing the StateBench distribution directories.",
    )


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    if args.model_type is None or args.distribution is None:
        raise OLMoConfigurationError(
            "Specify both --model-type and --distribution for a single ladder configuration. "
            "The `launch` command expands omitted values into the corresponding suite."
        )

    tokenizer = TokenizerConfig.dolma2()
    distribution = STATE_BENCH_DISTRIBUTION_ALIASES[args.distribution]
    sizes = get_requested_sizes(args)
    size_for_duration = sizes[0]
    run_configurator = WSDSChinchillaRunConfigurator(
        chinchilla_multiple=args.chinchilla_multiple,
        lr_multiplier=args.lr_multiplier,
        stepped_schedule=args.stepped_schedule,
    )
    model_configurator = _model_configurator(args)
    model_config = model_configurator.configure_model(
        size_spec=str(size_for_duration),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        device_type=get_gpu_type(args.cluster),
    )
    draft_ladder = ModelLadder(
        name=args.name,
        project=args.project,
        dir=str(join_path(_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=sizes,
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=model_configurator,
        run_configurator=run_configurator,
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=[_state_bench_source(args, tokenizer, distribution)],
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
    )
    global_batch_size, *_ = draft_ladder._configure_batch_size_and_num_devices(
        str(size_for_duration), model_config.num_non_embedding_params
    )
    training_tokens = run_configurator.configure_duration(
        model_config.num_non_embedding_params, global_batch_size
    ).value

    return StateBenchLadder(
        name=args.name,
        project=args.project,
        dir=str(join_path(_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=sizes,
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=model_configurator,
        run_configurator=run_configurator,
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        # Sampling repeats the selected distribution as needed to reach the Chinchilla budget.
        instance_sources=[
            SamplingInstanceSourceConfig(
                sources=[_state_bench_source(args, tokenizer, distribution)],
                max_tokens=training_tokens,
                label="state-bench",
            )
        ],
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
        model_type=str(args.model_type),
        distribution=distribution,
        state_bench_tokens=STATE_BENCH_DISTRIBUTION_TOKENS[distribution],
        training_tokens=training_tokens,
        chinchilla_multiple=args.chinchilla_multiple,
        init_seed=args.init_seed,
    )


def _condition_argv(model_type: str, distribution: str) -> list[str]:
    """Return the current command line with one concrete StateBench condition."""
    condition_flags = {"--model-type", "--distribution"}
    resolved_argv = [sys.argv[0]]
    index = 1
    while index < len(sys.argv):
        argument = sys.argv[index]
        if argument in condition_flags:
            index += 2
        elif any(argument.startswith(f"{flag}=") for flag in condition_flags):
            index += 1
        else:
            resolved_argv.append(argument)
            index += 1
    return [
        *resolved_argv,
        "--model-type",
        model_type,
        "--distribution",
        distribution,
    ]


def launch_state_bench(args: argparse.Namespace) -> None:
    """Launch every StateBench condition selected by the optional suite filters."""
    from olmo_core.utils import prepare_cli_environment

    prepare_cli_environment()
    model_types = [args.model_type] if args.model_type is not None else list(StateBenchModelType)
    distributions = (
        [args.distribution]
        if args.distribution is not None
        else list(STATE_BENCH_DISTRIBUTION_ALIASES)
    )
    suite_size = len(model_types) * len(distributions)

    for model_type in model_types:
        for distribution in distributions:
            args.model_type = model_type
            args.distribution = distribution
            ladder = configure_ladder(args)

            # ``configure_launcher`` builds the command from ``sys.argv``. Add the resolved
            # suite condition so the Beaker job's ``run`` command is always concrete.
            original_argv = sys.argv
            sys.argv = _condition_argv(str(model_type), distribution)
            try:
                launcher = configure_launcher(args, ladder, "run")
            finally:
                sys.argv = original_argv
            if suite_size > 1:
                # The standard launcher enables a log-following soft timeout. A suite must
                # submit every condition without following the first job, so disable that
                # follow-only timeout for its individual submissions.
                launcher.step_timeout = None
                launcher.step_soft_timeout = None

            _launch_run(
                ladder,
                launcher,
                args.size_enum(args.size),
                # Following a multi-condition suite would block before later jobs launch.
                follow=args.follow if suite_size == 1 else False,
                slack_notifications=args.slack_notifications,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main(
        configure_ladder=configure_ladder,
        size_enum=StateBenchSize,
        default_name="state-bench",
        add_additional_args=add_args,
    )
