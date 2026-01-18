import argparse
import logging
import math

import olmo_core.io as io
from olmo_core.config import DType
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    InstanceFilterConfig,
    InstanceSourceConfig,
    NumpyDocumentSourceMixConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    ModelLadder,
    TransformerModelConfigurator,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.model_ladder.utils import format_count
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.fla.layer import FLAConfig
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerBlockType,
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import OptimConfig, Scheduler
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import ensure_multiple_of, warn_once

log = logging.getLogger(__name__)


def get_mix_base_dir(cluster: str) -> str:
    if cluster == "lambda":
        return "/data/caia-mltrain/data/"
    else:
        return "gs://ai2-llm/"


class GatedDeltaNetModelConfigurator(TransformerModelConfigurator):
    """
    Model configurator for pure GatedDeltaNet.

    Args:
        use_gate: Whether to use output gating in GatedDeltaNet. Default: True.
        allow_neg_eigval: Whether to allow negative eigenvalues. Default: True.
    """

    def __init__(self, use_gate: bool = True, allow_neg_eigval: bool = True):
        super().__init__()
        self.use_gate = use_gate
        self.allow_neg_eigval = allow_neg_eigval

    def configure_model(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        tokenizer: TokenizerConfig,
        device_type: str,
    ) -> TransformerConfig:
        device_type = device_type.lower()
        assert "h100" in device_type or "b200" in device_type
        assert sequence_length in {2048, 4096, 8192}
        size_spec = TransformerSize(size_spec)
        vocab_size = tokenizer.padded_vocab_size()

        model: TransformerConfig
        if size_spec == TransformerSize.size_60M:
            model = TransformerConfig.olmo3_60M(vocab_size)
        elif size_spec == TransformerSize.size_100M:
            model = TransformerConfig.olmo3_100M(vocab_size)
        elif size_spec == TransformerSize.size_190M:
            model = TransformerConfig.olmo3_190M(vocab_size)
        elif size_spec == TransformerSize.size_370M:
            model = TransformerConfig.olmo3_370M(vocab_size)
        elif size_spec == TransformerSize.size_600M:
            model = TransformerConfig.olmo3_600M(vocab_size)
        elif size_spec == TransformerSize.size_760M:
            model = TransformerConfig.olmo3_760M(vocab_size)
        elif size_spec == TransformerSize.size_1B:
            model = TransformerConfig.olmo3_1B(vocab_size)
        elif size_spec == TransformerSize.size_3B:
            model = TransformerConfig.olmo3_3B(vocab_size)
        elif size_spec == TransformerSize.size_7B:
            model = TransformerConfig.olmo3_7B(vocab_size)
        elif size_spec == TransformerSize.size_13B:
            model = TransformerConfig.olmo3_13B(vocab_size)
        else:
            raise OLMoConfigurationError(f"Unsupported model size '{size_spec}'")

        # Convert to pure FLA (GatedDeltaNet) block
        model.block.name = TransformerBlockType.fla
        model.block.attention = AttentionConfig(n_heads=model.block.attention.n_heads)

        # Configure GatedDeltaNet
        # FLA repo recommends: num_heads * head_dim = 0.75 * hidden_size
        model.block.fla = FLAConfig(
            name="GatedDeltaNet",
            dtype=model.dtype,
            fla_layer_kwargs={
                "head_dim": ensure_multiple_of(
                    int(0.75 * model.d_model / model.block.attention.n_heads), 128
                ),
                "use_gate": self.use_gate,
                "allow_neg_eigval": self.allow_neg_eigval,
            },
        )

        # Make sure actual number of params is close to target number.
        if (
            pct_diff := (
                math.fabs(model.num_non_embedding_params - size_spec.approx_num_params)
                / size_spec.approx_num_params
            )
        ) > 0.05:
            warn_once(
                f"Configured model has {format_count(model.num_non_embedding_params)} (non-embedding) parameters, "
                f"which differs from target of {size_spec} by ~{100 * pct_diff:.1f}%.",
                UserWarning,
            )
        return model

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
        device_type = device_type.lower()
        assert "h100" in device_type or "b200" in device_type
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
            ac_config=TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.budget, activation_memory_budget=0.5
            ),
            z_loss_multiplier=1e-5,
            max_grad_norm=1.0,
            scheduler=scheduler,
        )

        # Build the model.
        model = model_config.build(init_device="meta")

        # Build the train module.
        train_module = train_module_config.build(model)
        assert isinstance(train_module, TransformerTrainModule)

        return train_module


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceMixConfig(
                    tokenizer=tokenizer,
                    mix=(
                        DataMix.OLMo_mix_0925_official
                        if args.cluster == "lambda"
                        else DataMix.OLMo_mix_0925
                    ),
                    mix_base_dir=get_mix_base_dir(args.cluster),
                )
            ],
            sequence_length=args.sequence_length,
        ),
    ]
    ladder = ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=list(TransformerSize),
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=GatedDeltaNetModelConfigurator(
            use_gate=args.use_gate,
            allow_neg_eigval=args.allow_neg_eigval,
        ),
        run_configurator=WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
    )
    return ladder


def add_gdn_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    """Add GatedDeltaNet-specific arguments to the parser."""
    del cmd  # unused, but required by the callback signature
    parser.add_argument(
        "--use-gate",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Whether to use output gating in GatedDeltaNet (default: True)",
    )
    parser.add_argument(
        "--allow-neg-eigval",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Whether to allow negative eigenvalues (default: False)",
    )


if __name__ == "__main__":
    main(configure_ladder, add_additional_args=add_gdn_args)
