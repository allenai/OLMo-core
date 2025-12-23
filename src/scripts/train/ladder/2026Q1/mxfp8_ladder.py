import argparse
import logging
import math

import olmo_core.io as io
from olmo_core.config import DType
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOMXLinearConfig, Float8Config
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import *
from olmo_core.model_ladder.utils import format_count
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import OptimConfig, Scheduler
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import warn_once

log = logging.getLogger(__name__)


class MXFP8TransformerModelConfigurator(TransformerModelConfigurator):
    def configure_model(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        tokenizer: TokenizerConfig,
        device_type: str,
    ) -> TransformerConfig:
        # TODO: configure context-parallelism if needed.
        device_type = device_type.lower()
        assert "h100" in device_type or "b200" in device_type
        assert sequence_length in {2048, 4096, 8192}
        size_spec = TransformerSize(size_spec)
        vocab_size = tokenizer.padded_vocab_size()
        kwargs = dict(attn_backend=AttentionBackendName.flash_2)

        model: TransformerConfig
        if size_spec == TransformerSize.size_190M:
            model = TransformerConfig.olmo3_190M(vocab_size, **kwargs)
        elif size_spec == TransformerSize.size_370M:
            model = TransformerConfig.olmo3_370M(vocab_size, **kwargs)
        elif size_spec == TransformerSize.size_600M:
            model = TransformerConfig.olmo3_600M(vocab_size, **kwargs)
        elif size_spec == TransformerSize.size_760M:
            model = TransformerConfig.olmo3_760M(vocab_size, **kwargs)
        elif size_spec == TransformerSize.size_1B:
            model = TransformerConfig.olmo3_1B(vocab_size, **kwargs)
        elif size_spec == TransformerSize.size_3B:
            model = TransformerConfig.olmo3_3B(vocab_size, **kwargs)
        elif size_spec == TransformerSize.size_7B:
            model = TransformerConfig.olmo3_7B(vocab_size, **kwargs)
        elif size_spec == TransformerSize.size_13B:
            model = TransformerConfig.olmo3_13B(vocab_size, **kwargs)
        else:
            raise OLMoConfigurationError(f"Unsupported model size '{size_spec}'")

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
        # TODO: configure context-parallelism if needed.
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
            z_loss_multiplier=1e-5,
            max_grad_norm=1.0,
            scheduler=scheduler,
            float8_config=Float8Config(
                enabled=True,
                ao_mx=AOMXLinearConfig.mxfp8_cublas_rceil(),  # <- this is the intervention
            ),
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
                    mix=DataMix.OLMo_mix_0925,
                    mix_base_dir=get_root_dir(args.cluster),
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
        model_configurator=MXFP8TransformerModelConfigurator(),
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


if __name__ == "__main__":
    main(configure_ladder)
