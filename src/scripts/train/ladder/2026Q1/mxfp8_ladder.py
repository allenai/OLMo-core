import argparse
import logging

from olmo_core.config import DType
from olmo_core.data.composable import *
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import AOMXLinearConfig, Float8Config
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import Olmo3ModelConfigurator, TransformerModelConfigurator
from olmo_core.model_ladder.transformer_model_configurator import TransformerSize
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

log = logging.getLogger(__name__)


class MXFP8Olmo3ModelConfigurator(Olmo3ModelConfigurator):
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


def configure_model(args: argparse.Namespace) -> TransformerModelConfigurator:
    return MXFP8Olmo3ModelConfigurator(
        rank_microbatch_size=None
        if args.rank_mbz is None
        else args.rank_mbz * args.sequence_length,
    )


if __name__ == "__main__":
    main(configure_model=configure_model)
