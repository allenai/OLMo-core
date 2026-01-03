import argparse
import logging
from dataclasses import dataclass

import olmo_core.io as io
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import *
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import *
from olmo_core.model_ladder.utils import get_mix_base_dir
from olmo_core.optim.muon import MuonConfig

log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MuonWSDSChinchillaRunConfigurator(WSDSChinchillaRunConfigurator):
    def configure_optimizer(self, num_params: int, batch_size: int) -> MuonConfig:
        del batch_size  # unused
        # Calculate LR according to https://api.semanticscholar.org/CorpusID:270764838
        # but divide by 2 for WSD schedule (seems to work emperically).
        lr = 0.0047 * (num_params / 108_000_000) ** (-1 / 3)
        lr /= 2.0
        return MuonConfig(lr=lr, weight_decay=0.1, adjust_lr="rms_norm", use_triton=True)


class MuonLadder(ModelLadder):
    def _configure_batch_size_and_num_devices(
        self, size_spec: str, num_params: int
    ) -> tuple[int, int, int, int]:
        # Configure global batch size and device micro-batch size.
        global_batch_size = self.run_configurator.configure_target_batch_size(num_params)
        rank_microbatch_size = self.model_configurator.configure_rank_microbatch_size(
            size_spec=size_spec,
            sequence_length=self.sequence_length,
            device_type=self.device_type,
        )
        rank_microbatch_size = min(rank_microbatch_size, global_batch_size)

        # Configure minimal device mesh spec, i.e. the minimum number of devices needed and the
        # corresponding minimum data parallel world size.
        (
            min_world_size,
            min_dp_world_size,
        ) = self.model_configurator.configure_minimal_device_mesh_spec(
            size_spec=size_spec,
            sequence_length=self.sequence_length,
            device_type=self.device_type,
        )
        if min_dp_world_size is None:
            min_dp_world_size = min_world_size
        if min_world_size % min_dp_world_size != 0:
            raise OLMoConfigurationError(
                f"Invalid device mesh spec for model of size '{size_spec}': "
                f"minimum world size {min_world_size} is not divisible by "
                f"the minimum data parallel world size {min_dp_world_size}."
            )
        if self.max_devices < min_world_size:
            raise OLMoConfigurationError(
                f"Not enough devices ({self.max_devices}) to run model of size '{size_spec}' "
                f"which requires at least {min_world_size} devices."
            )

        # And from that we adjust the global batch size to be a multiple of
        # `rank_microbatch_size x min_dp_world_size`.
        gbz_factor = rank_microbatch_size * min_dp_world_size
        global_batch_size = max(1, round(global_batch_size / gbz_factor)) * gbz_factor

        # Then we can determine the actual number of devices to allocate to the run. In particular
        # we can expand `min_world_size` up to the number of devices available (`self.max_devices`)
        # by a factor that's just the number of gradient accumulation steps needed with the minimum
        # requested number of devices.
        max_num_grad_accum_steps = global_batch_size // gbz_factor
        expansion_factor = min(self.max_devices // min_world_size, max_num_grad_accum_steps)
        num_devices = min_world_size * expansion_factor
        # Ensure num_devices is a power of 2 by rounding down to the nearest power of 2
        num_devices = 2 ** (num_devices.bit_length() - 1) if num_devices > 0 else 1
        dp_world_size = min_dp_world_size * expansion_factor

        # Finally we ensure `global_batch_size` is divisible by the micro-batch size.
        microbatch_size = rank_microbatch_size * dp_world_size
        global_batch_size = max(1, round(global_batch_size / microbatch_size)) * microbatch_size

        return global_batch_size, rank_microbatch_size, num_devices, dp_world_size


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources: list[InstanceSourceConfig] = [
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceMixConfig(
                    tokenizer=tokenizer,
                    mix=DataMix.OLMo_mix_0925_official
                    if args.cluster == "lambda"
                    else DataMix.OLMo_mix_0925,
                    mix_base_dir=get_mix_base_dir(args.cluster),
                )
            ],
            sequence_length=args.sequence_length,
        ),
    ]
    ladder = MuonLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=list(TransformerSize),
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=TransformerModelConfigurator(),
        run_configurator=MuonWSDSChinchillaRunConfigurator(
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
