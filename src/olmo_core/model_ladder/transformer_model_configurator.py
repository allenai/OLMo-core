import dataclasses
import math
import re
from dataclasses import dataclass
from typing import Any

from olmo_core.config import DType, StrEnum
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import OptimConfig, Scheduler
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import warn_once

from .base import DeviceMeshSpec, ModelConfigurator
from .utils import format_count


class TransformerSize(StrEnum):
    size_60M = "60M"
    size_100M = "100M"
    size_190M = "190M"
    size_370M = "370M"
    size_600M = "600M"
    size_760M = "760M"
    size_1B = "1B"
    size_3B = "3B"
    size_7B = "7B"
    size_13B = "13B"

    @property
    def approx_num_params(self) -> int:
        size = self.replace(" ", "").upper()
        if (m := re.match(r"^([\d\.]+)([KMBT])$", size)) is not None:
            value, unit = m.groups()
            multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[unit]
            return int(float(value) * multiplier)
        else:
            raise ValueError(f"Invalid size descriptor '{self}'")

    def __lt__(self, other) -> bool:
        """Less than comparison based on approximate number of parameters."""
        if isinstance(other, TransformerSize):
            return self.approx_num_params < other.approx_num_params
        else:
            return super().__lt__(other)

    def __le__(self, other) -> bool:
        """Less than or equal comparison based on approximate number of parameters."""
        if isinstance(other, TransformerSize):
            return self.approx_num_params <= other.approx_num_params
        else:
            return super().__le__(other)

    def __gt__(self, other) -> bool:
        """Greater than comparison based on approximate number of parameters."""
        if isinstance(other, TransformerSize):
            return self.approx_num_params > other.approx_num_params
        else:
            return super().__gt__(other)

    def __ge__(self, other) -> bool:
        """Greater than or equal comparison based on approximate number of parameters."""
        if isinstance(other, TransformerSize):
            return self.approx_num_params >= other.approx_num_params
        else:
            return super().__ge__(other)


@dataclass(kw_only=True, eq=True)
class TransformerModelConfigurator(ModelConfigurator[TransformerConfig]):
    """
    Generic model configurator for transformer models.
    """

    rank_microbatch_size: int | None = None
    """
    Optional fixed rank micro-batch size. If set, this value is used directly instead of
    computing it based on model size and device type.
    """

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

        # TODO: configure context-parallelism if needed.
        device_type = device_type.lower()
        assert "h100" in device_type or "b200" in device_type
        assert sequence_length in {2048, 4096, 8192}
        size_spec = TransformerSize(size_spec)

        num_params = size_spec.approx_num_params
        mbz: int
        if num_params <= 100e6:
            # mbz * dp_world_size constrains the smallest global batch size we can use.
            # for small models this means that we need to use a smaller mbz than is optimal for
            # throughput / the hardware so that we can get a roughly optimal global batch size.
            # but since the models are so small we can get away with it.
            mbz = 4 * 4096
        elif num_params <= 190e6:
            mbz = 16 * 4096
        elif num_params <= 370e6:
            mbz = 12 * 4096
        elif num_params <= 760e6:
            mbz = 10 * 4096
        elif num_params <= 1e9:
            mbz = 8 * 4096
        elif num_params <= 3e9:
            mbz = 4 * 4096
        elif num_params <= 7e9:
            mbz = 2 * 4096
        else:
            mbz = 2 * 4096

        if "b200" in device_type:
            mbz = mbz * 2  # warning: this can affect the smallest global batch size we can use

        return mbz

    def configure_minimal_device_mesh_spec(
        self,
        *,
        size_spec: str,
        sequence_length: int,
        device_type: str,
    ) -> DeviceMeshSpec:
        # TODO: configure context-parallelism if needed.
        device_type = device_type.lower()
        assert "h100" in device_type or "b200" in device_type
        assert sequence_length in {2048, 4096, 8192}
        size_spec = TransformerSize(size_spec)

        num_params = size_spec.approx_num_params
        if num_params < 13e9:
            return DeviceMeshSpec(world_size=8, dp_world_size=None)
        else:
            return DeviceMeshSpec(world_size=32, dp_world_size=None)

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
        )

        # Build the model.
        model = model_config.build(init_device="meta")

        # Build the train module.
        train_module = train_module_config.build(model)
        assert isinstance(train_module, TransformerTrainModule)

        return train_module


@dataclass(kw_only=True, eq=True)
class Olmo3ModelConfigurator(TransformerModelConfigurator):
    """
    Model configurator for Olmo 3 transformer models.
    """

    model_construction_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """
    Keyword arguments to pass to the model constructor.
    """

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

        attn_backend = AttentionBackendName.torch
        if "h100" in device_type:
            try:
                AttentionBackendName.flash_3.assert_supported()
                attn_backend = AttentionBackendName.flash_3
            except RuntimeError:
                pass
        elif "b200" in device_type:
            try:
                AttentionBackendName.flash_2.assert_supported()
                attn_backend = AttentionBackendName.flash_2
            except RuntimeError:
                pass

        kwargs = dict(attn_backend=attn_backend, **self.model_construction_kwargs)

        model: TransformerConfig
        if size_spec == TransformerSize.size_60M:
            model = TransformerConfig.olmo3_60M(vocab_size, **kwargs)
        elif size_spec == TransformerSize.size_100M:
            model = TransformerConfig.olmo3_100M(vocab_size, **kwargs)
        elif size_spec == TransformerSize.size_190M:
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
