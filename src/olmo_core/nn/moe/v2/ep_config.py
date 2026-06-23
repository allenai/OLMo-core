from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from olmo_core.config import Config, StrEnum
from olmo_core.exceptions import OLMoConfigurationError


class ExpertParallelPath(StrEnum):
    sync_1d = "sync_1d"
    no_sync_1d = "no_sync_1d"
    no_sync_2d_removed = "no_sync_2d_removed"
    rowwise_nvshmem = "rowwise_nvshmem"
    rowwise_tma_ibgda = "rowwise_tma_ibgda"
    wave_mega = "wave_mega"


class ExpertParallelSchedule(StrEnum):
    normal = "normal"
    tbo = "tbo"


@dataclass
class ExpertParallelConfig(Config):
    path: ExpertParallelPath = ExpertParallelPath.sync_1d
    schedule: ExpertParallelSchedule = ExpertParallelSchedule.normal

    capacity_factor: float = 1.25
    shared_slots: int = 1
    major_align: int = 1

    rowwise_nblocks: int = 256
    share_dispatch_out: bool = False
    share_combine_out: bool = False
    restore_unpermute_backend: str = "te_fused"
    rowwise_symm_dispatch_in: Optional[bool] = None
    rowwise_symm_combine_out: Optional[bool] = None
    rowwise_symm_combine_gather: Optional[bool] = None

    tma_ibgda_num_sms: Optional[int] = None
    tma_ibgda_symmetric_expert_out: bool = False
    wave_use_bf16_persistent_mega_forward: bool = False
    checkpoint_tbo: bool = False

    def validate(self) -> None:
        self.path = ExpertParallelPath(self.path)
        self.schedule = ExpertParallelSchedule(self.schedule)
        self.restore_unpermute_backend = self.restore_unpermute_backend.lower()

        if self.path == ExpertParallelPath.no_sync_2d_removed:
            raise OLMoConfigurationError(
                "no_sync_2d_removed is not supported: the 2D all_to_all path was "
                "removed due to correctness/performance issues."
            )
        if self.capacity_factor <= 0:
            raise OLMoConfigurationError(
                f"EP capacity_factor must be > 0 (got {self.capacity_factor})"
            )
        if self.shared_slots < 1:
            raise OLMoConfigurationError(
                f"EP shared_slots must be >= 1 (got {self.shared_slots})"
            )
        if self.major_align < 1:
            raise OLMoConfigurationError(
                f"EP major_align must be >= 1 (got {self.major_align})"
            )
        if self.rowwise_nblocks < 0:
            raise OLMoConfigurationError(
                f"EP rowwise_nblocks must be >= 0 (got {self.rowwise_nblocks})"
            )
        if self.tma_ibgda_num_sms is not None and self.tma_ibgda_num_sms < 32:
            raise OLMoConfigurationError(
                "EP tma_ibgda_num_sms must be >= 32 when set "
                f"(got {self.tma_ibgda_num_sms})"
            )
        if (
            self.tma_ibgda_num_sms is not None
            and self.path != ExpertParallelPath.rowwise_tma_ibgda
        ):
            raise OLMoConfigurationError(
                "EP tma_ibgda_num_sms is only valid with "
                f"path={ExpertParallelPath.rowwise_tma_ibgda!r} "
                f"(got path={self.path!r})"
            )
        if (
            self.tma_ibgda_symmetric_expert_out
            and self.path != ExpertParallelPath.rowwise_tma_ibgda
        ):
            raise OLMoConfigurationError(
                "EP tma_ibgda_symmetric_expert_out=True is only valid with "
                f"path={ExpertParallelPath.rowwise_tma_ibgda!r} "
                f"(got path={self.path!r})"
            )
        if (
            self.wave_use_bf16_persistent_mega_forward
            and self.path != ExpertParallelPath.wave_mega
        ):
            raise OLMoConfigurationError(
                "EP wave_use_bf16_persistent_mega_forward=True is only valid with "
                f"path={ExpertParallelPath.wave_mega!r} "
                f"(got path={self.path!r})"
            )
        if self.restore_unpermute_backend not in ("te_fused", "te_unfused", "cuda"):
            raise OLMoConfigurationError(
                "EP restore_unpermute_backend must be one of "
                "'te_fused'|'te_unfused'|'cuda' "
                f"(got {self.restore_unpermute_backend!r})"
            )
        if (
            self.schedule == ExpertParallelSchedule.tbo
            and self.path != ExpertParallelPath.rowwise_nvshmem
        ):
            raise OLMoConfigurationError(
                "EP schedule='tbo' is only supported with "
                f"path={ExpertParallelPath.rowwise_nvshmem!r} "
                f"(got path={self.path!r})"
            )
        if self.checkpoint_tbo and self.path != ExpertParallelPath.rowwise_nvshmem:
            raise OLMoConfigurationError(
                "EP checkpoint_tbo=True is only supported with "
                f"path={ExpertParallelPath.rowwise_nvshmem!r} "
                f"(got path={self.path!r})"
            )

    @property
    def no_sync(self) -> bool:
        return self.path != ExpertParallelPath.sync_1d

    @property
    def is_no_sync_1d(self) -> bool:
        return self.path == ExpertParallelPath.no_sync_1d

    @property
    def is_rowwise(self) -> bool:
        return self.path in {
            ExpertParallelPath.rowwise_nvshmem,
            ExpertParallelPath.rowwise_tma_ibgda,
        }

    @property
    def is_wave(self) -> bool:
        return self.path == ExpertParallelPath.wave_mega

    @property
    def uses_rowwise_buffers(self) -> bool:
        # The current wave/Mega path still reuses rowwise routing and symmetric
        # scratch plumbing, even though it is not a rowwise transport backend.
        return self.is_rowwise or self.is_wave

    @property
    def rowwise_transport(self) -> Optional[str]:
        if self.path == ExpertParallelPath.rowwise_nvshmem:
            return "nvshmem"
        if self.path == ExpertParallelPath.rowwise_tma_ibgda:
            return "tma_ibgda"
        return None
