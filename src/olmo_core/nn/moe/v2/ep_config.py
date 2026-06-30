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
    rowwise_wave = "rowwise_wave"


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

    rowwise_nblocks: int = 32
    share_dispatch_out: bool = False
    share_combine_out: bool = False
    restore_unpermute_backend: str = "te_fused"
    rowwise_symm_dispatch_in: Optional[bool] = None
    rowwise_symm_combine_out: Optional[bool] = None
    rowwise_symm_combine_gather: Optional[bool] = None

    rowwise_wave_num_waves: int = 1
    rowwise_wave_mode: str = "expert"
    rowwise_wave_recompute_linear1: bool = False
    rowwise_wave_recompute_act: bool = False
    checkpoint_tbo: bool = False

    def validate(self) -> None:
        self.path = ExpertParallelPath(self.path)
        self.schedule = ExpertParallelSchedule(self.schedule)
        self.restore_unpermute_backend = self.restore_unpermute_backend.lower()
        self.rowwise_wave_mode = self.rowwise_wave_mode.lower()

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
        if self.rowwise_wave_num_waves < 1:
            raise OLMoConfigurationError(
                "EP rowwise_wave_num_waves must be >= 1 "
                f"(got {self.rowwise_wave_num_waves})"
            )
        if self.rowwise_wave_mode != "expert":
            raise OLMoConfigurationError(
                "EP rowwise_wave_mode currently supports only 'expert' "
                f"(got {self.rowwise_wave_mode!r})"
            )
        if (
            self.path != ExpertParallelPath.rowwise_wave
            and self.rowwise_wave_num_waves != 1
        ):
            raise OLMoConfigurationError(
                "EP rowwise_wave_num_waves is only valid with "
                f"path={ExpertParallelPath.rowwise_wave!r} "
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
            ExpertParallelPath.rowwise_wave,
        }

    @property
    def uses_rowwise_buffers(self) -> bool:
        return self.is_rowwise

    @property
    def rowwise_transport(self) -> Optional[str]:
        if self.path in {
            ExpertParallelPath.rowwise_nvshmem,
            ExpertParallelPath.rowwise_wave,
        }:
            return "nvshmem"
        return None
