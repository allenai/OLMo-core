from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from olmo_core.config import Config, StrEnum
from olmo_core.exceptions import OLMoConfigurationError


class ExpertParallelPath(StrEnum):
    # Synchronized all-to-all EP. Does not require symmetric memory.
    sync_1d = "sync_1d"
    # Legacy no-sync EP over symmetric-memory all_to_all_vdev primitives. It
    # still pays permute/unpermute overhead, so rowwise_nvshmem is preferred for
    # no-sync symmetric-memory EP.
    no_sync_1d = "no_sync_1d"
    # Production OLMo-owned rowwise NVSHMEM EP. Uses static rowwise buffers and
    # device-side dispatch/combine metadata.
    rowwise_nvshmem = "rowwise_nvshmem"
    # Experimental rowwise NVSHMEM waves. This tries to overlap the same batch's
    # dispatch, expert GEMMs, and combine, but the current NVSHMEM implementation
    # consumes enough SM resources that overlap has not made this faster. Not
    # recommended for training.
    rowwise_wave = "rowwise_wave"
    # DeepEP V2 ElasticBuffer backend. Uses DeepEP's own communication buffers
    # instead of OLMo symmetric-memory buffers.
    deepep_v2 = "deepep_v2"


class ExpertParallelSchedule(StrEnum):
    normal = "normal"
    tbo = "tbo"


@dataclass
class DeepEPConfig(Config):
    # Optional DeepEP source/build path to add to sys.path before import. If not
    # set, the backend falls back to OLMO_DEEPEP_PATH or /workspace/DeepEP.
    path: Optional[str] = None
    # SMs to use for DeepEP dispatch/combine kernels. 0 lets DeepEP estimate a
    # value from topology, bandwidth, number of experts, and top-k.
    num_sms: int = 0
    # RDMA queue pairs to use for DeepEP dispatch/combine. 0 lets DeepEP infer a
    # value from num_sms and the selected communication mode.
    num_qps: int = 0
    # RDMA queue pairs to pre-allocate in the ElasticBuffer. 0 lets DeepEP choose
    # its default upper bound; otherwise this must be >= num_qps.
    num_allocated_qps: int = 0
    # Padding alignment for received rows per expert. The current model path
    # consumes packed rows, so deepep_v2 validates this as 1 for now.
    expert_alignment: int = 1
    # Pass async_with_compute_stream=True to DeepEP. When enabled, DeepEP returns
    # an event instead of making the current stream wait for comm immediately;
    # callers must place explicit waits before consuming comm outputs.
    async_mode: bool = False
    # Hint DeepEP's auto-tuner to reserve fewer SMs so communication can overlap
    # with GEMM. If false, DeepEP tends to choose a larger SM count for standalone
    # communication speed.
    prefer_overlap_with_compute: bool = True
    # Allow DeepEP's hybrid communication mode. This can use more QPs but is the
    # intended path for mixed NVLink/RDMA topologies.
    allow_hybrid_mode: bool = True
    # Allow DeepEP combine to reduce multiple contributions for a token inside
    # its communication/reduction path.
    allow_multiple_reduction: bool = True
    # Where to apply top-k weights. "swiglu" fuses the weights into the routed
    # expert path and currently requires bias-free down projections.
    weighting: str = "swiglu"

    def validate(self) -> None:
        self.weighting = self.weighting.lower()
        if self.num_sms < 0:
            raise OLMoConfigurationError(
                f"EP DeepEP num_sms must be >= 0 (got {self.num_sms})"
            )
        if self.num_qps < 0:
            raise OLMoConfigurationError(
                f"EP DeepEP num_qps must be >= 0 (got {self.num_qps})"
            )
        if self.num_allocated_qps < 0:
            raise OLMoConfigurationError(
                "EP DeepEP num_allocated_qps must be >= 0 "
                f"(got {self.num_allocated_qps})"
            )

@dataclass
class ExpertParallelConfig(Config):
    path: ExpertParallelPath = ExpertParallelPath.sync_1d
    schedule: ExpertParallelSchedule = ExpertParallelSchedule.normal

    # Rowwise and deepep_v2 use this as destination-rank expanded-row capacity
    # and may tail-drop overflow routes.
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

    deepep: DeepEPConfig = field(default_factory=DeepEPConfig)

    def validate(self) -> None:
        self.path = ExpertParallelPath(self.path)
        self.schedule = ExpertParallelSchedule(self.schedule)
        self.restore_unpermute_backend = self.restore_unpermute_backend.lower()
        self.rowwise_wave_mode = self.rowwise_wave_mode.lower()
        self.deepep.validate()

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
        if self.path == ExpertParallelPath.deepep_v2:
            if self.deepep.expert_alignment != 1:
                raise OLMoConfigurationError(
                    "EP deepep_v2 currently supports only "
                    "deepep.expert_alignment=1 "
                    f"(got {self.deepep.expert_alignment})"
                )
            if self.deepep.weighting != "swiglu":
                raise OLMoConfigurationError(
                    "EP deepep_v2 model path currently supports only "
                    f"deepep.weighting='swiglu' (got {self.deepep.weighting!r})"
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
    def is_deepep(self) -> bool:
        return self.path == ExpertParallelPath.deepep_v2

    @property
    def uses_olmo_symm(self) -> bool:
        return self.no_sync and not self.is_deepep

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
