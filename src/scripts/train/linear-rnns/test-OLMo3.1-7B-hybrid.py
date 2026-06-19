# --- B300 (sm_103) Triton ptxas workaround ---
# Triton's bundled ptxas is too old for sm_103a (B300). Point it at a newer ptxas: torch's bundled
# one (13.0 on cu130 images) if present, else the CUDA 12.9 nvcc wheel pulled in via the 'b300'
# extra. Done here (in-container, before any torch.compile) so it doesn't depend on the launcher's
# installed olmo_core / setup_steps.
import glob as _glob
import os as _os

if "TRITON_PTXAS_PATH" not in _os.environ:
    for _pat in (
        "/opt/conda/lib/python*/site-packages/torch/bin/ptxas",
        "/opt/conda/lib/python*/site-packages/nvidia/cuda_nvcc/bin/ptxas",
    ):
        _hits = _glob.glob(_pat)
        if _hits:
            _os.environ["TRITON_PTXAS_PATH"] = _hits[0]
            break

# --- B300 (sm_103) CUPTI workaround (for torch.profiler) ---
# torch's kineto loads a CUPTI too old for sm_103 (CUPTI_ERROR_INVALID_DEVICE), so the profiler
# can't capture CUDA activities. Preload a newer CUPTI (12.9, from the 'b300' extra) by full path
# BEFORE torch is imported — once its soname (libcupti.so.12) is in the process, kineto's later
# dlopen binds to it. LD_LIBRARY_PATH can't be used here (glibc caches the loader path at startup).
import ctypes as _ctypes

for _cupti in sorted(
    _glob.glob("/opt/conda/lib/python*/site-packages/nvidia/cuda_cupti/lib/libcupti.so*")
):
    try:
        _ctypes.CDLL(_cupti, mode=_ctypes.RTLD_GLOBAL)
        break
    except OSError:
        pass

import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, Dict, List

import torch

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    build_config,
    main,
)
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.fla.layer import FLAConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.distributed.utils import get_rank
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)


@dataclass
class CudaEventBlockTimer(Callback):
    """
    Measures per-block-type forward GPU wall-time with ``torch.cuda.Event`` (no CUPTI needed),
    to localize the B300 per-step gap to attention vs the FLA/GatedDeltaNet blocks. Registers
    block-level forward hooks on rank 0 for a window of steps (after compile settles), then logs
    the attention-vs-FLA split and removes itself.

    Notes:
    - Block-level hooks fire *outside* the per-block compiled region, so they add no graph breaks.
    - Times are GPU wall-time on the compute stream, so they include any exposed FSDP comm wait.
    - If activation checkpointing is on, the forward hook also fires during backward recompute;
      that's counted for both block types, so the relative split stays valid.
    """

    start_step: int = 25
    num_steps: int = 5
    _records: List[Any] = field(default_factory=list)
    _pending: Dict[int, Any] = field(default_factory=dict)
    _handles: List[Any] = field(default_factory=list)
    _collecting: bool = False
    _done: bool = False

    @staticmethod
    def _classify(block) -> str:
        if hasattr(block, "fla"):
            return "fla"
        if hasattr(block, "attention"):
            return "attn"
        return "other"

    def pre_train(self):
        # Register hooks ONCE, before the first step / compile — registering mid-run mutates the
        # modules and forces a torch.compile recapture, after which the recorded events are invalid
        # ("Both events must be recorded"). Hooks are no-ops until _collecting is set in the window.
        if get_rank() != 0:
            return
        model = self.trainer.train_module.model
        for block in model.blocks.values():
            btype = self._classify(block)

            def _pre(mod, args, _bt=btype):
                if not self._collecting:
                    return
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self._pending[id(mod)] = (_bt, ev)

            def _post(mod, args, output):
                if not self._collecting or id(mod) not in self._pending:
                    return
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                bt, start = self._pending.pop(id(mod))
                self._records.append((bt, start, ev))

            self._handles.append(block.register_forward_pre_hook(_pre))
            self._handles.append(block.register_forward_hook(_post))
        log.info(f"[cuda-event timer] hooked {len(model.blocks)} blocks (collect at step {self.start_step})")

    def pre_step(self, batch):
        if get_rank() == 0 and not self._done and self.step == self.start_step:
            self._collecting = True

    def post_step(self):
        if get_rank() != 0 or self._done or not self._collecting:
            return
        if self.step < self.start_step + self.num_steps - 1:
            return
        self._collecting = False
        self._done = True
        torch.cuda.synchronize()
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        skipped = 0
        for bt, start, end in self._records:
            try:
                dt = start.elapsed_time(end)
            except (ValueError, RuntimeError):
                skipped += 1
                continue
            totals[bt] = totals.get(bt, 0.0) + dt
            counts[bt] = counts.get(bt, 0) + 1
        parts = [
            f"{bt}={totals[bt]:.1f}ms/{counts[bt]}calls ({totals[bt] / max(counts[bt], 1):.2f}ms each)"
            for bt in sorted(totals)
        ]
        log.info(
            f"[cuda-event block timing | forward | rank0 | skipped={skipped}] " + " | ".join(parts)
        )
        for h in self._handles:
            h.remove()
        self._handles.clear()


@dataclass
class GpuUtilSampler(Callback):
    """
    Samples GPU SM/mem utilization via ``nvidia-smi dmon`` in a background thread during steady-state
    training (rank 0's node). Distinguishes **launch-bound** (sm% well below ~100% — the GatedDeltaNet
    chunked kernels can't keep the GPU fed) from **GPU-bound** (sm% near 100%) — which decides whether
    B300 kernel tuning can help. Uses basic NVML, so it's unaffected by the CUPTI profiling restriction.
    """

    start_step: int = 25
    seconds: int = 15
    _started: bool = False

    def pre_step(self, batch):
        if get_rank() != 0 or self._started or self.step != self.start_step:
            return
        self._started = True
        import subprocess
        import threading

        def _run():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-c", str(self.seconds)],
                    capture_output=True,
                    text=True,
                    timeout=self.seconds + 60,
                )
                log.info(
                    f"[gpu-util | nvidia-smi dmon -s u | ~{self.seconds}s steady-state, rank0 node]\n"
                    + out.stdout
                    + (("\n" + out.stderr) if out.stderr else "")
                )
            except Exception as e:  # noqa: BLE001
                log.warning(f"[gpu-util] nvidia-smi dmon failed: {e}")

        threading.Thread(target=_run, daemon=True).start()


SEQUENCE_LENGTH = 8 * 1024
GLOBAL_BATCH_SIZE = 4 * 1024 * 1024  # ~4M tokens

# Uncomment for benchmarking TPS (simulate TPS at 64 nodes with 16 nodes)
# GLOBAL_BATCH_SIZE //= 4

# Reduce per-device batch size to save on memory.
MICROBATCH_DISCOUNT = 1

# Remove heads to match params/TPS of transformer.
REMOVE_HEADS = 2

### OLMo "3.1" 7B Settings (from OLMo 3 32B)
DATA_MIX = DataMix.OLMo_mix_0925
MAX_DURATION = Duration.epochs(1)
HARD_STOP = None
INSTANCE_FILTER = True


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo3_7B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        # See README for how to override with flash_3 using CLI.
        attn_backend=AttentionBackendName.flash_2,
    )

    # Remove heads (and scale down d_model) to compensate for extra params.
    config.d_model -= REMOVE_HEADS * 128
    config.block.attention.n_heads -= REMOVE_HEADS
    assert config.d_model / config.block.attention.n_heads == 128

    ### Copied below from hybrid/gated_deltanet_0_25_rnn_first.py ###

    # Update the config to use an FLA block.
    config.block.name = TransformerBlockType.fla_hybrid
    assert config.n_layers % 4 == 0, "Current logic assumes n_layers is multiple of 4"
    config.block.fla_hybrid_attention_indices = [i for i in range(config.n_layers) if i % 4 == 3]

    # Configure the non-attention part of the block to be a DeltaNet.
    config.block.fla = FLAConfig(
        name="GatedDeltaNet",
        dtype=config.dtype,
        fla_layer_kwargs={
            # FLA repo says num_heads * head_dim = 0.75 * hidden_size
            "head_dim": int(0.75 * config.d_model / config.block.attention.n_heads),
            "use_gate": True,
            "allow_neg_eigval": True,
        },
    )

    return config


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    rank_microbatch_size = common.max_sequence_length

    # if common.launch is not None:
    #     gpus = {CLUSTER_TO_GPU_TYPE.get(c, "unknown") for c in common.launch.clusters}
    #     if all("B200" in g for g in gpus):
    #         rank_microbatch_size *= 2

    # Added because FLA models seem to use more memory than transformers.
    rank_microbatch_size = int(rank_microbatch_size // MICROBATCH_DISCOUNT)

    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.max_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
) -> DataComponents:
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DATA_MIX,
        tokenizer=common.tokenizer,
        mix_base_dir=common.root_dir,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        # max target sequence length doesn't affect how the data is loaded, just how it's cached behind the scenes
        max_target_sequence_length=max(common.max_sequence_length, 8192),
        generate_doc_lengths=intra_document_masking,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=8
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 10

    cluster = common.cluster

    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"

    return (
        TrainerConfig(
            # willm: Adapted this from 1B linear RNN runs.
            # save_folder=f"{common.root_dir}/checkpoints/willm/linear-rnns/{common.run_name}/",
            save_folder=f"{common.root_dir}/checkpoints/akshitab/linear-rnns/{common.run_name}/",
            save_overwrite=True,
            metrics_collect_interval=50,
            cancel_check_interval=cancel_check_interval,
            max_duration=MAX_DURATION,
            hard_stop=HARD_STOP,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                entity="ai2-llm",
                project="holmes-testing",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        # NOTE: torch.profiler disabled — CUPTI is blocked on this B300 cluster
        # (CUPTI_ERROR_INVALID_DEVICE), so it only captured CPU activities while with_stack=True
        # added heavy overhead. Using CudaEventBlockTimer below instead. Re-enable if CUPTI works.
        # .with_callback(
        #     "profiler",
        #     ProfilerCallback(
        #         skip_first=10,
        #         wait=1,
        #         warmup=3,
        #         active=5,
        #         repeat=1,
        #         with_stack=True,
        #     ),
        # )
        .with_callback(
            # CUPTI-free per-block GPU timing (attention vs FLA). Fires steps 25-29 on rank 0.
            "block_timer",
            CudaEventBlockTimer(start_step=25, num_steps=5),
        )
        .with_callback(
            # GPU SM/mem utilization sampling (launch-bound vs GPU-bound). ~15s from step 25, rank 0.
            "gpu_util",
            GpuUtilSampler(start_step=25, seconds=15),
        )
        .with_recommended_evals(common.tokenizer, SEQUENCE_LENGTH, cluster, task_set="fast")
    )


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        data_config_builder=build_data_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        include_instance_filter=INSTANCE_FILTER,
        beaker_workspace="ai2/holmes-testing",
        use_hostname_constraints=True,
        num_execution_units=1,
    )
    main(config_builder=config_builder)
