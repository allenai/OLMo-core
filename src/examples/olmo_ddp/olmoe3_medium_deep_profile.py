"""Matched medium profiling on64/128B300; frozen model plus PR855 in both arms."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import olmoe3_small_medium_profile as base
from olmoe3_integration_policy import QUALIFIED_POLICY, integration_policy
from olmoe3_medium_profile_plan import MediumProfileTopology, inspect_route_metrics
from olmoe3_nsys_tools import NsysSettings

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.experiment import build_config, main
from olmo_core.optim.scheduler import WSD
from olmo_core.train import Duration
from olmo_core.train.callbacks import Callback, NvidiaProfilerCallback, ProfilerCallback

TOPOLOGY = MediumProfileTopology(int(os.environ.get("OLMOE3_MEDIUM_GPUS", "128")))
SYSTEM = base.SystemConfig("medium", TOPOLOGY.nodes, 1, 8, 2)
TEST = os.environ.get("OLMOE3_DEEP_PROFILE_TEST", "baseline")
if TEST not in ("baseline", "optimized"):
    raise ValueError(f"Unqualified medium variant: {TEST}")
SETTINGS = integration_policy("reference" if TEST == "baseline" else "optimized", QUALIFIED_POLICY)
FLAGS = {**SETTINGS["flags"], "OLMO_PROFILE_ROUNDED_WGRAD_EP": "1" if TEST == "optimized" else "0"}
os.environ.update(FLAGS)
PASS = os.environ.get("OLMOE3_DEEP_PROFILE_PASS", "timing")
if PASS not in ("timing", "nsys", "torch"):
    raise ValueError(f"Unqualified profile pass: {PASS}")
STEPS = 60 if PASS == "torch" else 100
LR = 8e-4  # Systems placeholder, not the final medium training LR.
ROOT = Path("/weka/olmo-3p5-checkpoints/production-profiling")
DATA_WORK = (
    "/weka/olmo-3p5-checkpoints/production-cbs/work/olmoe3-small-cbs-8mi-100b-lr1p3em3-uploader-r1"
)
PROFILE_RANKS = list(range(0, TOPOLOGY.gpus, 8))
NSYS = NsysSettings.from_env(world_size=TOPOLOGY.gpus)
EXPECTED_ACTIVE = 2_387_533_440
EXPECTED_TOTAL = 42_759_806_592


def model_config(common):
    """Apply only the requested per-head gains to the locked medium geometry."""
    from kernel_fun._common import support

    import olmo_core.ops.moe as moe_ops

    if support.MIN_CTAS not in (128, 256):
        raise RuntimeError("Kernel CTA heuristic changed")
    support.MIN_CTAS = SETTINGS["kda_min_ctas"]
    if SETTINGS["inverse_scatter"]:
        moe_ops.pool_keep_mask = moe_ops.pool_keep_mask_inverse_scatter
    model = base.build_model_config_from_common(common, SYSTEM)
    model.init_seed = 12536
    for index in (7, 15, 23):
        model.block_overrides[index].sequence_mixer.qk_norm_per_head_gains = True
    model.validate()
    if (model.num_active_params, model.num_params) != (EXPECTED_ACTIVE, EXPECTED_TOTAL):
        raise RuntimeError("Locked medium parameter counts changed unexpectedly")
    return model


def train_module_config(common):
    """Same BF16/FP32 optimizer and WSD for every arm, with no activation recomputation."""
    config = base.build_train_module_config(common, SYSTEM)
    config.dp_config.use_reduce_scatter = SETTINGS["reduce_scatter"]
    config.scheduler = WSD(warmup=20, decay=1, decay_fraction=None)
    return config


def common_components(cli_context, **kwargs):
    """Reuse the established Dolma data order; artifacts stay on Weka, not Beaker results."""
    common = base.build_common_components(cli_context, f"medium-{TOPOLOGY.gpus}g", SYSTEM, **kwargs)
    common.work_dir = DATA_WORK
    common.save_folder = str(ROOT / common.run_name)
    common.init_seed = 12536
    return common


@dataclass
class MediumAudit(Callback):
    """Check actual topology/initialization/data and retain every step's metrics."""

    output_dir: str = ""

    def _verify_hashes(self, filename, hashes):
        if get_rank() == 0:
            output = Path(self.output_dir)
            text = json.dumps(hashes)
            prefix = os.environ["OLMOE3_MEDIUM_RUN_PREFIX"]
            canonical = ROOT / f"{prefix}-{filename}"
            if canonical.exists() and canonical.read_text() != text:
                raise RuntimeError(f"Matched profile fingerprint changed: {canonical}")
            if not canonical.exists():
                canonical.write_text(text)
            (output / filename).write_text(text)

    def pre_train(self):
        tm = self.trainer.train_module
        if (
            self.step != 0
            or get_world_size() != TOPOLOGY.gpus
            or tm.dp_world_size != TOPOLOGY.gpus
            or tm.pp_enabled
        ):
            raise RuntimeError("Expected fresh medium PP1 run at the selected world size")
        if not tm.ep_enabled or dist.get_world_size(tm.ep_mp_group) != 8:
            raise RuntimeError("Expected EP8")
        hosts = [None] * TOPOLOGY.gpus
        dist.all_gather_object(hosts, os.environ["BEAKER_NODE_HOSTNAME"])
        TOPOLOGY.validate_rank_groups(tm.moe_mesh.mesh.reshape(-1, 8).tolist(), hosts)
        if any(os.environ.get(key) != value for key, value in FLAGS.items()):
            raise RuntimeError("Optimization switches changed after construction")
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        # Divide parameter hashing across physical nodes. Every selected node hashes
        # all8EP shards for its assigned parameters; dense replicas are redundant.
        host_order = sorted(set(hosts))
        node_index = host_order.index(hosts[get_rank()])
        digest = hashlib.sha256()
        for index, (name, parameter) in enumerate(tm.model.named_parameters()):
            if index % TOPOLOGY.nodes == node_index:
                value = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                digest.update(name.encode())
                digest.update(
                    value.detach()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                )
        hashes = [None] * TOPOLOGY.gpus
        dist.all_gather_object(hashes, digest.hexdigest(), group=self.trainer.bookkeeping_pg)
        self._verify_hashes("initial-weights-sha256.json", hashes)
        self._first_batch = True
        if get_rank() == 0:
            provenance = {
                "model": "medium",
                "test": TEST,
                "variant": TEST,
                "pass": PASS,
                "git_commit": os.environ.get("GIT_REF"),
                "gpus": TOPOLOGY.gpus,
                "source_step": 0,
                "source_checkpoint": "fresh-init12536-data928543231",
                "routing_caveat": "Fresh matched initialization, not mature hero-run routing",
                "active_params": EXPECTED_ACTIVE,
                "total_params": EXPECTED_TOTAL,
                "global_batch_tokens": TOPOLOGY.batch_tokens,
                "sequence_length": 8192,
                "microbatch_sequences": SYSTEM.rank_microbatch_sequences,
                "gradient_accumulation": TOPOLOGY.accumulation,
                "dense_dp": TOPOLOGY.gpus,
                "expert_parallel": 8,
                "ep_capacity_factor": 1.25,
                "expected_route_metrics": 23,
                "route_drop_policy": "record-and-qualify-windows",
                "expert_dp": TOPOLOGY.gpus // 8,
                "pipeline_parallel": 1,
                "qk_norm_pr": 855,
                "lr": LR,
                "warmup_steps": 20,
                "flops_per_token": tm.num_flops_per_token(8192),
                "kernel_fun_commit": "7a6983baf2beb4ec4d7fe914ec9f6670438af99b",
                "optimization_settings": SETTINGS,
                "flags": FLAGS,
                "nsys_profiled_ranks": PROFILE_RANKS if PASS == "nsys" else None,
                "nsys_version": NSYS.version if PASS == "nsys" else None,
                "nsys_relative_steps": [NSYS.start, NSYS.end] if PASS == "nsys" else None,
                "clean_windows_relative_steps": (
                    NSYS.clean_windows(STEPS)
                    if PASS == "nsys"
                    else [[31, 100]] if PASS == "timing" else [[21, 30], [51, 60]]
                ),
            }
            (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
            print("MEDIUM_PROFILE_START", json.dumps(provenance), flush=True)

    def pre_step(self, batch):
        if self._first_batch:
            digest = hashlib.sha256(
                batch["input_ids"].detach().contiguous().cpu().numpy().tobytes()
            ).hexdigest()
            hashes = [None] * TOPOLOGY.gpus
            dist.all_gather_object(hashes, digest, group=self.trainer.bookkeeping_pg)
            self._verify_hashes("first-batch-sha256.json", hashes)
            self._first_batch = False

    def log_metrics(self, step, metrics):
        for name, value in metrics.items():
            if name in ("train/CE loss", "optim/total grad norm") and not math.isfinite(
                float(value)
            ):
                raise RuntimeError(f"Nonfinite {name} at step{step}")
        routing = inspect_route_metrics(metrics)
        if routing["blocks"] != 23:
            raise RuntimeError(f"Expected all23 medium per-block route metrics, got {routing}")
        if get_rank() == 0:
            with (Path(self.output_dir) / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step, **metrics}) + "\n")
            if routing["blocks_with_drops"] and (step <= 10 or step % 10 == 0):
                print(
                    "MEDIUM_ROUTE_DROPS_REVIEW", json.dumps({"step": step, **routing}), flush=True
                )

    def post_train(self):
        if self.step != STEPS:
            raise RuntimeError(f"Incomplete medium pass: {self.step} != {STEPS}")
        memory = {
            "rank": get_rank(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocated_bytes_at_end": torch.cuda.memory_allocated(),
        }
        (Path(self.output_dir) / f"memory-rank-{get_rank()}.json").write_text(json.dumps(memory))


def trainer_config(common):
    """Isolate profiler passes and leave checkpoint/eval writes disabled."""
    config = base.build_trainer_config(common, f"medium-{TOPOLOGY.gpus}g", SYSTEM)
    config.save_folder = common.save_folder
    config.work_dir = common.save_folder
    config.save_overwrite = False
    config.max_duration = Duration.steps(1000)
    config.hard_stop = Duration.steps(STEPS)
    config.no_checkpoints = True
    config.no_evals = True
    config.callbacks["wandb"].tags = [
        "medium",
        f"gpus:{TOPOLOGY.gpus}",
        "pp1",
        "ep8",
        "mb2",
        "16mi",
        f"ga:{TOPOLOGY.accumulation}",
        "emo",
        "qknorm-pr855",
        f"arm:{TEST}",
        f"pass:{PASS}",
        "no-checkpoints",
    ]
    config.callbacks["wandb"].notes = (
        "Locked medium; PR855 in both arms; fresh matched initialization. Qualified100B small optimization bundle only in candidate; no new communication switches. No recomputation, FP8, shared EP outputs, TP/PP/CP/TBO or fused attention."
    )
    config.add_callback("medium_audit", MediumAudit(output_dir=common.save_folder))
    if PASS == "nsys":
        config.add_callback(
            "nsys_capture",
            NvidiaProfilerCallback(
                start=NSYS.start,
                end=NSYS.end,
                profile_ranks=PROFILE_RANKS,
                emit_autograd_nvtx=False,
            ),
        )
    elif PASS == "torch":
        config.add_callback(
            "torch_capture",
            ProfilerCallback(
                skip_first=30,
                wait=4,
                warmup=1,
                active=2,
                repeat=1,
                with_stack=False,
                profile_memory=False,
                enable_cuda_sync_events=True,
                export_distributed_event_summary=True,
                ranks=PROFILE_RANKS,
            ),
        )
    return config


if __name__ == "__main__":
    if sys.argv[1:] == ["--validate-only"]:
        from types import SimpleNamespace

        common = SimpleNamespace(
            tokenizer=SimpleNamespace(eos_token_id=100257, padded_vocab_size=lambda: 100352),
            max_sequence_length=8192,
        )
        model = model_config(common)
        tm = train_module_config(common)
        assert model.n_layers == 24 and model.d_model == 1536
        assert set(model.block_overrides) == {0, 7, 15, 23}
        assert all(
            model.block_overrides[i].sequence_mixer.qk_norm_per_head_gains for i in (7, 15, 23)
        )
        assert tm.ep_config.degree == 8 and tm.rank_microbatch_size == 16384
        assert tm.pp_config is None and tm.ac_config is None and tm.float8_config is None
        assert tm.dp_config.accumulate_grads_in_fp32 and tm.dp_config.reduce_grads_in_fp32
        assert tm.dp_config.use_reduce_scatter == (TEST == "optimized")
        print(
            "MEDIUM_CONFIG_VALIDATED",
            json.dumps(
                {
                    "gpus": TOPOLOGY.gpus,
                    "ga": TOPOLOGY.accumulation,
                    "arm": TEST,
                    "active": model.num_active_params,
                    "total": model.num_params,
                    "flags": FLAGS,
                }
            ),
            flush=True,
        )
        raise SystemExit(0)
    main(
        config_builder=partial(
            build_config,
            global_batch_size=TOPOLOGY.batch_tokens,
            max_sequence_length=8192,
            num_nodes=TOPOLOGY.nodes,
            common_config_builder=common_components,
            data_config_builder=base.build_data_components,
            model_config_builder=model_config,
            train_module_config_builder=train_module_config,
            trainer_config_builder=trainer_config,
            beaker_image=base.BEAKER_IMAGE,
            beaker_workspace=base.WORKSPACE,
            include_default_evals=False,
            num_execution_units=1,
        )
    )
