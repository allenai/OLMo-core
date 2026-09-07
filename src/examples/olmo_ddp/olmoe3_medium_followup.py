"""Bounded medium gradient diagnostics and profile-guided follow-up experiments."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from olmoe3_medium_followup_plan import microbatch_sequence_sizes, parse_test, sample_offsets

VARIANT, TEST_MB, TEST_BATCH = parse_test(os.environ.get("OLMOE3_DEEP_PROFILE_TEST", "optimized"))
os.environ["OLMOE3_DEEP_PROFILE_TEST"] = "baseline" if VARIANT == "baseline" else "optimized"

import olmoe3_medium_deep_profile as base
import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.experiment import build_config, main
from olmo_core.train.callbacks import Callback

DIAGNOSTIC = os.environ.get("OLMOE3_MEDIUM_DIAGNOSTIC", "0") == "1"
MB = TEST_MB or int(os.environ.get("OLMOE3_MEDIUM_MB", "2"))
BATCH = TEST_BATCH or int(os.environ.get("OLMOE3_MEDIUM_BATCH", "16777216"))
if MB not in (1, 2, 3, 4) or BATCH not in (8388608, 16777216, 33554432):
    raise ValueError((MB, BATCH))
MICROBATCHES = microbatch_sequence_sizes(BATCH, base.TOPOLOGY.gpus, MB)


class BalancedSystemConfig(base.base.SystemConfig):
    """Experimental PP1 metadata: count actual balanced chunks, never floor division."""

    @property
    def gradient_accumulation_steps(self):
        return len(MICROBATCHES)

    def validate(self):
        assert self.model_size == "medium" and self.pp == 1 and self.ep == 8
        assert self.rank_microbatch_sequences == 3
        assert sum(MICROBATCHES) * self.num_gpus * 8192 == BATCH


base.TEST = VARIANT
system_type = BalancedSystemConfig if MB == 3 else base.base.SystemConfig
base.SYSTEM = system_type("medium", base.TOPOLOGY.nodes, 1, 8, MB)
base.TOPOLOGY = SimpleNamespace(
    gpus=base.TOPOLOGY.gpus,
    nodes=base.TOPOLOGY.nodes,
    batch_tokens=BATCH,
    accumulation=len(MICROBATCHES),
    validate_rank_groups=base.TOPOLOGY.validate_rank_groups,
)
base.base.GLOBAL_BATCH_SIZE = BATCH
base.STEPS = 1 if DIAGNOSTIC else (60 if base.PASS == "torch" else 100)

if VARIANT in ("no-wgrad", "core-only"):
    base.FLAGS["OLMO_PROFILE_ROUNDED_WGRAD"] = "0"
    base.FLAGS["OLMO_PROFILE_ROUNDED_WGRAD_EP"] = "0"
if VARIANT in ("no-rs", "core-only"):
    base.SETTINGS["reduce_scatter"] = False
    base.FLAGS["OLMO_PROFILE_RS_SINGLE_PARAM_FAST_PATH"] = "0"
if VARIANT == "no-kda":
    base.SETTINGS["kda_min_ctas"] = 256
if VARIANT == "no-routing":
    base.SETTINGS["inverse_scatter"] = False
    base.FLAGS["OLMO_PROFILE_EMO_DOCUMENT_POOL"] = "0"
    base.FLAGS["OLMO_PROFILE_EMO_TOP16"] = "0"
BATCH_COUNTS = VARIANT in ("optimized-lb-batched", "optimized-lb-batched-metrics5")
base.FLAGS["OLMO_PROFILE_BALANCED_MICROBATCH"] = "1" if MB == 3 else "0"
base.FLAGS["OLMO_PROFILE_BALANCED_MICROBATCH_UNIT"] = "16" if MB == 3 else "0"
base.FLAGS["OLMO_PROFILE_LB_COUNT_BATCHED_EP"] = "1" if BATCH_COUNTS else "0"
if BATCH_COUNTS:
    base.FLAGS["OLMO_PROFILE_LB_COUNT_BATCHED"] = "1"
os.environ.update(base.FLAGS)

# Memory candidates use the existing attention-only checkpoint hook. They never
# recompute MoE routing/EP collectives or enable shared EP output buffers.
CHECKPOINT_BLOCKS = []
if VARIANT in ("optimized-ackda", "optimized-ackda-half"):
    CHECKPOINT_BLOCKS = [i for i in range(24) if i not in (7, 15, 23)]
    if VARIANT.endswith("-half"):
        CHECKPOINT_BLOCKS = CHECKPOINT_BLOCKS[::2]
original_model_config = base.model_config


def model_config(common):
    """Apply only selected existing KDA-sublayer recomputation hooks."""
    model = original_model_config(common)
    for index in CHECKPOINT_BLOCKS:
        block = deepcopy(model.block_overrides.get(index, model.block))
        block.checkpoint_attn = True
        model.block_overrides[index] = block
    if CHECKPOINT_BLOCKS:
        assert (model.num_active_params, model.num_params) == (
            base.EXPECTED_ACTIVE,
            base.EXPECTED_TOTAL,
        )
    return model


base.model_config = model_config
base.SETTINGS["checkpoint_attention_blocks"] = CHECKPOINT_BLOCKS
base.SETTINGS["microbatch_sequence_sizes"] = MICROBATCHES
base.SETTINGS["metrics_collect_interval"] = (
    5 if VARIANT in ("optimized-metrics5", "optimized-lb-batched-metrics5") else 1
)
base.SETTINGS["nccl_protocol"] = "Simple" if VARIANT == "optimized-simple" else "auto"


def family(name):
    """Keep gradient comparisons grouped by functional parameter family."""
    if "router" in name:
        return "router"
    if "routed_experts" in name:
        return "expert-down" if "w_down" in name else "expert-up-gate"
    if "sequence_mixer" in name or "attention" in name:
        return "sequence-mixer"
    return "dense-other"


def local(tensor):
    """Use local shards without gathering model-scale tensors."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


@dataclass
class GradientAudit(Callback):
    """Capture compact gradient summaries and fixed samples, never full gradients."""

    output_dir: str = ""

    def pre_train(self):
        self.output = Path(self.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.pre_rows = []
        self.samples = {}
        self.gpu_norms = []
        tm = self.trainer.train_module
        optim = tm.optim
        self.optim = optim
        # Skip actual Adam updates for a one-step, matched-initialization diagnostic.
        optim._step_foreach = lambda *args, **kwargs: None
        optim._copy_main_params_to_model_params = lambda: None
        for reducer in (
            tm.model_parts if os.environ.get("OLMOE3_GRAD_PRE_REDUCTION", "0") == "1" else []
        ):
            original_launch = reducer._launch_bucket_grad_reduce

            def launch(index, reducer=reducer, original=original_launch):
                if self.step == 1:
                    bucket = reducer._grad_buckets[index]
                    for param in bucket.params:
                        name = reducer._param_to_name[param]
                        grad = reducer._param_to_bucket_view[param]
                        self.capture(
                            "pre-reduction",
                            name,
                            grad,
                            str(dist.get_world_size(bucket.process_group)),
                            "full-local",
                        )
                return original(index)

            reducer._launch_bucket_grad_reduce = launch
        original_clip = optim._clip_grad

        def clip():
            for group in optim.param_groups:
                for name, param in group["named_params"].items():
                    if not param.requires_grad or name not in optim.main_grad:
                        continue
                    placements = optim.states[f"{name}.main"].placements
                    shard = any(p.is_shard() for p in placements)
                    grad = local(optim.main_grad[name]).detach()
                    index = self.capture(
                        "optimizer-intake", name, grad, group["pg"], str(placements)
                    )
                    # Correct multiplicity when summing scalar squared norms over WORLD.
                    # Dense shards repeat over EP-MP only if actually replicated; EP shards
                    # repeat over EP-DP only when the optimizer does not shard them further.
                    copies = (
                        1
                        if shard
                        else (get_world_size() if group["pg"] == "dp" else get_world_size() // 8)
                    )
                    self.gpu_norms.append((index, copies))
            values = torch.stack(
                [self.pre_rows[i]["norm_tensor"].square() / copies for i, copies in self.gpu_norms]
            ).sum()
            dist.all_reduce(values)
            independent = values.sqrt().item()
            reported = original_clip()
            torch.cuda.synchronize()
            rows = []
            for row in self.pre_rows:
                norm = float(row.pop("norm_tensor").item())
                rows.append({**row, "norm": norm})
            summary = {
                "variant": VARIANT,
                "rank": get_rank(),
                "gpus": get_world_size(),
                "mb": MB,
                "batch": BATCH,
                "independent_norm": independent,
                "reported_norm": float(reported.item()),
                "no_weight_update": True,
                "first_step_rng": getattr(self, "first_step_rng", None),
                "parameters": rows,
            }
            (self.output / f"gradients-rank-{get_rank()}.json").write_text(json.dumps(summary))
            if self.samples:
                import numpy as np

                np.savez(self.output / f"gradient-samples-rank-{get_rank()}.npz", **self.samples)
            if get_rank() == 0:
                print(
                    "GRADIENT_DIAGNOSTIC",
                    json.dumps({k: v for k, v in summary.items() if k != "parameters"}),
                    flush=True,
                )
            return reported

        optim._clip_grad = clip

    def pre_step(self, batch):
        """Fingerprint stochastic-routing RNG after compilation/dry-run consumption."""
        del batch
        if self.step == 1:
            self.first_step_rng = {
                "cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                "cuda": hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest(),
            }

    def capture(self, stage, name, grad, group, placements):
        """Retain full-shard norms but only 2048 regularly spaced tensor elements."""
        grad = local(grad).detach().reshape(-1)
        row = {
            "stage": stage,
            "name": name,
            "family": family(name),
            "group": group,
            "placements": placements,
            "numel": grad.numel(),
            "norm_tensor": torch.linalg.vector_norm(grad, dtype=torch.float64),
        }
        index = len(self.pre_rows)
        self.pre_rows.append(row)
        if get_rank() < 8 and grad.numel():
            indices = torch.tensor(
                sample_offsets(grad.numel()), dtype=torch.int64, device=grad.device
            )
            self.samples[f"{stage}/{name}"] = grad[indices].float().cpu().numpy()
        return index


@dataclass
class PeakMemoryAudit(Callback):
    """Preserve worst per-step rank-local allocator and whole-device observations."""

    output_dir: str = ""

    def pre_train(self):
        self.peak = {"allocated": 0, "reserved": 0, "device_used": 0}

    def post_train_batch(self):
        free, total = torch.cuda.mem_get_info()
        self.peak = {
            "allocated": max(self.peak["allocated"], torch.cuda.max_memory_allocated()),
            "reserved": max(self.peak["reserved"], torch.cuda.max_memory_reserved()),
            "device_used": max(self.peak["device_used"], total - free),
        }

    def post_train(self):
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"full-run-memory-rank-{get_rank()}.json").write_text(json.dumps(self.peak))


def trainer_config(common):
    """Attach diagnostics without changing model or normal optimizer arithmetic."""
    cfg = base.trainer_config(common)
    cfg.metrics_collect_interval = base.SETTINGS["metrics_collect_interval"]
    # The original audit's geometry guards are still appropriate for EP8/PP1.
    # Override its reporting metadata separately below for non-MB2/batch16Mi tests.
    if DIAGNOSTIC:
        cfg.add_callback("gradient_audit", GradientAudit(output_dir=common.save_folder))
    cfg.add_callback("full_run_memory", PeakMemoryAudit(output_dir=common.save_folder))
    cfg.callbacks["wandb"].tags = [
        t for t in cfg.callbacks["wandb"].tags if t not in ("mb2", "16mi")
    ] + [
        "medium-followup",
        f"variant:{VARIANT}",
        f"actual-mb:{MB}",
        f"actual-batch:{BATCH}",
        f"diagnostic:{DIAGNOSTIC}",
    ]
    cfg.callbacks["wandb"].notes = (
        f"Locked medium PP1/EP8, variant {VARIANT}, MB{MB}, batch {BATCH}; "
        f"KDA attention recompute blocks {CHECKPOINT_BLOCKS}; "
        f"metrics interval {cfg.metrics_collect_interval}; "
        f"NCCL protocol {base.SETTINGS['nccl_protocol']}. "
        "BF16/FP32 unchanged, no shared EP outputs, no MoE recomputation, no FP8."
    )
    return cfg


if __name__ == "__main__":
    if sys.argv[1:] == ["--validate-only"]:

        common = SimpleNamespace(
            tokenizer=SimpleNamespace(eos_token_id=100257, padded_vocab_size=lambda: 100352),
            max_sequence_length=8192,
        )
        model, tm = base.model_config(common), base.train_module_config(common)
        assert [
            i for i in range(24) if model.block_overrides.get(i, model.block).checkpoint_attn
        ] == CHECKPOINT_BLOCKS
        assert (model.num_active_params, model.num_params) == (
            base.EXPECTED_ACTIVE,
            base.EXPECTED_TOTAL,
        )
        assert (
            tm.rank_microbatch_size == MB * 8192
            and tm.ac_config is None
            and tm.float8_config is None
        )
        assert tm.ep_config.degree == 8 and tm.pp_config is None
        assert sum(MICROBATCHES) * base.TOPOLOGY.gpus * 8192 == BATCH
        print(
            "FOLLOWUP_CONFIG_VALIDATED",
            json.dumps(
                {
                    "variant": VARIANT,
                    "gpus": base.TOPOLOGY.gpus,
                    "mb": MB,
                    "batch": BATCH,
                    "microbatch_sequence_sizes": MICROBATCHES,
                    "diagnostic": DIAGNOSTIC,
                    "flags": base.FLAGS,
                }
            ),
            flush=True,
        )
    else:
        main(
            config_builder=partial(
                build_config,
                global_batch_size=BATCH,
                max_sequence_length=8192,
                num_nodes=base.TOPOLOGY.nodes,
                common_config_builder=base.common_components,
                data_config_builder=base.base.build_data_components,
                model_config_builder=base.model_config,
                train_module_config_builder=base.train_module_config,
                trainer_config_builder=trainer_config,
                beaker_image=base.base.BEAKER_IMAGE,
                beaker_workspace=base.base.WORKSPACE,
                include_default_evals=False,
                num_execution_units=1,
            )
        )
