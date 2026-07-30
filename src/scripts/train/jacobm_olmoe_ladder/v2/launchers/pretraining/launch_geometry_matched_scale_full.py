#!/usr/bin/env python3
"""Validate and optionally launch the larger geometry-matched production wave."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any

from gantry.api import GitRepoState, Recipe

from scripts.train.jacobm_olmoe_ladder.v2.launchers.pretraining.launch_geometry_matched_scale_nope_smokes import (
    load_manifest,
    validate_remote_commit,
)
from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_scale import (
    MXFP8_ALIGNED_EXPERT_HIDDEN_SIZES,
    build_geometry_matched_scale_gdn2_model_config,
    build_geometry_matched_scale_kda_model_config,
    build_geometry_matched_scale_kda_latent_moe_model_config,
    build_geometry_matched_scale_model_config,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "manifests" / "geometry_matched_scale_nope_full.yaml"
DEFAULT_RECORD = SCRIPT_DIR / "generated" / "geometry_matched_scale_full_submissions.json"
DIAGNOSTIC_RECORD = SCRIPT_DIR / "generated" / "nonfinite_diagnostic_submissions.json"
GRAD_DEBUG_RECORD = SCRIPT_DIR / "generated" / "gradient_debug_resume_submissions.json"
DIAGNOSTIC_DUMP_ROOT = Path(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/debug/nonfinite-grad"
)
GDN2_FLA_OVERLAY = "/tmp/fla-gdn2-cbb0a72"
GDN2_FLA_SPEC = (
    "flash-linear-attention[cuda] @ git+https://github.com/fla-org/"
    "flash-linear-attention.git@cbb0a72efb55c18ca0ef4f298298317573ad2cb3"
)
GDN2_FLA_EXPECTED_COMMIT = "cbb0a72efb55c18ca0ef4f298298317573ad2cb3"

GLOBAL_BATCHES = {
    1: 262_144,
    2: 393_216,
    4: 524_288,
    8: 786_432,
}
TRANSFERRED_WIDE_LRS = {
    ("480m", 1): "1.2e-3",
    ("480m", 2): "9e-4",
    ("480m", 4): "8e-4",
    ("480m", 8): "8e-4",
    ("810m", 1): "6e-4",
    ("810m", 2): "5.6e-4",
    ("810m", 4): "4e-4",
    ("810m", 8): "4e-4",
    ("1p2b", 1): "4e-4",
    ("1p2b", 2): "6e-4",
    ("1p2b", 4): "3e-4",
    ("1p2b", 8): "4e-4",
}
ACCELERATED_LAYOUT = {
    # (nodes, GPUs/node, EP, EP path, rank microbatch sequences)
    ("480m", 1): (1, 8, 1, "rowwise_nvshmem", 4),
    ("480m", 2): (1, 8, 1, "rowwise_nvshmem", 6),
    ("480m", 4): (1, 8, 1, "rowwise_nvshmem", 8),
    ("480m", 8): (1, 8, 1, "rowwise_nvshmem", 12),
    ("810m", 1): (2, 8, 1, "rowwise_nvshmem", 2),
    ("810m", 2): (2, 8, 1, "rowwise_nvshmem", 3),
    ("810m", 4): (2, 8, 1, "rowwise_nvshmem", 4),
    ("810m", 8): (2, 8, 1, "rowwise_nvshmem", 6),
    ("1p2b", 1): (2, 8, 8, "sync_1d", 2),
    ("1p2b", 2): (2, 8, 8, "sync_1d", 3),
    ("1p2b", 4): (4, 8, 8, "sync_1d", 2),
    ("1p2b", 8): (4, 8, 8, "sync_1d", 3),
}
GDN2_WALLCLOCK_CANDIDATE_LAYOUT = {
    # GDN2 carries more activation memory than GDN1. Keep the established
    # accumulation-free layout, but split 480M Cx8 across two nodes so its
    # per-rank microbatch can fall from 12 to 6. This remains locked behind
    # capacity qualification in the candidate manifest.
    # (nodes, GPUs/node, EP, EP path, rank microbatch sequences)
    ("480m", 1): (1, 8, 1, "rowwise_nvshmem", 4),
    ("480m", 2): (1, 8, 1, "rowwise_nvshmem", 6),
    ("480m", 4): (1, 8, 1, "rowwise_nvshmem", 8),
    ("480m", 8): (2, 8, 1, "rowwise_nvshmem", 6),
    ("810m", 1): (2, 8, 1, "rowwise_nvshmem", 2),
    ("810m", 2): (2, 8, 1, "rowwise_nvshmem", 3),
    ("810m", 4): (2, 8, 1, "rowwise_nvshmem", 4),
    ("810m", 8): (2, 8, 1, "rowwise_nvshmem", 6),
    ("1p2b", 1): (2, 8, 8, "sync_1d", 2),
    ("1p2b", 2): (2, 8, 8, "sync_1d", 3),
    ("1p2b", 4): (4, 8, 8, "sync_1d", 2),
    ("1p2b", 8): (4, 8, 8, "sync_1d", 3),
}
GDN2_BALANCED_LAYOUT = {
    # Preserve the qualified smaller-model layouts, while using the
    # resource-balanced 8/16/16/32-GPU 1.2B allocation. This matches the
    # whole-wave wall time of the 96-GPU candidate because Cx8 remains the
    # critical path, but saves 24 concurrent GPUs.
    # (nodes, GPUs/node, EP, EP path, rank microbatch sequences)
    **{key: value for key, value in GDN2_WALLCLOCK_CANDIDATE_LAYOUT.items() if key[0] != "1p2b"},
    ("1p2b", 1): (1, 8, 8, "sync_1d", 4),
    ("1p2b", 2): (2, 8, 8, "sync_1d", 3),
    ("1p2b", 4): (2, 8, 8, "sync_1d", 4),
    ("1p2b", 8): (4, 8, 8, "sync_1d", 3),
}
KDA_480M_LAYOUT = {key: value for key, value in GDN2_BALANCED_LAYOUT.items() if key[0] == "480m"}
KDA_MXFP8_480M_LAYOUT = dict(KDA_480M_LAYOUT)
KDA_810M_LAYOUT = {key: value for key, value in GDN2_BALANCED_LAYOUT.items() if key[0] == "810m"}
KDA_810M_CX4_LAYOUT = {
    key: value for key, value in KDA_810M_LAYOUT.items() if key == ("810m", 4)
}
KDA_1P2B_LAYOUT = {
    # Keep the qualified balanced GPU/MB/EP layout, but use the current
    # codebase's fixed default rowwise collective rather than the legacy
    # sync_1d workaround.
    key: (nodes, gpus_per_node, ep, "rowwise_nvshmem", rank_mb)
    for key, (nodes, gpus_per_node, ep, _ep_path, rank_mb) in GDN2_BALANCED_LAYOUT.items()
    if key[0] == "1p2b"
}
COMPACT_V1_LAYOUT = {
    # Reuse the demonstrated first-hybrid layouts for 480M/810M, then retain
    # extra nodes only for the larger 1.2B data-multiple cells.
    # (nodes, GPUs/node, EP, EP path, rank microbatch sequences)
    ("480m", 1): (1, 4, 1, "rowwise_nvshmem", 8),
    ("480m", 2): (1, 4, 1, "rowwise_nvshmem", 12),
    ("480m", 4): (1, 4, 1, "rowwise_nvshmem", 8),
    ("480m", 8): (1, 8, 1, "rowwise_nvshmem", 12),
    ("810m", 1): (1, 8, 1, "rowwise_nvshmem", 4),
    ("810m", 2): (1, 8, 1, "rowwise_nvshmem", 6),
    ("810m", 4): (1, 8, 1, "rowwise_nvshmem", 4),
    ("810m", 8): (1, 8, 1, "rowwise_nvshmem", 6),
    ("1p2b", 1): (1, 8, 8, "sync_1d", 4),
    ("1p2b", 2): (2, 8, 8, "sync_1d", 3),
    ("1p2b", 4): (2, 8, 8, "sync_1d", 4),
    ("1p2b", 8): (4, 8, 8, "sync_1d", 3),
}
LAYOUT_PROFILES = {
    "accelerated": ACCELERATED_LAYOUT,
    "compact_v1": COMPACT_V1_LAYOUT,
    "gdn2_wallclock_candidate": GDN2_WALLCLOCK_CANDIDATE_LAYOUT,
    "gdn2_balanced": GDN2_BALANCED_LAYOUT,
    "kda_480m": KDA_480M_LAYOUT,
    "kda_mxfp8_480m": KDA_MXFP8_480M_LAYOUT,
    "kda_810m": KDA_810M_LAYOUT,
    "kda_810m_cx4": KDA_810M_CX4_LAYOUT,
    "kda_1p2b": KDA_1P2B_LAYOUT,
}
MODEL_VARIANTS = {
    "geometry_matched_gdn_ev2_nope": {
        "rope": False,
        "attention_gate": False,
        "gdn2": False,
    },
    "geometry_matched_gdn_ev2_nope_gated": {
        "rope": False,
        "attention_gate": True,
        "gdn2": False,
    },
    "geometry_matched_gdn_ev2_rope_gated": {
        "rope": True,
        "attention_gate": True,
        "gdn2": False,
    },
    "geometry_matched_gdn2_ev2_nope_gated": {
        "rope": False,
        "attention_gate": True,
        "gdn2": True,
        "expand_v": 2.0,
        "allow_neg_eigval": True,
    },
    # Same architecture as above, but routed through the exact 275M geometry
    # builder. This alias is also useful for read-only checkpoint diagnostics.
    "geometry_275m_gdn2_ev2_nope_gated": {
        "rope": False,
        "attention_gate": True,
        "gdn2": True,
        "expand_v": 2.0,
        "allow_neg_eigval": True,
    },
    "geometry_matched_gdn2_ev1_noneg_nope_gated": {
        "rope": False,
        "attention_gate": True,
        "gdn2": True,
        "expand_v": 1.0,
        "allow_neg_eigval": False,
    },
    "geometry_matched_kda_ev1_noneg_nope_gated": {
        "rope": False,
        "attention_gate": True,
        "gdn2": False,
        "kda": True,
        "expand_v": 1.0,
        "allow_neg_eigval": False,
    },
    "geometry_matched_kda_ev2_neg_nope_gated": {
        "rope": False,
        "attention_gate": True,
        "gdn2": False,
        "kda": True,
        "expand_v": 2.0,
        "allow_neg_eigval": True,
    },
    "geometry_matched_kda_ev2_neg_nope_gated_mxfp8_aligned": {
        "rope": False,
        "attention_gate": True,
        "gdn2": False,
        "kda": True,
        "expand_v": 2.0,
        "allow_neg_eigval": True,
        "mxfp8_aligned": True,
    },
    "geometry_matched_kda_ev2_neg_nope_gated_latent2x_papermatched": {
        "rope": False,
        "attention_gate": True,
        "gdn2": False,
        "kda": True,
        "expand_v": 2.0,
        "allow_neg_eigval": True,
        "latent_moe_compression": 2,
    },
    "geometry_matched_kda_ev2_neg_nope_gated_latent4x_1000experts": {
        "rope": False,
        "attention_gate": True,
        "gdn2": False,
        "kda": True,
        "expand_v": 2.0,
        "allow_neg_eigval": True,
        "latent_moe_compression": 4,
        "latent_moe_num_experts": 1000,
    },
}
NONFINITE_DIAGNOSTIC_STOPS = {
    ("geometry_matched_gdn_ev2_nope", "1p2b-cx8"): 18_500,
    ("geometry_matched_gdn_ev2_nope_gated", "1p2b-cx2"): 21_500,
}


def validate(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    training = manifest["training"]
    variant = str(training["model_variant"])
    try:
        profile = MODEL_VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(f"unsupported production model variant {variant!r}") from exc
    smoke = bool(training.get("smoke", False))
    layout_profile = str(training.get("layout_profile", "accelerated"))
    try:
        selected_layout = LAYOUT_PROFILES[layout_profile]
    except KeyError as exc:
        raise ValueError(f"unsupported production layout profile {layout_profile!r}") from exc
    hard_stop_steps = int(training["hard_stop_steps"])
    if smoke:
        if hard_stop_steps < 12:
            raise ValueError("capacity smokes require at least 12 optimizer steps")
        if bool(training["checkpoints"]):
            raise ValueError("capacity smokes must disable checkpoints")
    else:
        if hard_stop_steps != 0:
            raise ValueError("production runs must set hard_stop_steps: 0")
        if not bool(training["checkpoints"]):
            raise ValueError("production runs must enable checkpoints")
    if bool(training["evals"]):
        raise ValueError("production runs must disable in-loop evaluation")
    if not smoke and str(training["checkpoint_removal"]) != "ephemeral_only":
        raise ValueError("production runs must retain only rolling ephemeral checkpoints")

    mxfp8_aligned = bool(profile.get("mxfp8_aligned", False))
    mxfp8_experts = bool(training.get("mxfp8_experts", False))
    mxfp8_attention = bool(training.get("mxfp8_attention", False))
    attention_impl = str(training.get("attention_impl", "default"))
    attention_backend = str(training.get("attention_backend", "te"))
    mxfp8_scale_mode = str(training.get("mxfp8_scale_mode", ""))
    if mxfp8_aligned:
        if not mxfp8_experts or not mxfp8_attention:
            raise ValueError("aggressive MXFP8 requires both expert and attention MXFP8")
        if attention_impl != "fused_v2" or attention_backend != "flash_4":
            raise ValueError("aggressive MXFP8 requires fused_v2 attention with FlashAttention-4")
        if mxfp8_scale_mode != "rceil":
            raise ValueError("aggressive MXFP8 requires mxfp8_scale_mode: rceil")
    elif mxfp8_experts or mxfp8_attention or mxfp8_scale_mode:
        raise ValueError("MXFP8 controls require the audited MXFP8-aligned model variant")

    rows = manifest["runs"]
    keys = {(str(row["model_size"]), int(row["cx"])) for row in rows}
    if keys != set(selected_layout):
        raise ValueError(
            f"manifest must contain the complete selected layout; missing="
            f"{sorted(set(selected_layout) - keys)}, extra={sorted(keys - set(selected_layout))}"
        )
    if len(rows) != len(keys):
        raise ValueError("manifest contains duplicate model/Cx cells")

    sequence_length = int(training["sequence_length"])
    task_names: set[str] = set()
    run_names: set[str] = set()
    for row in rows:
        model_size = str(row["model_size"])
        cx = int(row["cx"])
        key = (model_size, cx)
        task_name = str(row["task_name"])
        run_name = str(row["run_name"])
        if task_name in task_names or run_name in run_names:
            raise ValueError(f"duplicate task/run name: {task_name} / {run_name}")
        task_names.add(task_name)
        run_names.add(run_name)

        expected_layout = selected_layout[key]
        actual_layout = (
            int(row["num_nodes"]),
            int(row["gpus_per_node"]),
            int(row["expert_parallel_size"]),
            str(row.get("expert_parallel_path", "rowwise_nvshmem")),
            int(row["rank_microbatch_sequences"]),
        )
        if actual_layout != expected_layout:
            raise ValueError(
                f"{task_name}: expected layout {expected_layout}, found {actual_layout}"
            )
        if str(row["learning_rate"]) != TRANSFERRED_WIDE_LRS[key]:
            raise ValueError(
                f"{task_name}: expected transferred LR {TRANSFERRED_WIDE_LRS[key]}, "
                f"found {row['learning_rate']}"
            )
        if int(row["global_batch_size"]) != GLOBAL_BATCHES[cx]:
            raise ValueError(
                f"{task_name}: expected global batch {GLOBAL_BATCHES[cx]}, "
                f"found {row['global_batch_size']}"
            )

        world_size = int(row["num_nodes"]) * int(row["gpus_per_node"])
        ep_size = int(row["expert_parallel_size"])
        if world_size % ep_size:
            raise ValueError(f"{task_name}: EP={ep_size} does not divide world={world_size}")
        global_sequences = int(row["global_batch_size"]) // sequence_length
        if global_sequences % world_size:
            raise ValueError(
                f"{task_name}: {global_sequences} global sequences do not divide world={world_size}"
            )
        rank_sequences = global_sequences // world_size
        microbatch = int(row["rank_microbatch_sequences"])
        if rank_sequences % microbatch:
            raise ValueError(
                f"{task_name}: rank batch {rank_sequences} does not divide MB={microbatch}"
            )
        row["world_size"] = world_size
        row["rank_sequences"] = rank_sequences
        row["accumulation_steps"] = rank_sequences // microbatch

    for model_size in sorted({str(row["model_size"]) for row in rows}):
        if latent_moe_compression := profile.get("latent_moe_compression"):
            model = build_geometry_matched_scale_kda_latent_moe_model_config(
                model_size,
                compression=int(latent_moe_compression),
                num_experts_override=(
                    int(num_experts)
                    if (num_experts := profile.get("latent_moe_num_experts")) is not None
                    else None
                ),
            )
        elif bool(profile.get("kda", False)):
            model = build_geometry_matched_scale_kda_model_config(
                model_size,
                expand_v=float(profile["expand_v"]),
                allow_neg_eigval=bool(profile["allow_neg_eigval"]),
                expert_hidden_size=(
                    MXFP8_ALIGNED_EXPERT_HIDDEN_SIZES[model_size] if mxfp8_aligned else None
                ),
            )
        elif bool(profile["gdn2"]):
            model = build_geometry_matched_scale_gdn2_model_config(
                model_size,
                rope=bool(profile["rope"]),
                attention_gate=bool(profile["attention_gate"]),
                expand_v=float(profile["expand_v"]),
                allow_neg_eigval=bool(profile["allow_neg_eigval"]),
            )
        else:
            model = build_geometry_matched_scale_model_config(
                model_size,
                rope=bool(profile["rope"]),
                attention_gate=bool(profile["attention_gate"]),
            )
        print(
            f"{model_size}: active={model.num_active_params:,} "
            f"active_non_embedding={model.num_active_non_embedding_params:,} "
            f"total={model.num_params:,}"
        )
    return rows


def recipe_for(
    manifest: dict[str, Any],
    row: dict[str, Any],
    *,
    commit: str,
    diagnose_nonfinite: bool = False,
    debug_gradients: bool = False,
    diagnostic_stop_step: int | None = None,
    runtime_env_overrides: dict[str, str] | None = None,
    recipe_suffix_override: str | None = None,
    description_suffix: str | None = None,
    gdn2_fla_overlay: str | None = None,
    gdn2_fla_spec: str | None = None,
    gdn2_fla_expected_commit: str | None = None,
) -> Recipe:
    source = manifest["source"]
    beaker = manifest["beaker"]
    training = manifest["training"]
    is_gdn2 = bool(MODEL_VARIANTS[str(training["model_variant"])]["gdn2"])
    selected_gdn2_fla_overlay = gdn2_fla_overlay or GDN2_FLA_OVERLAY
    selected_gdn2_fla_spec = gdn2_fla_spec or GDN2_FLA_SPEC
    selected_gdn2_fla_expected_commit = gdn2_fla_expected_commit or GDN2_FLA_EXPECTED_COMMIT
    env_vars = [
        (
            "PYTHONPATH",
            f"{selected_gdn2_fla_overlay}:src" if is_gdn2 else "src",
        ),
        ("PYTHONUNBUFFERED", "1"),
        ("CUDA_SCALE_LAUNCH_QUEUES", "4x"),
        ("OLMO_SHARED_FS", "1"),
        ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        ("OMP_NUM_THREADS", "8"),
        ("NCCL_DEBUG", "INFO"),
        ("LOG_FILTER_TYPE", "local_rank0_only"),
        ("OLMOE3_HYBRID_RUN_NAME", str(row["run_name"])),
        ("OLMOE3_HYBRID_MODEL_SIZE", str(row["model_size"])),
        ("OLMOE3_HYBRID_MODEL_VARIANT", str(training["model_variant"])),
        ("OLMOE3_HYBRID_LR", str(row["learning_rate"])),
        ("OLMOE3_HYBRID_CHINCHILLA_MULTIPLE", str(row["cx"])),
        ("OLMOE3_HYBRID_GLOBAL_BATCH_SIZE", str(row["global_batch_size"])),
        ("OLMOE3_HYBRID_WORLD_SIZE", str(row["world_size"])),
        ("OLMOE3_HYBRID_NUM_NODES", str(row["num_nodes"])),
        ("OLMOE3_HYBRID_EP_SIZE", str(row["expert_parallel_size"])),
        (
            "OLMOE3_HYBRID_EP_PATH",
            str(row.get("expert_parallel_path", "rowwise_nvshmem")),
        ),
        (
            "OLMOE3_HYBRID_RANK_MICROBATCH_SEQUENCES",
            str(row["rank_microbatch_sequences"]),
        ),
        ("OLMOE3_HYBRID_SEQUENCE_LENGTH", str(training["sequence_length"])),
        (
            "OLMOE3_HYBRID_HARD_STOP_STEPS",
            str(diagnostic_stop_step or int(training["hard_stop_steps"])),
        ),
        ("OLMOE3_HYBRID_CHECKPOINTS", str(int(bool(training["checkpoints"])))),
        ("OLMOE3_HYBRID_CHECKPOINT_WRITES", str(int(bool(training["checkpoints"])))),
        ("OLMOE3_HYBRID_SAVE_INTERVAL", str(training["save_interval"])),
        (
            "OLMOE3_HYBRID_EPHEMERAL_SAVE_INTERVAL",
            str(training["ephemeral_save_interval"]),
        ),
        ("OLMOE3_HYBRID_CHECKPOINT_REMOVAL", str(training["checkpoint_removal"])),
        ("OLMOE3_HYBRID_EVALS", "0"),
        ("OLMOE3_HYBRID_EVAL_ON_FINISH", "0"),
        ("OLMOE3_HYBRID_EVAL_INTERVAL", str(training["eval_interval"])),
        ("OLMOE3_HYBRID_EVAL_STEPS", "0"),
        ("OLMOE3_HYBRID_USE_COMPILE", str(int(bool(training["compile"])))),
        ("OLMOE3_HYBRID_WANDB", str(int(bool(training["wandb"])))),
        ("OLMOE3_HYBRID_ATTENTION_IMPL", str(training.get("attention_impl", "default"))),
        ("OLMOE3_HYBRID_ATTENTION_BACKEND", str(training.get("attention_backend", "te"))),
        (
            "OLMOE3_HYBRID_MXFP8_EXPERTS",
            str(int(bool(training.get("mxfp8_experts", False)))),
        ),
        (
            "OLMOE3_HYBRID_MXFP8_ATTENTION",
            str(int(bool(training.get("mxfp8_attention", False)))),
        ),
        ("OLMOE3_HYBRID_SAVE_ROOT", str(manifest["experiment"]["checkpoint_root"])),
    ]
    if mxfp8_scale_mode := training.get("mxfp8_scale_mode"):
        env_vars.append(("OLMO_MXFP8_SCALE_MODE", str(mxfp8_scale_mode)))
    if diagnose_nonfinite:
        if diagnostic_stop_step is None:
            raise ValueError("diagnostic_stop_step is required for a diagnostic run")
        env_vars.extend(
            [
                ("OLMO_DDP_DEBUG_NONFINITE_GRAD", "1"),
                ("OLMO_DDP_DEBUG_GRAD_NORMS", "100"),
                ("OLMO_DDP_DEBUG_GRAD_NORMS_RANKS", "all"),
                ("OLMO_DEBUG_DUMP_OPTIM_GRAD_NORMS", "1"),
                ("OLMO_DEBUG_DUMP_DIR", str(DIAGNOSTIC_DUMP_ROOT)),
                ("OLMO_DEBUG_RUN_ID", f"{row['run_name']}-diagnostic-r1"),
            ]
        )
    if debug_gradients:
        env_vars.extend(
            [
                ("OLMO_DDP_DEBUG_NONFINITE_GRAD", "1"),
                ("OLMO_DDP_DEBUG_NONFINITE_GRAD_RANKS", "all"),
                ("OLMO_DDP_DEBUG_NONFINITE_GRAD_TOPK", "50"),
                ("OLMO_DDP_DEBUG_GRAD_NORMS", "20"),
                ("OLMO_DDP_DEBUG_GRAD_NORMS_RANKS", "all"),
                ("OLMO_DDP_DEBUG_GRAD_NORMS_MIN", "100"),
            ]
        )
    if runtime_env_overrides:
        override_names = set(runtime_env_overrides)
        env_vars = [(name, value) for name, value in env_vars if name not in override_names]
        env_vars.extend(runtime_env_overrides.items())
    env_secrets = [(str(name), str(secret)) for name, secret in manifest.get("secrets", {}).items()]
    weka = [(str(item["bucket"]), str(item["mount"])) for item in manifest.get("weka", [])]
    git_repo = GitRepoState.from_env(ref=commit, branch=str(source["branch"]))
    num_nodes = int(row["num_nodes"])
    default_recipe_suffix = (
        "-nonfinite-diagnostic-r1"
        if diagnose_nonfinite
        else "-grad-debug-r1"
        if debug_gradients
        else ""
    )
    recipe_suffix = (
        recipe_suffix_override if recipe_suffix_override is not None else default_recipe_suffix
    )
    pre_setup = "unset S3_PROFILE"
    post_setup = None
    if (
        int(row["expert_parallel_size"]) > 1
        and str(row.get("expert_parallel_path", "rowwise_nvshmem")) == "rowwise_nvshmem"
    ):
        # The source branch contains the fixed rowwise implementation, while
        # the current base image predates its small CUDA helper extension.
        # Build once per replica before torchrun rather than racing an
        # import-time auto-build across local ranks.
        post_setup = (
            "cd /gantry-runtime && "
            "PYTHONPATH=/gantry-runtime/src python -m "
            "olmo_core.kernels.build_symm_mem_vdev2d_ext "
            "--inplace --backend cmake"
        )
    if is_gdn2:
        verify_script = f"""
import json
from importlib.metadata import distributions

import fla
from fla.ops.gdn2 import chunk_gdn2

dist = next(
    dist
    for dist in distributions(path=[{selected_gdn2_fla_overlay!r}])
    if dist.metadata["Name"] == "flash-linear-attention"
)
direct_url = json.loads(dist.read_text("direct_url.json"))
commit = direct_url["vcs_info"]["commit_id"]
print(f"GDN2 FLA overlay: version={{fla.__version__}} commit={{commit}} path={{fla.__file__}}")
print(f"GDN2 kernel: {{chunk_gdn2.__module__}}.{{chunk_gdn2.__name__}}")
assert fla.__version__ == "0.5.2"
assert commit == {selected_gdn2_fla_expected_commit!r}
"""
        pre_setup += (
            f"\nrm -rf {shlex.quote(selected_gdn2_fla_overlay)}"
            "\npython -m pip install "
            f"--target {shlex.quote(selected_gdn2_fla_overlay)} --no-deps "
            f"--no-build-isolation {shlex.quote(selected_gdn2_fla_spec)}"
            f"\nPYTHONPATH={shlex.quote(selected_gdn2_fla_overlay)} "
            f"python -c {shlex.quote(verify_script)}"
        )
    return Recipe(
        args=[
            "src/scripts/train/jacobm_olmoe3_hybrid_scale.py",
            "train",
            str(row["run_name"]),
            "local",
        ],
        name=f"{row['run_name']}{recipe_suffix}",
        description=(
            f"{manifest['experiment']['description']} ({description_suffix})"
            if description_suffix is not None
            else (
                f"{manifest['experiment']['description']} (non-finite gradient diagnostic)"
                if diagnose_nonfinite
                else (
                    f"{manifest['experiment']['description']} (gradient-debug resume)"
                    if debug_gradients
                    else str(manifest["experiment"]["description"])
                )
            )
        ),
        workspace=str(beaker["workspace"]),
        task_name=f"{row['task_name']}{recipe_suffix}",
        git_repo=git_repo,
        allow_dirty=False,
        yes=True,
        clusters=[str(beaker["cluster"])],
        gpus=int(row["gpus_per_node"]),
        shared_memory=str(beaker["shared_memory"]),
        beaker_image=str(source["image"]),
        env_vars=env_vars,
        env_secrets=env_secrets,
        weka=weka,
        priority=str(beaker["priority"]),
        min_runtime=str(beaker["min_runtime"]),
        auto_resume=bool(beaker["auto_resume"]),
        task_timeout=str(beaker["timeout"]),
        replicas=num_nodes if num_nodes > 1 else None,
        leader_selection=num_nodes > 1,
        host_networking=True,
        propagate_failure=True if num_nodes > 1 else None,
        propagate_preemption=True if num_nodes > 1 else None,
        synchronized_start_timeout="90m" if num_nodes > 1 else None,
        torchrun=True,
        no_python=True,
        pre_setup=pre_setup,
        post_setup=post_setup,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument(
        "--run-suffix",
        default="",
        help=(
            "Append a validated suffix to both task and run names for a clean "
            "from-scratch reproduction with a distinct checkpoint directory."
        ),
    )
    parser.add_argument(
        "--diagnose-nonfinite",
        action="store_true",
        help="Resume a known unstable cell with non-finite gradient dumps and a short hard stop.",
    )
    parser.add_argument(
        "--debug-gradients",
        action="store_true",
        help="Resume a production cell with non-finite and large-gradient diagnostics enabled.",
    )
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest.resolve())
    rows = validate(manifest)
    if args.task:
        wanted = set(args.task)
        available = {str(row["task_name"]) for row in rows}
        if missing := wanted - available:
            raise ValueError(f"unknown selected tasks: {sorted(missing)}")
        rows = [row for row in rows if str(row["task_name"]) in wanted]

    if args.run_suffix:
        if not re.fullmatch(r"-[a-z0-9][a-z0-9-]*", args.run_suffix):
            raise ValueError(
                "--run-suffix must start with '-' and contain only lowercase "
                "letters, digits, and hyphens"
            )
        if args.resume_existing or args.diagnose_nonfinite or args.debug_gradients:
            raise ValueError(
                "--run-suffix is only for a clean from-scratch launch; do not "
                "combine it with --resume-existing or --diagnose-nonfinite"
            )
        rows = [
            {
                **row,
                "task_name": f"{row['task_name']}{args.run_suffix}",
                "run_name": f"{row['run_name']}{args.run_suffix}",
            }
            for row in rows
        ]

    variant = str(manifest["training"]["model_variant"])
    diagnostic_stops: dict[str, int] = {}
    if args.debug_gradients:
        if not args.resume_existing:
            raise ValueError("--debug-gradients requires --resume-existing")
        if args.diagnose_nonfinite:
            raise ValueError("--debug-gradients and --diagnose-nonfinite are mutually exclusive")
        if args.record == DEFAULT_RECORD:
            args.record = GRAD_DEBUG_RECORD
    if args.diagnose_nonfinite:
        if not args.resume_existing:
            raise ValueError("--diagnose-nonfinite requires --resume-existing")
        for row in rows:
            task_name = str(row["task_name"])
            try:
                diagnostic_stops[task_name] = NONFINITE_DIAGNOSTIC_STOPS[(variant, task_name)]
            except KeyError as exc:
                raise ValueError(
                    f"No approved non-finite diagnostic stop for {variant}/{task_name}"
                ) from exc
        if args.record == DEFAULT_RECORD:
            args.record = DIAGNOSTIC_RECORD

    print("\nsize  Cx nodes GPU/node world EP rank_seq MB accum LR run")
    for row in rows:
        print(
            f"{row['model_size']:<5} {row['cx']:>2} {row['num_nodes']:>5} "
            f"{row['gpus_per_node']:>8} {row['world_size']:>5} "
            f"{row['expert_parallel_size']:>2} {row['rank_sequences']:>8} "
            f"{row['rank_microbatch_sequences']:>2} {row['accumulation_steps']:>5} "
            f"{row['learning_rate']:<7} {row['run_name']}"
        )
    print(f"\nSelected {len(rows)} runs using {sum(row['world_size'] for row in rows)} GPUs.")
    if not args.submit:
        print("Dry run only; pass --submit to launch.")
        return
    if not bool(manifest["training"].get("smoke", False)):
        qualified_model_sizes = {
            str(model_size)
            for model_size in manifest["training"].get("capacity_qualified_model_sizes", [])
        }
        unqualified_model_sizes = {str(row["model_size"]) for row in rows} - qualified_model_sizes
        if unqualified_model_sizes:
            raise RuntimeError(
                "Submission is locked for model sizes without recorded checkpoint-free "
                f"capacity qualification: {sorted(unqualified_model_sizes)}"
            )

    checkpoint_root = Path(str(manifest["experiment"]["checkpoint_root"]))
    existing = (
        [
            checkpoint_root / str(row["run_name"])
            for row in rows
            if (checkpoint_root / str(row["run_name"])).exists()
        ]
        if bool(manifest["training"]["checkpoints"])
        else []
    )
    if existing and not args.resume_existing:
        raise RuntimeError(
            "Refusing to submit existing checkpoint directories:\n"
            + "\n".join(f"  - {path}" for path in existing)
            + "\nPass --resume-existing only for an intentional continuation."
        )

    source = manifest["source"]
    commit = validate_remote_commit(str(source["remote"]), str(source["branch"]))
    records: list[dict[str, Any]] = []
    for row in rows:
        task_name = str(row["task_name"])
        workload = recipe_for(
            manifest,
            row,
            commit=commit,
            diagnose_nonfinite=args.diagnose_nonfinite,
            debug_gradients=args.debug_gradients,
            diagnostic_stop_step=diagnostic_stops.get(task_name),
        ).launch(show_logs=False)
        experiment = workload.experiment
        record = {
            "model_variant": manifest["training"]["model_variant"],
            "task_name": row["task_name"],
            "run_name": row["run_name"],
            "run_suffix": args.run_suffix or None,
            "commit": commit,
            "diagnose_nonfinite": args.diagnose_nonfinite,
            "debug_gradients": args.debug_gradients,
            "diagnostic_stop_step": diagnostic_stops.get(task_name),
            "diagnostic_dump_root": (
                str(DIAGNOSTIC_DUMP_ROOT) if args.diagnose_nonfinite else None
            ),
            "experiment_id": experiment.id,
            "task_ids": [task.id for task in experiment.tasks],
            "url": (
                "https://beaker.org/orgs/ai2/workspaces/"
                f"OLMo-3-moe-experiments/work/{experiment.id}"
            ),
        }
        records.append(record)
        print(f"Submitted {row['task_name']}: {record['url']}")

    args.record.parent.mkdir(parents=True, exist_ok=True)
    existing_records: list[dict[str, Any]] = []
    if args.record.is_file():
        existing_records = json.loads(args.record.read_text())
    existing_records.extend(records)
    args.record.write_text(json.dumps(existing_records, indent=2) + "\n")
    print(f"Recorded {len(records)} submissions in {args.record}")


if __name__ == "__main__":
    main()
