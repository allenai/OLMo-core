"""Opt-in EMO settings shared by the OLMoE3 systems probes."""

from __future__ import annotations

import os

from olmo_core.nn.moe import EmoRouterConfig

EMO_ENV_VARS = (
    "OLMOE3_EMO_ENABLED",
    "OLMOE3_EMO_MIN_POOL",
    "OLMOE3_EMO_MAX_POOL",
    "OLMOE3_EMO_EVAL_POOL",
)


def _enabled() -> bool:
    value = os.environ.get("OLMOE3_EMO_ENABLED", "false").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("OLMOE3_EMO_ENABLED must be one of 1/0, true/false, yes/no, or on/off")


def _required_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise ValueError(f"Set {name} when OLMOE3_EMO_ENABLED is true")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def emo_router_config(*, eos_token_id: int, num_experts: int, top_k: int) -> EmoRouterConfig | None:
    """Build the explicitly requested EMO policy, or return ``None`` when disabled.

    The training pool bounds are deliberately required so a systems test cannot silently choose a
    scientific EMO policy. The evaluation pool defaults to all experts, matching the current HF
    export requirement, when omitted.
    """

    if not _enabled():
        return None
    min_pool = _required_int("OLMOE3_EMO_MIN_POOL")
    max_pool = _required_int("OLMOE3_EMO_MAX_POOL")
    eval_pool = int(os.environ.get("OLMOE3_EMO_EVAL_POOL", str(num_experts)))
    config = EmoRouterConfig(
        eos_token_id=eos_token_id,
        min_document_expert_pool=min_pool,
        max_document_expert_pool=max_pool,
        eval_document_expert_pool=eval_pool,
    )
    config.validate_for_router(num_experts=num_experts, top_k=top_k)
    return config


def emo_tags(config: EmoRouterConfig | None) -> list[str]:
    """Return unambiguous experiment tags for an EMO policy."""

    if config is None:
        return ["emo:false"]
    return [
        "emo:true",
        f"emo-min-pool:{config.min_document_expert_pool}",
        f"emo-max-pool:{config.max_document_expert_pool}",
        f"emo-eval-pool:{config.eval_pool_size()}",
        "emo-global-lb:true",
        "lb-granularity:local_batch",
    ]


def emo_note(config: EmoRouterConfig | None) -> str:
    """Return a compact human-readable EMO description."""

    if config is None:
        return "EMO disabled"
    return (
        "EMO document pool="
        f"[{config.min_document_expert_pool}, {config.max_document_expert_pool}], "
        f"eval pool={config.eval_pool_size()}, EOS={config.eos_token_id}, "
        "global LB with local-batch loss"
    )
