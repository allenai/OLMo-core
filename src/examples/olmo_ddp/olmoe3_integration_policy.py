"""Pure, explicit settings for comparing against either integration baseline."""

FLAGS = {
    "OLMO_PROFILE_SAFE_NOOP_NVTX": "0",
    "OLMO_PROFILE_RS_SINGLE_PARAM_FAST_PATH": "0",
    "OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE": "0",
    "OLMO_PROFILE_SWIGLU_PAIRWISE": "0",
    "OLMO_PROFILE_EMO_DOCUMENT_POOL": "0",
    "OLMO_PROFILE_EMO_TOP16": "0",
    "OLMO_PROFILE_ROUNDED_WGRAD": "0",
    "OLMO_PROFILE_DDP_DEFER_REPLICATED_REDUCTIONS": "0",
    "OLMO_PROFILE_LB_COUNT_OVERLAP": "0",
    "OLMO_PROFILE_LB_COUNT_BATCHED": "0",
}
QUALIFIED_POLICY = "core-docpool-top16-wgrad-rs"
BASELINE_COMMIT = "107dfa3ff42f4ee4984d5c445d587ceb7db5e4f4"


def integration_policy(arm, policy, baseline="original", communication="none"):
    """Return every switch, rejecting mixed or ambiguous baseline selections.

    The original campaign remains reproducible. With ``optimized100b``, both arms
    use its complete qualified bundle; only the candidate enables new collectives.
    No environment variable is implicitly inherited into the returned flags.
    """
    if arm not in ("reference", "optimized") or policy not in (
        "core-docpool",
        "core-docpool-top16",
        "core-docpool-wgrad",
        "core-docpool-top16-wgrad",
        QUALIFIED_POLICY,
    ):
        raise ValueError((arm, policy))
    if baseline not in ("original", "optimized100b") or communication not in (
        "none",
        "deferred",
        "lb-overlap",
        "deferred-lb",
    ):
        raise ValueError((baseline, communication))
    if baseline == "original" and communication != "none":
        raise ValueError("New communication comparisons require the optimized100b baseline")
    if baseline == "optimized100b" and policy != QUALIFIED_POLICY:
        raise ValueError("The optimized100b baseline requires its complete qualified policy")
    qualified = arm == "optimized" or baseline == "optimized100b"
    flags = dict(FLAGS)
    if qualified:
        for key in (
            "OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE",
            "OLMO_PROFILE_SWIGLU_PAIRWISE",
            "OLMO_PROFILE_EMO_DOCUMENT_POOL",
        ):
            flags[key] = "1"
        flags["OLMO_PROFILE_EMO_TOP16"] = "1" if "top16" in policy else "0"
        flags["OLMO_PROFILE_ROUNDED_WGRAD"] = "1" if "wgrad" in policy else "0"
        flags["OLMO_PROFILE_RS_SINGLE_PARAM_FAST_PATH"] = "1" if policy.endswith("-rs") else "0"
    if baseline == "optimized100b" and arm == "optimized":
        flags["OLMO_PROFILE_DDP_DEFER_REPLICATED_REDUCTIONS"] = (
            "1" if communication in ("deferred", "deferred-lb") else "0"
        )
        flags["OLMO_PROFILE_LB_COUNT_OVERLAP"] = (
            "1" if communication in ("lb-overlap", "deferred-lb") else "0"
        )
    return {
        "flags": flags,
        "kda_min_ctas": 128 if qualified else 256,
        "inverse_scatter": qualified,
        "reduce_scatter": qualified and policy.endswith("-rs"),
        "baseline": baseline,
        "qualified_baseline_commit": BASELINE_COMMIT if baseline == "optimized100b" else None,
        "communication": communication if arm == "optimized" else "none",
    }
