try:
    import transformer_engine.pytorch as te  # type: ignore
except Exception:
    # TE can fail at import time with errors other than ImportError (e.g. a
    # subprocess.CalledProcessError when it probes for libnvrtc via `ldconfig`). Treat any
    # import-time failure as "TE unavailable" rather than crashing every backend's import.
    te = None


def has_te_attn() -> bool:
    """Check if Transformer Engine attention is available."""
    return te is not None


TEDotProductAttention = te.DotProductAttention if te is not None else None
