import logging

log = logging.getLogger(__name__)

try:
    import transformer_engine.pytorch as te  # type: ignore
except (ImportError, OSError) as e:
    # ImportError: TE not installed. OSError: TE installed but its native library fails to load
    # (e.g. a cuBLAS/CUDA ABI mismatch in the image). Either way, degrade to "TE unavailable" so
    # non-TE paths keep working instead of breaking every import that reaches nn.transformer.
    log.warning(f"transformer_engine unavailable, disabling TE attention: {e}")
    te = None


def has_te_attn() -> bool:
    """Check if Transformer Engine attention is available."""
    return te is not None


TEDotProductAttention = te.DotProductAttention if te is not None else None
