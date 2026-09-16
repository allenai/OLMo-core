"""
Named launch presets for :class:`~olmo_core.launch.beaker.BeakerLaunchConfig`.

Internal recipes apply these defaults before merging their CLI overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PRESETS", "LaunchPreset", "get_preset"]


@dataclass
class LaunchPreset:
    """
    A named bundle of launch defaults layered onto a
    :class:`~olmo_core.launch.beaker.BeakerLaunchConfig`.

    :param name: The preset's name (e.g. ``"olmo-ddp"``).
    :param description: One-line summary.
    :param beaker_image: A default Beaker image for this preset.
    :param env_vars: ``(NAME, VALUE)`` environment variables to add.
    :param env_secrets: ``(NAME, SECRET_NAME)`` env vars sourced from Beaker secrets.
    :param pre_setup: A shell command to run *before* the repo clone + package install.
        May only touch the image/system (``olmo_core`` isn't installed yet).
    :param post_setup: A shell command to run *after* the package install. This is where
        steps that import ``olmo_core`` belong (e.g. building a runtime CUDA extension).
    """

    name: str
    description: str = ""
    beaker_image: str | None = None
    env_vars: list[tuple[str, str]] = field(default_factory=list)
    env_secrets: list[tuple[str, str]] = field(default_factory=list)
    pre_setup: str | None = None
    post_setup: str | None = None


# Build the shared-memory extension once per replica before torchrun starts its ranks.
OLMO_DDP = LaunchPreset(
    name="olmo-ddp",
    description="OLMoDDP on B300 with FlashAttention 4 and shared-memory transport.",
    beaker_image="akshitab/olmo-core-tch2110cu130-fa4-rma-2026-07-24",
    env_vars=[
        ("PYTHONPATH", "/gantry-runtime/src"),
        ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        ("OLMO_SYMM_VDEV2D_AUTO_BUILD", "1"),
    ],
    post_setup="python -m olmo_core.kernels.build_symm_mem_vdev2d_ext --inplace --backend cmake",
)


PRESETS: dict[str, LaunchPreset] = {p.name: p for p in (OLMO_DDP,)}


def get_preset(name: str) -> LaunchPreset:
    """
    Look up a launch preset by name.

    :raises KeyError: If no preset with that name is registered.
    """
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown launch preset '{name}'. Available presets: {sorted(PRESETS)}"
        ) from None
