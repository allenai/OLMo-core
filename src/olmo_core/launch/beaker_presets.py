"""
Named launch presets for :class:`~olmo_core.launch.beaker.BeakerLaunchConfig`.

A :class:`LaunchPreset` bundles a few launch defaults — environment variables, env
secrets, and pre/post-setup shell steps — that are commonly needed together for a class
of runs. Presets are applied *on top of* an otherwise-normal launch config; explicit
values (e.g. CLI ``--env``/``--pre-setup``) take precedence over the preset, which in
turn takes precedence over the launcher's built-in defaults.

Presets live here (in the library) rather than in any one training script so they can be
reused from the generic launcher CLI (``python -m olmo_core.launch.beaker --preset ...``),
from the internal experiment launcher, or when building a ``BeakerLaunchConfig`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["LaunchPreset", "PRESETS", "get_preset"]


@dataclass
class LaunchPreset:
    """
    A named bundle of launch defaults layered onto a
    :class:`~olmo_core.launch.beaker.BeakerLaunchConfig`.

    :param name: The preset's CLI name (e.g. ``"olmo-ddp"``).
    :param description: One-line summary shown in ``--help``.
    :param beaker_image: A default Beaker image for this preset. Overrides the launcher's
        stable-image default, but an explicit ``--beaker-image`` still overrides this.
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


# The OLMoDDP (fused MoE-v2) preset.
#
#  - beaker_image: the B300 image with flash-attn 4 + the symm-mem/RMA build prerequisites
#    (nvcc + NVSHMEM), which the rowwise-EP / PP transport kernels need.
#  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: the fused OLMoDDP stack sits close to
#    device-memory capacity and fragments the caching allocator; expandable segments reclaim
#    the stranded reserved memory that otherwise triggers spurious OOMs.
#  - post_setup builds the symm_mem_vdev2d extension once per node (post_setup runs on each
#    replica before torchrun spawns ranks), so the rowwise-EP path imports a ready .so with no
#    cross-rank build race and no first-step compile stall. Requires an image with nvcc + NVSHMEM
#    (the 'rma' images); harmless-but-wasteful for EP=1 runs that don't use the extension.
#  - OLMO_SYMM_VDEV2D_AUTO_BUILD=1 is only a fallback if the prebuilt .so is somehow missing.
#    NOTE: the runtime auto-build is only race-safe where symm_mem_vdev2d builds on local-rank-0
#    with a barrier; without that, prefer relying on the post_setup prebuild alone.
OLMO_DDP = LaunchPreset(
    name="olmo-ddp",
    description="OLMoDDP / fused MoE-v2 runs: B300 fa4-rma image, alloc-fragmentation fix, symm_mem_vdev2d prebuild.",
    beaker_image="akshitab/olmo-core-tch2110cu130-fa4-rma-2026-07-24",
    env_vars=[
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
