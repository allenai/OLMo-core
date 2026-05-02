"""Shared `--smoke-1gpu` plumbing for ladder scripts.

Single-GPU smoke runs queue much faster than 8-GPU jobs on busy clusters and
catch most code-path bugs (NaN, shape mismatches, dataloader plumbing) before
committing 8-GPU compute. Two production defaults block 1-GPU runs:

1. :class:`Olmo3ModelConfigurator.configure_minimal_device_mesh_spec` hardcodes
   ``world_size=8`` for sub-13B models.
2. :class:`WSDSChinchillaRunConfigurator` enforces ``chinchilla_multiple >= 0.5``
   and a power-of-2 check, which forces ~3500+ steps even at the minimum.

This module provides drop-in replacements that override both for smoke mode:

- :class:`Olmo3SmokeConfigurator` extends ``Olmo3ModelConfigurator`` with a
  ``smoke_1gpu`` field that switches the device-mesh spec to
  ``world_size=1, dp_world_size=1``.
- :class:`WSDSChinchillaSmoke` extends ``WSDSChinchillaRunConfigurator`` with
  relaxed validation, a single-anneal schedule, a 16 384-token target batch
  size, and a 16 384-token warmup so any tiny ``chinchilla_multiple`` produces
  a valid LR schedule.
- :func:`add_smoke_args` registers ``--smoke-1gpu`` on a parser; pass it as
  ``add_additional_args`` to ``main()``.

Usage in a ladder script::

    from olmo_core.internal.smoke import (
        Olmo3SmokeConfigurator,
        WSDSChinchillaSmoke,
        add_smoke_args,
    )

    def configure_ladder(args):
        smoke_1gpu = getattr(args, "smoke_1gpu", False)
        ladder = ModelLadder(
            ...,
            max_devices=1 if smoke_1gpu else args.max_gpus,
            model_configurator=Olmo3SmokeConfigurator(
                ...,
                smoke_1gpu=smoke_1gpu,
            ),
            run_configurator=(
                WSDSChinchillaSmoke(chinchilla_multiple=args.chinchilla_multiple)
                if smoke_1gpu
                else WSDSChinchillaRunConfigurator(
                    chinchilla_multiple=args.chinchilla_multiple
                )
            ),
            ...,
        )
        return ladder

    if __name__ == "__main__":
        main(configure_ladder, add_additional_args=add_smoke_args)

Caveats: doesn't catch distributed-only bugs (NCCL ordering, FSDP shard
semantics) — those still need the 8-GPU smoke. At >=1B parameters, microbatch
sizing may not transfer between 1-GPU and 8-GPU because activation memory no
longer dominates; verify before relying on smokes for memory-pressure
validation at that scale.
"""
import argparse
import dataclasses

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.model_ladder import (
    DeviceMeshSpec,
    Olmo3ModelConfigurator,
    WSDSChinchillaRunConfigurator,
)


@dataclasses.dataclass(kw_only=True)
class WSDSChinchillaSmoke(WSDSChinchillaRunConfigurator):
    """Smoke variant of WSDS Chinchilla. Differences from the parent:

    1. Drops the ``chinchilla_multiple >= 0.5`` floor and the power-of-2 check
       so smokes can use any small value (e.g. ``--chinchilla-multiple 0.001``
       gives ~230 optimizer steps).
    2. Single anneal at the requested multiple instead of one anneal per
       power of 2 from 2^-1 up — the production list is empty when the
       multiple is < 0.5.
    3. Warmup shrunk to 16 384 tokens. Production warmup is 1 token per param
       (e.g. 190 M tokens for the 190 M model), which exceeds the entire
       smoke training budget at tiny chinchilla multiples and would make the
       schedule builder reject "warmup + decay > period".
    4. Target batch size pinned to 16 384 tokens (one instance pair at
       seq=8192) so each optimizer step is one grad-accum step. Production
       batch size is ~475 K tokens which would mean ~30 grad-accum steps per
       optimizer step on a single GPU.
    """

    def __post_init__(self):
        if self.chinchilla_multiple <= 0:
            raise OLMoConfigurationError("'chinchilla_multiple' must be positive")
        if not (0 < self.decay_fraction < 0.5):
            raise OLMoConfigurationError(
                "'decay_fraction' must be greater than 0.0 and less than 0.5"
            )

    def configure_chinchilla_periods(self, num_params: int) -> tuple[int, list[float]]:
        return 16384, [self.chinchilla_multiple]

    def configure_target_batch_size(self, num_params: int) -> int:
        return 16384


@dataclasses.dataclass(kw_only=True, eq=True)
class Olmo3SmokeConfigurator(Olmo3ModelConfigurator):
    """``Olmo3ModelConfigurator`` with a ``smoke_1gpu`` flag that overrides
    the parent's hardcoded 8-GPU minimum device-mesh spec to allow a
    single-GPU run.

    At 190 M parameters microbatch sizing transfers between 1-GPU and 8-GPU
    because the binding memory constraint is activation memory (the
    ``(mbz, seq, vocab)`` fp32 tensor materialized at the LM head), which
    isn't sharded under FSDP. So a 1-GPU smoke faithfully reproduces the
    per-GPU memory profile of the 8-GPU production run.
    """

    smoke_1gpu: bool = False

    def configure_minimal_device_mesh_spec(
        self,
        *,
        size_spec,
        sequence_length: int,
        device_type: str,
    ) -> DeviceMeshSpec:
        if self.smoke_1gpu:
            return DeviceMeshSpec(world_size=1, dp_world_size=1)
        return super().configure_minimal_device_mesh_spec(
            size_spec=size_spec,
            sequence_length=sequence_length,
            device_type=device_type,
        )


def add_smoke_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    """Register ``--smoke-1gpu`` on ``parser``. Designed to be passed as the
    ``add_additional_args`` kwarg to ``olmo_core.internal.ladder.main``.
    """
    parser.add_argument(
        "--smoke-1gpu",
        action="store_true",
        help=(
            "Run on a single GPU for fast end-to-end smoke testing. Overrides "
            "the configurator's 8-GPU minimum and forces max_devices=1. Pair "
            "with --chinchilla-multiple 0.001 for a ~3-5 minute smoke. "
            "Doesn't catch distributed-only bugs (NCCL ordering, FSDP shard "
            "semantics) — those still need the 8-GPU smoke."
        ),
    )
