"""Config-level tests for the Stage-2 training script.

These cover two fixes whose whole point is that they live in the *config* layer, and
which would otherwise have no test that fails on revert:

* the ``--pack_sequences=false`` trap — a non-zero ``DL_NUM_WORKERS`` default made that
  flag an unconditional startup error;
* ``ignore_shuffle_algo_version_mismatch`` — the shuffle-version guard's escape hatch was
  a constructor kwarg with no route from the CLI, so every pre-guard checkpoint was
  un-resumable.

The script is not an importable module (it is ``src/scripts/train/Molmo2-Stage2.py``), so
it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "train" / "Molmo2-Stage2.py"


@pytest.fixture(scope="module")
def stage2():
    if not _SCRIPT.is_file():  # pragma: no cover - layout guard
        pytest.skip(f"Stage-2 script not found at {_SCRIPT}")
    spec = importlib.util.spec_from_file_location("_molmo2_stage2", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_molmo2_stage2"] = module
    spec.loader.exec_module(module)
    return module


def test_ignore_shuffle_algo_version_mismatch_is_a_config_field(stage2):
    import dataclasses

    names = {f.name for f in dataclasses.fields(stage2.ExperimentConfig)}
    assert "ignore_shuffle_algo_version_mismatch" in names


def test_pack_sequences_false_does_not_require_also_setting_dl_num_workers(stage2):
    """`--pack_sequences=false` alone must resolve, not raise."""
    from olmo_core.data.multimodal.mixture_data_loader import MixtureLoaderStrategy

    assert stage2.DL_NUM_WORKERS > 0, "precondition: the trap only exists with a non-zero default"

    cfg = _minimal_config(stage2, pack_sequences=False)
    assert cfg.effective_dl_num_workers == 0
    assert cfg.loader_strategy is MixtureLoaderStrategy.synchronous
    # The declared field is untouched, so the value survives a later merge.
    assert cfg.dl_num_workers == stage2.DL_NUM_WORKERS


def test_packed_default_uses_worker_processes(stage2):
    from olmo_core.data.multimodal.mixture_data_loader import MixtureLoaderStrategy

    cfg = _minimal_config(stage2, pack_sequences=True)
    assert cfg.effective_dl_num_workers == stage2.DL_NUM_WORKERS
    assert cfg.loader_strategy is MixtureLoaderStrategy.processes


def _minimal_config(stage2, **overrides):
    """An ``ExperimentConfig`` with only the loader-relevant fields realised.

    Building the real one needs the HF model config; these tests only exercise
    ``__post_init__`` and the derived properties, so construct the dataclass directly
    with placeholders for the heavyweight sub-configs.
    """
    import dataclasses

    kwargs = {}
    for field in dataclasses.fields(stage2.ExperimentConfig):
        if field.name in overrides:
            kwargs[field.name] = overrides[field.name]
        elif (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING  # type: ignore[misc]
        ):
            kwargs[field.name] = None  # required sub-config we don't exercise
    return stage2.ExperimentConfig(**kwargs)
