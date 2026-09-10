"""``_append_extra_sft_sources`` must apply to every mixture tier, not just plain v9.

The call used to sit inside the final ``else`` of ``_build_mixture``, i.e. only on the
``image-only-v9`` path, so ``--mmfinereason_rate`` silently did nothing on
``single-image-only-v9``, on every v10/v11 tier, and on all 16 per-source ablation tiers.
A rate flag that is accepted and then ignored produces a plausible-looking run at the
wrong mixture, which is why this is a test and not a comment.

The dataset configs are stubbed: the real ones memory-map Arrow off weka, and what is
under test here is the weight arithmetic and the guards, not data loading.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "train" / "Molmo2-Stage2.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("molmo2_stage2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_script()


class _StubConfig:
    """Stands in for CaptionDatasetConfig / ChartVerseDatasetConfig / MMFineReason..."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build(self, _tokenizer):
        return ("stub", self.kwargs)


@pytest.fixture
def stub_dataset_configs(monkeypatch):
    import olmo_core.data.multimodal as mm

    for name in (
        "CaptionDatasetConfig",
        "ChartVerseDatasetConfig",
        "MMFineReasonDatasetConfig",
        "FineVisionDatasetConfig",
    ):
        monkeypatch.setattr(mm, name, _StubConfig, raising=True)


def _config(mod, **overrides):
    kwargs = dict(mixture="single-image-only-v9")
    kwargs.update(overrides)
    cfg = object.__new__(mod.ExperimentConfig)
    # Only the fields _append_extra_sft_sources reads; building a full ExperimentConfig
    # would construct a model config and load a tokenizer.
    cfg.mixture = kwargs["mixture"]
    cfg.mmfinereason_rate = kwargs.get("mmfinereason_rate", 0.0)
    cfg.finevision_rate = kwargs.get("finevision_rate", 0.0)
    cfg.caption_subsets = kwargs.get("caption_subsets", [])
    cfg.caption_rate = kwargs.get("caption_rate", 0.0)
    cfg.chartverse_rate = kwargs.get("chartverse_rate", 0.0)
    cfg.chartverse_subset = kwargs.get("chartverse_subset", "sft_600k")
    return cfg


BASE_NAMES = ["pixmo_cap", "chart_qa_weighted", "tulu4"]
BASE_WEIGHTS = [0.5, 0.3, 0.2]


def test_no_rates_is_a_no_op(mod, stub_dataset_configs):
    cfg = _config(mod)
    datasets, weights, names = mod._append_extra_sft_sources(
        cfg, None, [1, 2, 3], list(BASE_WEIGHTS), list(BASE_NAMES)
    )
    assert names == BASE_NAMES
    assert weights == BASE_WEIGHTS


def test_c1_composition_rescales_base_and_sums_to_one(mod, stub_dataset_configs):
    """The C1 arm: omniscience 35 / mmfinereason 15 / chartverse 15 / v9 replay 35."""
    cfg = _config(
        mod,
        caption_subsets=["omniscience"],
        caption_rate=0.35,
        mmfinereason_rate=0.15,
        chartverse_rate=0.15,
    )
    _datasets, weights, names = mod._append_extra_sft_sources(
        cfg, None, [1, 2, 3], list(BASE_WEIGHTS), list(BASE_NAMES)
    )

    assert sum(weights) == pytest.approx(1.0)
    by_name = dict(zip(names, weights))
    assert by_name["omniscience"] == pytest.approx(0.35)
    assert by_name["mmfinereason"] == pytest.approx(0.15)
    assert by_name["chartverse"] == pytest.approx(0.15)
    # The replay term is whatever is left, split in the base tier's own proportions.
    assert sum(by_name[n] for n in BASE_NAMES) == pytest.approx(0.35)
    assert by_name["pixmo_cap"] == pytest.approx(0.5 * 0.35)


def test_caption_rate_splits_evenly_across_subsets(mod, stub_dataset_configs):
    cfg = _config(mod, caption_subsets=["omniscience", "omniscience-full"], caption_rate=0.50)
    _datasets, weights, names = mod._append_extra_sft_sources(
        cfg, None, [1, 2, 3], list(BASE_WEIGHTS), list(BASE_NAMES)
    )
    by_name = dict(zip(names, weights))
    assert by_name["omniscience"] == pytest.approx(0.25)
    assert by_name["omniscience-full"] == pytest.approx(0.25)
    assert sum(weights) == pytest.approx(1.0)


def test_chartverse_subset_reaches_the_loader(mod, stub_dataset_configs):
    """Without this knob the 250k default is the only reachable copy."""
    cfg = _config(mod, chartverse_rate=0.65, chartverse_subset="sft_1800k")
    datasets, _weights, names = mod._append_extra_sft_sources(
        cfg, None, [1, 2, 3], list(BASE_WEIGHTS), list(BASE_NAMES)
    )
    built = dict(zip(names, datasets))["chartverse"]
    assert built[1]["subset"] == "sft_1800k"


def test_rates_summing_to_one_or_more_are_rejected(mod, stub_dataset_configs):
    cfg = _config(mod, caption_rate=0.7, caption_subsets=["omniscience"], chartverse_rate=0.3)
    with pytest.raises(ValueError, match="must be < 1"):
        mod._append_extra_sft_sources(cfg, None, [1, 2, 3], list(BASE_WEIGHTS), list(BASE_NAMES))


def test_appending_a_source_the_base_tier_already_has_is_rejected(mod, stub_dataset_configs):
    """v11 already contains chartverse and mmfinereason; double-counting must not be silent."""
    cfg = _config(mod, mixture="single-image-only-v11", chartverse_rate=0.15)
    with pytest.raises(ValueError, match="already in mixture"):
        mod._append_extra_sft_sources(
            cfg, None, [1, 2, 3, 4], BASE_WEIGHTS + [0.0], BASE_NAMES + ["chartverse"]
        )
