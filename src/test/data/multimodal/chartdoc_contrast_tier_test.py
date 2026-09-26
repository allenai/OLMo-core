"""The `single-image-only-v9-no-chartdoc` contrast tier.

This tier exists to be the negative half of an A/B: same base mixture, one coherent family
of sources removed. Two properties have to hold for the comparison to mean anything, and
both fail *silently* if they break, so they are pinned here.
"""

from olmo_core.data.multimodal.mixtures.image_only_v9 import (
    CHART_DOC_MIXTURE_DATASETS,
    SINGLE_IMAGE_ONLY_V9_DATASETS,
    VALIDATION_MIXTURES,
)
from olmo_core.data.multimodal.mixtures.mixture_pack_profiles import (
    get_mixture_pack_profile,
    mixture_is_multi_image,
)
from olmo_core.data.multimodal.mixtures.tiers import mixture_source_names

BASE = "single-image-only-v9"
CONTRAST = "single-image-only-v9-no-chartdoc"


def test_tier_is_registered_in_the_v9_table():
    # `_build_mixture` dispatches on dict membership; a v9 tier in the v10 table routes to
    # the wrong builder and raises at data-build time, inside an already-launched job.
    assert CONTRAST in VALIDATION_MIXTURES


def test_contrast_removes_exactly_the_chart_doc_family():
    base = set(mixture_source_names(BASE))
    contrast = set(mixture_source_names(CONTRAST))
    assert contrast < base
    assert base - contrast == set(CHART_DOC_MIXTURE_DATASETS)


def test_chart_doc_names_all_exist_in_the_base_mixture():
    """A typo here would silently remove nothing and make the contrast a no-op A/A."""
    assert set(CHART_DOC_MIXTURE_DATASETS) <= set(SINGLE_IMAGE_ONLY_V9_DATASETS)


def test_pack_geometry_matches_the_base_tier():
    """Packing must not be a confound: a tier with no profile entry falls back to the
    multi-image 125-crop budget, which would change examples/pack and steps/epoch between
    the two arms while looking like a data-only difference."""
    assert not mixture_is_multi_image(CONTRAST)
    assert get_mixture_pack_profile(CONTRAST) == get_mixture_pack_profile(BASE)


def test_stage2_default_load_path_exists():
    """The stage-2 default init must point at a checkpoint that is actually there.

    It previously pointed at a personal path that had been deleted. With the trainer's
    stock `if_available` load strategy that produced no error -- stage 2 trained from
    uninitialized weights at a flat CE of 11.93 (= ln(vocab)) and looked like a bad model.
    Both the default and the strategy are fixed; this pins the default so the same rot
    cannot recur unnoticed.
    """
    import importlib.util
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[4] / "src/scripts/train/Molmo2-Stage2.py"
    spec = importlib.util.spec_from_file_location("molmo2_stage2", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["molmo2_stage2"] = module
    spec.loader.exec_module(module)

    from olmo_core.train import LoadStrategy
    from olmo_core.train.checkpoint import Checkpointer

    assert module.LOAD_STRATEGY == LoadStrategy.always
    assert Checkpointer.contains_checkpoint(
        module.DEFAULT_LOAD_PATH
    ), f"DEFAULT_LOAD_PATH {module.DEFAULT_LOAD_PATH!r} has no checkpoint"
