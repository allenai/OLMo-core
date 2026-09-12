import json
from pathlib import Path

import pytest

from olmo_core.data.multimodal.alignment import (
    MultimodalMixtureConfig,
    MultimodalSourceConfig,
)
from olmo_core.data.multimodal.mixtures.vision_alignment import (
    expected_loss_mass,
    sampling_weights_from_loss_mass,
)
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.data.multimodal.pretraining_replay import PretrainingReplayConfig
from olmo_core.data.multimodal.vision_alignment_perception import (
    VisionAlignmentAuditedAlignmentDatasetConfig,
    VisionAlignmentOcrDocumentDatasetConfig,
)
from olmo_core.data.numpy_dataset import InstanceFilterConfig, NumpyFSLDatasetConfig
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.internal.vision_alignment_data import ALIGNMENT_LOSS_TARGETS
from olmo_core.internal.vision_midtraining_data import (
    DEFAULT_VISUAL_EXAMPLE_WEIGHTS,
    DEFAULT_VISUAL_LOSS_SHARES,
    DEFAULT_VISUAL_MEAN_LOSS_WEIGHTS,
    build_dataset,
    build_visual_sources,
    loss_mass_targets,
)


@pytest.fixture
def artifacts(tmp_path):
    sources = {
        name: {
            split: {
                "physical_split": "train" if name == "audited_alignment" else split,
                "selection": {"path": f"selections/{name}-{split}.indices"},
            }
            for split in ("train", "validation")
        }
        for name in ALIGNMENT_LOSS_TARGETS["perception"]
    }
    selection_root = tmp_path / "perception-provenance-v2"
    selection_root.mkdir()
    (selection_root / "vision-alignment-perception-provenance.json").write_text(
        json.dumps(
            {
                "source_spec": {
                    "finevision_root": "/finevision",
                    "finevision_visualweb_fingerprint": "prepared-visualweb",
                    "finevision_geo170k_fingerprint": "prepared-geo170k",
                    "ocr_source_names": ["text_vqa", "doc_qa", "info_qa", "chart_qa_weighted"],
                },
                "sources": sources,
            }
        )
    )
    return str(tmp_path)


@pytest.fixture
def means():
    names = [*DEFAULT_VISUAL_EXAMPLE_WEIGHTS, *DEFAULT_VISUAL_LOSS_SHARES, "text_midtraining"]
    return {name: float(2**i) for i, name in enumerate(names)}


@pytest.mark.parametrize("length,crops", [(8192, 8), (4096, 4)])
@pytest.mark.parametrize("split", ["train", "validation"])
def test_visual_sources_preserve_formats_and_reviewed_populations(artifacts, length, crops, split):
    sources = build_visual_sources(
        sequence_length=length,
        max_crops=crops,
        alignment_artifact_root=artifacts,
        midtraining_artifact_root=artifacts,
        split=split,
    )
    assert set(sources) == set(DEFAULT_VISUAL_EXAMPLE_WEIGHTS) | set(DEFAULT_VISUAL_LOSS_SHARES)
    assert len(sources) == 7
    for source in sources.values():
        config = source.dataset if isinstance(source, MultimodalSourceConfig) else source
        assert config.max_sequence_length == length
        assert config.max_crops == crops
        assert config.message_format == "document"
        assert config.loss_token_weighting == "none"

    ocr = sources["ocr_document"]
    assert isinstance(ocr, MultimodalSourceConfig)
    assert isinstance(ocr.dataset, VisionAlignmentOcrDocumentDatasetConfig)
    assert ocr.dataset.source_names == ("text_vqa", "doc_qa", "info_qa", "chart_qa_weighted")
    assert ocr.dataset.split == split
    assert ocr.dataset.skip_bad_rows is False
    assert ocr.selection_path.endswith(f"ocr_document-{split}.indices")
    assert ocr.excluded_selection_paths == []

    audited = sources["audited_alignment"]
    assert isinstance(audited, MultimodalSourceConfig)
    assert isinstance(audited.dataset, VisionAlignmentAuditedAlignmentDatasetConfig)
    assert audited.dataset.split == "train"
    assert audited.dataset.visualweb_path.endswith("visualwebinstruct-filtered")
    assert audited.dataset.geo170k_path.endswith("geo170k-align")
    assert audited.dataset.visualweb_fingerprint == "prepared-visualweb"
    assert audited.dataset.geo170k_fingerprint == "prepared-geo170k"
    assert audited.dataset.min_formatting == 4
    assert audited.dataset.min_visual_dependency == 4
    assert audited.dataset.min_relevance == 4
    assert audited.selection_path.endswith(f"audited_alignment-{split}.indices")
    other_split = "validation" if split == "train" else "train"
    assert len(audited.excluded_selection_paths) == 1
    assert audited.excluded_selection_paths[0].endswith(f"audited_alignment-{other_split}.indices")

    assert sources["pixmo_cap"].mode == "transcript_and_caption"
    assert sources["pixmo_cap"].style_length_conditioning is True
    assert sources["pixmo_count"].scalar_count_replay is True
    assert sources["pixmo_count"].mode == "grounded"
    assert sources["pixmo_count"].split == (
        "grounded_validation" if split == "validation" else "train"
    )
    for name in ("pixmo_points_basic", "pixmo_points_high_frequency"):
        assert sources[name].counting == "both"
        assert sources[name].both_mode == "duplicate"


@pytest.mark.parametrize("target_text", [0.5, 0.8, 0.9])
@pytest.mark.parametrize("mean_scale", [0.01, 1.0, 100.0])
def test_two_stage_mixer_retains_loss_shares_and_example_ratios(means, target_text, mean_scale):
    means = {name: value * mean_scale ** (i % 3) for i, (name, value) in enumerate(means.items())}
    targets = loss_mass_targets(means, target_text_loss_mass=target_text)
    probabilities = sampling_weights_from_loss_mass(targets, means)
    actual_mass = expected_loss_mass(probabilities, means)
    assert sum(targets.values()) == pytest.approx(1.0)
    assert actual_mass["text_midtraining"] == pytest.approx(target_text)
    visual_mass = 1.0 - target_text
    assert actual_mass["ocr_document"] / visual_mass == pytest.approx(8 / 65)
    assert actual_mass["audited_alignment"] / visual_mass == pytest.approx(5 / 65)
    example_mass = sum(actual_mass[name] for name in DEFAULT_VISUAL_EXAMPLE_WEIGHTS)
    assert example_mass / visual_mass == pytest.approx(0.8)
    example_probability = sum(probabilities[name] for name in DEFAULT_VISUAL_EXAMPLE_WEIGHTS)
    example_weight = sum(DEFAULT_VISUAL_EXAMPLE_WEIGHTS.values())
    for name, weight in DEFAULT_VISUAL_EXAMPLE_WEIGHTS.items():
        assert probabilities[name] / example_probability == pytest.approx(weight / example_weight)


def test_explicit_visual_loss_shares(means):
    targets = loss_mass_targets(
        means, visual_loss_shares={"ocr_document": 0.25, "audited_alignment": 0.15}
    )
    assert targets["text_midtraining"] == pytest.approx(0.9)
    assert targets["ocr_document"] == pytest.approx(0.025)
    assert targets["audited_alignment"] == pytest.approx(0.015)
    assert sum(targets[name] for name in DEFAULT_VISUAL_EXAMPLE_WEIGHTS) == pytest.approx(0.06)


@pytest.mark.parametrize("bad_value", [0, -1, True, float("inf"), float("nan")])
def test_reject_invalid_calibration_means(means, bad_value):
    means["ocr_document"] = bad_value
    with pytest.raises(ValueError):
        loss_mass_targets(means)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_reject_mismatched_calibration_sources(means, change):
    if change == "missing":
        means.pop("ocr_document")
    else:
        means["native_text_replay"] = 100.0
    with pytest.raises(ValueError):
        loss_mass_targets(means)


@pytest.mark.parametrize("ratio", [-0.1, 1.1, True, False, float("inf"), float("nan")])
def test_reject_invalid_text_ratios(means, ratio):
    with pytest.raises(ValueError):
        loss_mass_targets(means, target_text_loss_mass=ratio)


@pytest.mark.parametrize(
    "shares",
    [
        {},
        {"ocr_document": 0.1},
        {"ocr_document": 0.1, "audited_alignment": 0.1, "unknown": 0.1},
        {"ocr_document": 0, "audited_alignment": 0.1},
        {"ocr_document": -0.1, "audited_alignment": 0.1},
        {"ocr_document": True, "audited_alignment": 0.1},
        {"ocr_document": float("nan"), "audited_alignment": 0.1},
        {"ocr_document": 0.5, "audited_alignment": 0.5},
        {"ocr_document": 0.8, "audited_alignment": 0.4},
    ],
)
def test_reject_invalid_or_uncalibrated_visual_shares(means, shares):
    with pytest.raises(ValueError):
        loss_mass_targets(means, visual_loss_shares=shares)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequence_length": 1},
        {"sequence_length": 8192.0},
        {"max_crops": 0},
        {"max_crops": True},
        {"split": "all"},
    ],
)
def test_reject_invalid_visual_config(artifacts, kwargs):
    with pytest.raises(ValueError):
        build_visual_sources(alignment_artifact_root=artifacts, **kwargs)


@pytest.mark.parametrize("mismatch", ["missing", "extra", "sequence_length"])
def test_reject_inconsistent_visual_sources(artifacts, means, mismatch):
    visual = build_visual_sources(alignment_artifact_root=artifacts)
    if mismatch == "missing":
        visual.pop("ocr_document")
    elif mismatch == "extra":
        visual["extra"] = visual["ocr_document"].copy()
    else:
        visual["ocr_document"].dataset.max_sequence_length = 4096
    text = NumpyFSLDatasetConfig(
        tokenizer=TokenizerConfig(vocab_size=128, eos_token_id=126, pad_token_id=127),
        sequence_length=8192,
        paths=["/unused.npy"],
    )
    with pytest.raises(ValueError):
        build_dataset(text, means, visual_sources=visual)


def test_explicit_text_mixture_roundtrip_and_no_caller_mutation(artifacts, means):
    text = NumpyFSLDatasetConfig(
        tokenizer=TokenizerConfig(
            identifier="test-tokenizer", vocab_size=128, eos_token_id=126, pad_token_id=127
        ),
        sequence_length=8192,
        dtype=NumpyDatasetDType.uint16,
        work_dir="/text-cache",
        source_mixture_config=SourceMixtureDatasetConfig(
            source_list=SourceMixtureList(
                sources=[
                    SourceMixtureConfig("a", 0.25, ["/a.npy"], max_source_fraction=0.5),
                    SourceMixtureConfig("b", 0.75, ["/b.npy"], max_repetition_ratio=2.0),
                ]
            ),
            requested_tokens=50_000_297_984,
            global_batch_size=1_048_576,
            seed=73,
            render_tables=False,
        ),
    )
    visual = build_visual_sources(
        alignment_artifact_root=artifacts, midtraining_artifact_root=artifacts
    )
    original_text = text.as_config_dict()
    original_visual = {name: source.as_config_dict() for name, source in visual.items()}
    original_means = means.copy()
    dataset = build_dataset(
        text,
        means,
        visual_sources=visual,
        tokenizer_revision="tokenizer-revision",
        tokenizer_cache_dir="/tokenizer-cache",
        model_vocab_size=256,
    )
    dataset.validate()
    assert set(dataset.sources) == set(visual) | {"text_midtraining"}
    replay = dataset.sources["text_midtraining"]
    assert isinstance(replay, PretrainingReplayConfig)
    assert replay.checkpoint is None
    assert replay.split == "all"
    assert replay.dataset is not text
    assert replay.resolve_dataset().as_config_dict() == original_text
    assert dataset.tokenizer == text.tokenizer
    assert dataset.tokenizer_revision == "tokenizer-revision"
    assert dataset.tokenizer_cache_dir == "/tokenizer-cache"
    assert dataset.model_vocab_size == 256
    assert expected_loss_mass(dataset.sampling_weights(), means)[
        "text_midtraining"
    ] == pytest.approx(0.9)
    encoded = dataset.as_config_dict()
    assert MultimodalMixtureConfig.from_dict(encoded).as_config_dict() == encoded

    replay.dataset.source_mixture_config.seed = 123
    dataset.sources["ocr_document"].dataset.max_crops = 1
    dataset.mean_loss_weight["ocr_document"] = 123.0
    assert text.as_config_dict() == original_text
    assert {name: source.as_config_dict() for name, source in visual.items()} == original_visual
    assert means == original_means


def test_default_visual_means_match_bounded_calibration():
    calibration_path = (
        Path(__file__).resolve().parents[3]
        / "configs/vision_moe/vision_midtraining/visual_calibration_v1.json"
    )
    calibration = json.loads(calibration_path.read_text())
    assert DEFAULT_VISUAL_MEAN_LOSS_WEIGHTS == calibration["mean_loss_weight"]


@pytest.mark.parametrize("text_share", [0.0, 0.5, 0.9])
def test_custom_visual_groups_and_endpoint(text_share):
    means = {"captions": 200.0, "points": 100.0, "documents": 5.0}
    if text_share > 0:
        means["text_midtraining"] = 8191.0
    targets = loss_mass_targets(
        means,
        target_text_loss_mass=text_share,
        visual_example_weights={"captions": 3.0, "points": 1.0},
        visual_loss_shares={"documents": 0.2},
    )
    probabilities = sampling_weights_from_loss_mass(targets, means)
    actual = expected_loss_mass(probabilities, means)
    assert actual.get("text_midtraining", 0.0) == pytest.approx(text_share)
    assert actual["documents"] == pytest.approx((1.0 - text_share) * 0.2)
    assert probabilities["captions"] / probabilities["points"] == pytest.approx(3.0)
    assert sum(actual.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "examples,shares",
    [({"visual": 2.0}, {}), ({}, {"visual": 1.0})],
)
def test_single_visual_group(examples, shares):
    means = {"visual": 10.0, "text_midtraining": 8191.0}
    assert loss_mass_targets(
        means, visual_example_weights=examples, visual_loss_shares=shares
    ) == pytest.approx({"visual": 0.1, "text_midtraining": 0.9})


@pytest.mark.parametrize(
    "examples,shares",
    [
        ({"visual": 0.0}, {}),
        ({"visual": True}, {}),
        ({"visual": float("nan")}, {}),
        ({"visual": float("inf")}, {}),
        ({"visual": 1.0}, {"visual": 0.1}),
        ({"text_midtraining": 1.0}, {}),
        ({}, {"text_midtraining": 1.0}),
        ({}, {}),
        ({}, {"visual": 0.5}),
        ({}, {"visual": 1.1}),
        ({"": 1.0}, {}),
    ],
)
def test_invalid_visual_weight_groups(examples, shares):
    with pytest.raises(ValueError):
        loss_mass_targets(
            {"visual": 10.0, "text_midtraining": 8191.0},
            visual_example_weights=examples,
            visual_loss_shares=shares,
        )


def _text_config(**kwargs):
    return NumpyFSLDatasetConfig(
        tokenizer=TokenizerConfig(vocab_size=128, eos_token_id=126, pad_token_id=127),
        sequence_length=8192,
        paths=["/unused.npy"],
        **kwargs,
    )


@pytest.mark.parametrize("masked", [False, True])
def test_t100_never_constructs_visual_sources_or_prepares_datasets(monkeypatch, masked):
    from olmo_core.internal import vision_midtraining_data

    def unexpected(*args, **kwargs):
        pytest.fail("T100 config construction must not access visual sources or prepare datasets")

    monkeypatch.setattr(vision_midtraining_data, "build_visual_sources", unexpected)
    monkeypatch.setattr(vision_midtraining_data, "build_alignment_sources", unexpected)
    monkeypatch.setattr(NumpyFSLDatasetConfig, "build", unexpected)
    monkeypatch.setattr(SourceMixtureDatasetConfig, "build", unexpected)
    text = _text_config(
        label_mask_paths=["/unused-mask.npy"] if masked else None,
        instance_filter_config=InstanceFilterConfig(),
    )
    original = text.as_config_dict()
    dataset = build_dataset(text, target_text_loss_mass=1.0)
    assert list(dataset.sources) == ["text_midtraining"]
    assert dataset.target_loss_mass == {"text_midtraining": 1.0}
    assert dataset.sampling_weights() == {"text_midtraining": 1.0}
    assert dataset.mean_loss_weight == {"text_midtraining": 1.0 if masked else 8191.0}
    assert dataset.sources["text_midtraining"].resolve_dataset().as_config_dict() == original
    assert text.as_config_dict() == original
    assert MultimodalMixtureConfig.from_dict(dataset.as_config_dict()) == dataset
    assert loss_mass_targets(target_text_loss_mass=1.0) == {"text_midtraining": 1.0}


def test_t100_ignores_inactive_visual_configuration():
    dataset = build_dataset(
        _text_config(),
        {"unused_visual": float("nan")},
        target_text_loss_mass=1.0,
        visual_sources={"unused_visual": None},
        visual_example_weights={"unused_visual": float("nan")},
        visual_loss_shares={"unused_visual": -1.0},
    )
    assert dataset.sampling_weights() == {"text_midtraining": 1.0}


@pytest.mark.parametrize("text_share", [0.0, 0.9])
def test_build_dataset_accepts_arbitrary_visual_names(text_share):
    visual = {
        name: PixMoCapDatasetConfig(dataset_path="/unused", max_sequence_length=8192)
        for name in ("captions", "documents")
    }
    dataset = build_dataset(
        _text_config(),
        {"captions": 100.0, "documents": 5.0},
        visual_sources=visual,
        target_text_loss_mass=text_share,
        visual_example_weights={"captions": 1.0},
        visual_loss_shares={"documents": 0.2},
    )
    assert set(dataset.sources) == set(visual) | ({"text_midtraining"} if text_share else set())
    actual = expected_loss_mass(dataset.sampling_weights(), dataset.mean_loss_weight)
    assert actual.get("text_midtraining", 0.0) == pytest.approx(text_share)
    assert actual["documents"] == pytest.approx((1 - text_share) * 0.2)


def test_masked_mixed_text_requires_mean_and_preserves_masks():
    text = _text_config(
        label_mask_paths=["/unused-mask.npy"], instance_filter_config=InstanceFilterConfig()
    )
    kwargs = {
        "visual_sources": {
            "captions": PixMoCapDatasetConfig(dataset_path="/unused", max_sequence_length=8192)
        },
        "visual_example_weights": {"captions": 1.0},
        "visual_loss_shares": {},
    }
    with pytest.raises(ValueError, match="Masked mixed text requires"):
        build_dataset(text, {"captions": 100.0}, **kwargs)
    dataset = build_dataset(text, {"captions": 100.0, "text_midtraining": 200.0}, **kwargs)
    assert dataset.sources["text_midtraining"].dataset.as_config_dict() == text.as_config_dict()
    assert dataset.mean_loss_weight["text_midtraining"] == 200.0


def test_standard_source_mixture_config_construction_is_lazy(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Config construction must not allocate or open text arrays")

    monkeypatch.setattr(SourceMixtureDatasetConfig, "build", unexpected)
    monkeypatch.setattr(NumpyFSLDatasetConfig, "build", unexpected)
    monkeypatch.setattr(SourceMixtureConfig, "resolved_paths", property(unexpected))
    text = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=SourceMixtureList([SourceMixtureConfig("text", 1.0, ["/unused/*.npy"])]),
            requested_tokens=50_000_297_984,
            global_batch_size=1_048_576,
            seed=1337,
        ),
        tokenizer=TokenizerConfig.dolma2(),
        sequence_length=8192,
    )
    dataset = build_dataset(text, target_text_loss_mass=1.0)
    assert dataset.sources["text_midtraining"].dataset.source_mixture_config == (
        text.source_mixture_config
    )
