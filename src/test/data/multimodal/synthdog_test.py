"""CPU tests for the SynthDoG synthetic-document transcription source: ground-truth parsing,
split globbing, the style tag, example layout, epoch rotation, and the OCR registry / Molmo2-Stage1
wiring."""

import importlib.util
import io
import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from olmo_core.data.multimodal import SynthDogDatasetConfig
from olmo_core.data.multimodal.mixtures import ocr as ocr_mix
from olmo_core.data.multimodal.ocr_caption_tars import OcrCaptionTarsDatasetConfig
from olmo_core.data.multimodal.olmocr import OLMOCR_STYLE, OlmOcrMixDatasetConfig
from olmo_core.data.multimodal.synthdog import extract_text_sequence
from olmo_core.exceptions import OLMoConfigurationError


class _FakeTok:
    eos_token_id = 1
    bos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 90) + 10 for c in text]


# ---------------------------------------------------------------------------
# ground_truth envelope
# ---------------------------------------------------------------------------


def test_extract_text_sequence_reads_the_donut_envelope():
    """The whole target is the value of ``text_sequence``; the rest is Donut's task wrapper."""
    gt = json.dumps({"gt_parse": {"text_sequence": "  Dares Wins Vol. 5  "}})
    assert extract_text_sequence(gt) == "Dares Wins Vol. 5"
    # A top-level `text_sequence` is accepted too, though the corpus never uses that shape.
    assert extract_text_sequence(json.dumps({"text_sequence": "bare"})) == "bare"


@pytest.mark.parametrize(
    "gt",
    [
        "not json",
        json.dumps(["not", "an", "object"]),
        json.dumps({}),  # no gt_parse
        json.dumps({"gt_parse": {}}),  # no text_sequence
        json.dumps({"gt_parse": {"text_sequence": None}}),
        json.dumps({"gt_parse": {"text_sequence": "   "}}),  # blank
        json.dumps({"gt_parse": "a string, not a dict"}),
        json.dumps({"gt_parse": {"text_sequence": ["not", "a", "string"]}}),
    ],
)
def test_extract_text_sequence_rejects_unusable_envelopes(gt):
    """Every rejection is a row the loader skips rather than a crash."""
    with pytest.raises(ValueError):
        extract_text_sequence(gt)


# ---------------------------------------------------------------------------
# Fixture: a two-shard repo laid out like `hf download` leaves it
# ---------------------------------------------------------------------------

TRAIN_TEXTS = [
    "Closin g Time miniserie s.",  # SynthDoG's line-wrap artefacts, kept verbatim
    "nary rule does not apply to evidence foun d",
    "word " * 400,  # long page, for the truncation test
]
VALIDATION_TEXTS = ["ession of Ameri cani sm."]


def _jpeg(seed: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (48 + 4 * seed, 60), color=(200, 200 - seed, 180)).save(buf, format="JPEG")
    return buf.getvalue()


def _write_repo(tmp_path, *, bad_row: bool = False):
    root = tmp_path / "synthdog-en"
    (root / "data").mkdir(parents=True)

    def shard(path, texts, start):
        rows = {
            "image": [{"bytes": _jpeg(start + i), "path": None} for i in range(len(texts))],
            "ground_truth": [json.dumps({"gt_parse": {"text_sequence": t}}) for t in texts],
        }
        if bad_row:
            rows["image"].append({"bytes": _jpeg(99), "path": None})
            rows["ground_truth"].append("{not json")
        pq.write_table(pa.table(rows), str(path))

    # Two train shards, so the split glob has to cover both.
    shard(root / "data" / "train-00000-of-00002-abc.parquet", TRAIN_TEXTS[:2], 0)
    shard(root / "data" / "train-00001-of-00002-def.parquet", TRAIN_TEXTS[2:], 2)
    shard(root / "data" / "validation-00000-of-00001-ghi.parquet", VALIDATION_TEXTS, 9)
    return str(root)


def _cfg(root, **kw):
    kw.setdefault("max_crops", 1)
    return SynthDogDatasetConfig(dataset_path=root, **kw)


# ---------------------------------------------------------------------------
# Config / loading
# ---------------------------------------------------------------------------


def test_config_validation():
    with pytest.raises(OLMoConfigurationError):
        SynthDogDatasetConfig(dataset_path="").validate()
    with pytest.raises(OLMoConfigurationError):
        SynthDogDatasetConfig(split="test").validate()
    with pytest.raises(OLMoConfigurationError):
        SynthDogDatasetConfig(style="").validate()
    with pytest.raises(OLMoConfigurationError):
        SynthDogDatasetConfig(system_prompt="uber_model_v2").validate()
    SynthDogDatasetConfig(split="validation").validate()


def test_split_glob_does_not_pool_train_with_validation(tmp_path):
    """Both splits' shards share one `data/` directory, so a `*.parquet` glob would silently
    train on the validation pages too."""
    root = _write_repo(tmp_path)
    tok = _FakeTok()
    train = _cfg(root).build(tok)
    val = _cfg(root, split="validation").build(tok)
    assert len(train) == len(TRAIN_TEXTS)  # both train shards, and only those
    assert len(val) == len(VALIDATION_TEXTS)
    assert train.shard_glob.endswith("train-*.parquet")
    # `extract_text_sequence` strips, so compare against stripped fixtures.
    assert {train.transcription(train._data[i]) for i in range(len(train))} == {
        t.strip() for t in TRAIN_TEXTS
    }
    assert val.transcription(val._data[0]) == VALIDATION_TEXTS[0]


def test_missing_repo_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _cfg(str(tmp_path / "nope")).build(_FakeTok())


def test_config_fields_merge_from_cli():
    cfg = SynthDogDatasetConfig().merge(
        ["split=validation", "style=olmocr", "max_sequence_length=2560"]
    )
    assert (cfg.split, cfg.style, cfg.max_sequence_length) == ("validation", "olmocr", 2560)
    assert SynthDogDatasetConfig().style == OLMOCR_STYLE  # the default is the shared OCR tag


# ---------------------------------------------------------------------------
# Prompt / example
# ---------------------------------------------------------------------------


def test_prompt_is_the_shared_ocr_transcription_tag(tmp_path):
    """Same task as the olmOCR-mix pages, so the same tag: the user turn is ``olmocr:``."""
    root = _write_repo(tmp_path)
    ds = _cfg(root).build(_FakeTok())
    assert ds.config.style == OLMOCR_STYLE == "olmocr"
    assert {ds.user_prompt("x", np.random.RandomState(s)) for s in range(5)} == {"olmocr:"}
    # Separable if the wrap artefacts ever need their own tag.
    split_off = _cfg(root, style="synthdog").build(_FakeTok())
    assert split_off.user_prompt("x", np.random.RandomState(0)) == "synthdog:"
    v2 = _cfg(root, system_prompt="style_and_length_v2").build(_FakeTok())
    prompts = {v2.user_prompt("x" * 300, np.random.RandomState(s)) for s in range(20)}
    assert all(p.startswith("olmocr") and p.endswith(":") for p in prompts)
    assert any(" " in p for p in prompts)  # the length bucket


def test_example_layout(tmp_path):
    root = _write_repo(tmp_path)
    tok = _FakeTok()
    ds = _cfg(root, loss_token_weighting="none").build(tok)
    ex = ds[0]
    for key in ("input_ids", "labels", "loss_masks", "position_ids", "token_type_ids", "images"):
        assert key in ex
    assert "subsegment_ids" not in ex  # one transcription per page -> single branch
    text_ids = ex["input_ids"][ex["token_type_ids"] == 0].tolist()
    tag = tok.encode("olmocr:")
    assert any(text_ids[i : i + len(tag)] == tag for i in range(len(text_ids)))
    target = tok.encode(TRAIN_TEXTS[0])
    assert ex["loss_masks"].sum() == pytest.approx(len(target) + 1)  # + the EOS target
    assert ex["labels"][ex["loss_masks"] > 0][-1] == tok.eos_token_id
    np.testing.assert_array_equal(ex["input_ids"], ds[0]["input_ids"])  # deterministic


def test_message_weight_and_truncation(tmp_path):
    root = _write_repo(tmp_path)
    tok = _FakeTok()
    base = _cfg(root, loss_token_weighting="none").build(tok)[2]
    weighted = _cfg(root, loss_token_weighting="none", message_weight=0.4).build(tok)[2]
    nz = base["loss_masks"] > 0
    np.testing.assert_allclose(weighted["loss_masks"][nz], 0.4 * base["loss_masks"][nz])
    n_image = int((base["token_type_ids"] == 1).sum())
    cut = _cfg(root, max_sequence_length=n_image + 120).build(tok)[2]
    assert len(cut["input_ids"]) == n_image + 120
    assert (cut["token_type_ids"] == 1).sum() == n_image  # the image block is never cut


def test_unusable_row_is_skipped_not_raised(tmp_path):
    """An unparseable envelope is a data error, not a reason to spend the loader's error budget."""
    root = _write_repo(tmp_path, bad_row=True)
    ds = _cfg(root).build(_FakeTok())
    assert len(ds) == len(TRAIN_TEXTS) + 2  # two bad rows, one per shard
    with pytest.raises(ValueError):
        ds._build(2)  # the bad row of the first shard
    np.testing.assert_array_equal(ds[2]["input_ids"], ds[3]["input_ids"])  # skipped to the next


def test_epoch_rotation_is_wired(tmp_path):
    """Nothing in this source samples per example today, but the epoch has to reach it: the mixin
    is what a later per-example draw (an augmentation, a render size) would rely on."""
    root = _write_repo(tmp_path)
    ds = _cfg(root).build(_FakeTok())
    assert ds.epoch_rng(0).randint(10**9) == ds.epoch_rng(0).randint(10**9)
    first = ds.epoch_rng(0).randint(10**9)
    ds.set_epoch(3)
    assert ds.epoch_rng(0).randint(10**9) != first


# ---------------------------------------------------------------------------
# Registry / Stage-1 wiring
# ---------------------------------------------------------------------------


def test_registry_registers_synthdog_as_opt_in():
    assert ocr_mix.SYNTHDOG_SOURCES == {"synthdog_en": "train"}
    assert "synthdog_en" in ocr_mix.OCR_SOURCE_NAMES
    assert "synthdog_en" not in ocr_mix.DEFAULT_OCR_SOURCES
    assert "synthdog_en" in ocr_mix.OPT_IN_OCR_SOURCES
    assert set(ocr_mix.DEFAULT_OCR_SOURCES) == set(ocr_mix.OCR_SOURCE_NAMES) - set(
        ocr_mix.OPT_IN_OCR_SOURCES
    )
    assert len(ocr_mix.OCR_SOURCE_NAMES) == len(set(ocr_mix.OCR_SOURCE_NAMES)) == 22


def test_build_ocr_source_fills_the_synthdog_template(tmp_path):
    root = _write_repo(tmp_path)
    template = SynthDogDatasetConfig(dataset_path=root, max_crops=1, split="validation")
    ds = ocr_mix.build_ocr_source(
        "synthdog_en",
        _FakeTok(),
        olmocr=OlmOcrMixDatasetConfig(),
        tars=OcrCaptionTarsDatasetConfig(),
        synthdog=template,
    )
    # The registry owns the split; everything else comes from the template.
    assert ds.config.split == "train" and ds.config.dataset_path == root
    assert ds.config.max_crops == 1 and len(ds) == len(TRAIN_TEXTS)


def _load_stage1_module():
    try:
        import olmo_core.internal.common  # noqa: F401  (needs a recent beaker-py)
    except ImportError as e:  # pragma: no cover - env-dependent
        pytest.skip(f"Molmo2-Stage1.py imports fail here: {e}")
    spec = importlib.util.spec_from_file_location(
        "_stage1_synthdog", "src/scripts/train/Molmo2-Stage1.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_synthdog"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def test_stage1_exposes_the_synthdog_template():
    mod = _load_stage1_module()
    fields = {f.name for f in mod.ExperimentConfig.__dataclass_fields__.values()}
    assert "synthdog" in fields
    cfg = mod.build_config("x.py", "run", ["--ocr_rate=0.1", "--ocr_sources=[synthdog_en]"])
    assert cfg.ocr_sources == ("synthdog_en",)
    # The group template inherits the OCR group's prompt family and sequence budget.
    assert cfg.synthdog.system_prompt == mod.OCR_SYSTEM_PROMPT
    assert cfg.synthdog.max_sequence_length == mod.SEQUENCE_LENGTH
    assert cfg.synthdog.loss_token_weighting == "none"


def test_stage1_refuses_a_per_source_synthdog_split():
    """`build_ocr_source` sets the split per source, so a value here would be silently ignored."""
    mod = _load_stage1_module()
    with pytest.raises(OLMoConfigurationError, match="per OCR source"):
        mod.build_config("x.py", "run", ["--ocr_rate=0.1", "--synthdog.split=validation"])
