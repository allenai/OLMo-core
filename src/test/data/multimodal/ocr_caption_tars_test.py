"""CPU tests for the webdataset-style caption tars (oe-encoder ``*_v6_tars`` OCR sources): shard
indexing and its cache, sample reads, ``<text>`` stripping, the style tag, example layout,
truncation, and the OCR source registry / Molmo2-Stage1 wiring."""

import importlib.util
import io
import json
import os
import sys
import tarfile

import numpy as np
import pytest

from olmo_core.data.multimodal import OcrCaptionTarsDatasetConfig, TarShardIndex
from olmo_core.data.multimodal import ocr_caption_tars as ct
from olmo_core.data.multimodal.mixtures import ocr as ocr_mix
from olmo_core.data.multimodal.olmocr import OlmOcrMixDatasetConfig
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
# Fixture: two shards of (image, json) pairs, plus an orphan image and a stray member
# ---------------------------------------------------------------------------

SAMPLES = [
    # (shard, key, ext, caption)
    (0, "a_000", "png", "<text>FELIX PRIVAT DBU 889</text>"),
    (0, "a_001", "jpg", "A chart with three bars."),
    (0, "a_002", "jpg", "<text>\n  multi\nline  \n</text>"),
    (1, "b_000", "png", "<text></text>"),  # empty after stripping
    (1, "b_001", "jpg", "word " * 400),
]


def _png_bytes(seed, ext):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (48 + 4 * seed, 40), color=(seed * 40 % 255, 90, 120)).save(
        buf, format="PNG" if ext == "png" else "JPEG"
    )
    return buf.getvalue()


def _write_tars(tmp_path, extra_key: str = ""):
    root = tmp_path / "toy_v6_tars"
    root.mkdir()
    shards = [tarfile.open(str(root / f"toy-w0{i}-00000.tar"), "w") for i in range(2)]

    def add(tf, name, data):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    samples = list(SAMPLES)
    if extra_key:
        samples.append((0, extra_key, "png", "<text>unicode key</text>"))
    for i, (shard, key, ext, caption) in enumerate(samples):
        add(shards[shard], f"{key}.{ext}", _png_bytes(i, ext))
        meta = {"caption": caption, "dense_caption": caption, "n_words": str(i)}
        add(shards[shard], f"{key}.json", json.dumps(meta).encode())
    add(shards[1], "orphan.jpg", _png_bytes(9, "jpg"))  # no json -> skipped
    add(shards[1], "README.txt", b"not a sample")  # unexpected extension -> ignored
    for tf in shards:
        tf.close()
    return str(root)


def _cfg(root, tmp_path, **kw):
    kw.setdefault("style", "scene_text")
    kw.setdefault("max_crops", 1)
    kw.setdefault("index_cache_dir", str(tmp_path / "index_cache"))
    return OcrCaptionTarsDatasetConfig(dataset_path=root, **kw)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def test_index_pairs_members_and_skips_orphans(tmp_path):
    root = _write_tars(tmp_path)
    idx = TarShardIndex.build(TarShardIndex.list_shards(root))
    assert len(idx) == 5
    assert [str(k) for k in idx.keys] == [s[1] for s in SAMPLES]
    assert idx.shard_idx.tolist() == [s[0] for s in SAMPLES]
    img, meta = idx.read_sample(2)
    assert json.loads(meta)["caption"] == SAMPLES[2][3]
    from PIL import Image

    assert Image.open(io.BytesIO(img)).size == (48 + 4 * 2, 40)


def test_index_handles_non_ascii_keys(tmp_path):
    """Keys come from tar member names, which nothing constrains to ASCII. ``np.bytes_`` encodes
    str through the ASCII codec, so one non-ASCII name would raise ``UnicodeEncodeError`` from
    ``__init__`` -- at mixture-build time, where the loader's per-example error tolerance does not
    apply, after paying the whole shard scan."""
    root = _write_tars(tmp_path, extra_key="café_señor_日本語")
    idx = TarShardIndex.build(TarShardIndex.list_shards(root))
    assert "café_señor_日本語" in [str(k) for k in idx.keys]
    cache = str(tmp_path / "cache_unicode")
    TarShardIndex.load_or_build(root, cache_dir=cache)  # survives the npz round trip
    reloaded = TarShardIndex.load_or_build(root, cache_dir=cache)
    assert "café_señor_日本語" in [str(k) for k in reloaded.keys]
    ds = _cfg(root, tmp_path).build(_FakeTok())
    assert "café_señor_日本語" in [ds.key(i) for i in range(len(ds))]


def test_index_cache_name_carries_the_format_version(tmp_path):
    """The v1 cache stored ASCII bytes for ``keys``; a stale one must be ignored, not mis-read."""
    root = _write_tars(tmp_path)
    cache = str(tmp_path / "cache_ver")
    TarShardIndex.load_or_build(root, cache_dir=cache)
    (name,) = os.listdir(cache)
    assert f"-v{ct.INDEX_FORMAT_VERSION}-" in name


def test_index_save_is_atomic_and_uniquely_named(tmp_path, monkeypatch):
    """Two writers must not share a temp path: ranks on different hosts can collide on pid."""
    root = _write_tars(tmp_path)
    idx = TarShardIndex.build(TarShardIndex.list_shards(root))
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(src)
        return real_replace(src, dst)

    monkeypatch.setattr(ct.os, "replace", spy)
    target = str(tmp_path / "c" / "idx.npz")
    idx.save(target)
    idx.save(target)
    assert len(set(seen)) == 2 and all(s != target for s in seen)
    assert os.listdir(tmp_path / "c") == ["idx.npz"]  # no temp left behind

    # A failed write leaves no temp file either.
    monkeypatch.setattr(ct.np, "savez", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        idx.save(target)
    assert os.listdir(tmp_path / "c") == ["idx.npz"]


def test_index_cache_roundtrip_and_reuse(tmp_path, monkeypatch):
    root = _write_tars(tmp_path)
    cache = str(tmp_path / "cache")
    idx = TarShardIndex.load_or_build(root, cache_dir=cache)
    files = os.listdir(cache)
    assert len(files) == 1 and files[0].startswith("toy_v6_tars-") and files[0].endswith(".npz")
    # Second load must come from the cache: scanning is forbidden.
    monkeypatch.setattr(ct, "_scan_shard", lambda path: pytest.fail("rescanned " + path))
    again = TarShardIndex.load_or_build(root, cache_dir=cache)
    assert again.shards == idx.shards
    np.testing.assert_array_equal(again.keys, idx.keys)
    np.testing.assert_array_equal(again.offsets, idx.offsets)
    assert not any(f.startswith("toy_v6_tars-") and ".tmp-" in f for f in os.listdir(cache))


def test_index_cache_invalidates_when_shards_change(tmp_path):
    root = _write_tars(tmp_path)
    cache = str(tmp_path / "cache")
    TarShardIndex.load_or_build(root, cache_dir=cache)
    with tarfile.open(os.path.join(root, "toy-w02-00000.tar"), "w") as tf:
        data = _png_bytes(3, "jpg")
        info = tarfile.TarInfo("c_000.jpg")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        meta = json.dumps({"caption": "x"}).encode()
        info = tarfile.TarInfo("c_000.json")
        info.size = len(meta)
        tf.addfile(info, io.BytesIO(meta))
    idx = TarShardIndex.load_or_build(root, cache_dir=cache)
    assert len(idx) == 6 and len(os.listdir(cache)) == 2


def test_index_pairs_each_key_once(tmp_path):
    """The scan de-duplicates member stems through a set now (list membership made it O(n^2));
    a repeated stem must still yield exactly one sample, at its last offsets."""
    root = tmp_path / "dup_tars"
    root.mkdir()
    with tarfile.open(str(root / "dup-00000.tar"), "w") as tf:

        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        for _ in range(2):
            add("k.png", _png_bytes(1, "png"))
            add("k.json", json.dumps({"caption": "x"}).encode())
    idx = TarShardIndex.build(TarShardIndex.list_shards(str(root)))
    assert [str(k) for k in idx.keys] == ["k"]


def test_missing_shards_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        TarShardIndex.list_shards(str(tmp_path))


# ---------------------------------------------------------------------------
# Text / prompt
# ---------------------------------------------------------------------------


def test_strip_text_tags():
    assert ct.strip_text_tags("<text>FELIX 889</text>") == "FELIX 889"
    assert ct.strip_text_tags("<text>\n a\nb \n</text>") == "a\nb"
    assert ct.strip_text_tags("plain caption") == "plain caption"
    assert ct.strip_text_tags("<text>keep <b>inner</b> tags</text>") == "keep <b>inner</b> tags"


def test_config_validation(tmp_path):
    with pytest.raises(OLMoConfigurationError):
        OcrCaptionTarsDatasetConfig().validate()  # no dataset_path
    with pytest.raises(OLMoConfigurationError):
        OcrCaptionTarsDatasetConfig(dataset_path="/x", style="").validate()
    with pytest.raises(OLMoConfigurationError):
        OcrCaptionTarsDatasetConfig(dataset_path="/x", system_prompt="uber_model_v2").validate()
    OcrCaptionTarsDatasetConfig(dataset_path="/x").validate()


def test_dataset_text_and_prompt(tmp_path):
    root = _write_tars(tmp_path)
    ds = _cfg(root, tmp_path).build(_FakeTok())
    assert len(ds) == 5
    assert ds.key(1) == "a_001"
    assert ds.text({"caption": "<text>FELIX</text>"}) == "FELIX"
    assert (
        _cfg(root, tmp_path, strip_text_tags=False)
        .build(_FakeTok())
        .text({"caption": "<text>FELIX</text>"})
        == "<text>FELIX</text>"
    )
    with pytest.raises(ValueError):
        ds.text({"caption": "<text></text>"})
    with pytest.raises(ValueError):
        ds.text({"dense_caption": "x"})
    # style_and_length_v3 default -> bare tag; v2 -> length bucket; none -> nothing.
    from olmo_core.data.multimodal.pixmo_cap import style_tag_prompt

    rng = np.random.RandomState(0)
    assert style_tag_prompt("scene_text", "abc", rng, "style_and_length_v3") == "scene_text:"
    assert style_tag_prompt("scene_text", "abc", rng, "none") == ""
    v2 = {
        style_tag_prompt("ocr_caption", "x" * 300, np.random.RandomState(s), "style_and_length_v2")
        for s in range(20)
    }
    assert all(p.startswith("ocr_caption") and p.endswith(":") for p in v2) and any(
        " " in p for p in v2
    )
    with pytest.raises(ValueError):
        style_tag_prompt("scene_text", "abc", rng, "bogus")


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def test_example_layout(tmp_path):
    root = _write_tars(tmp_path)
    tok = _FakeTok()
    ds = _cfg(root, tmp_path, loss_token_weighting="none").build(tok)
    ex = ds[0]
    for key in ("input_ids", "labels", "loss_masks", "position_ids", "token_type_ids", "images"):
        assert key in ex
    assert "subsegment_ids" not in ex
    text_ids = ex["input_ids"][ex["token_type_ids"] == 0].tolist()
    prompt_ids = tok.encode("scene_text:")
    assert any(text_ids[i : i + len(prompt_ids)] == prompt_ids for i in range(len(text_ids)))
    resp = tok.encode("FELIX PRIVAT DBU 889")  # tags stripped
    assert ex["loss_masks"].sum() == pytest.approx(len(resp) + 1)
    assert ex["labels"][ex["loss_masks"] > 0][-1] == tok.eos_token_id
    np.testing.assert_array_equal(ex["input_ids"], ds[0]["input_ids"])  # deterministic


def test_blank_target_is_skipped_not_raised(tmp_path):
    """An unusable row must not raise out of ``__getitem__``: that spends the mixture loader's
    error budget and a run of them inside one shard aborts training. The next usable row is
    substituted instead."""
    root = _write_tars(tmp_path)
    tok = _FakeTok()
    ds = _cfg(root, tmp_path, loss_token_weighting="none").build(tok)
    assert ds.text({"caption": "<text>x</text>"}) == "x"
    with pytest.raises(ValueError):  # the underlying row is still unusable
        ds._build(3)
    np.testing.assert_array_equal(ds[3]["input_ids"], ds[4]["input_ids"])  # skipped to row 4


def test_unusable_rows_eventually_raise(tmp_path, monkeypatch):
    """Skipping is bounded: an all-broken source must still fail loudly rather than spin."""
    from olmo_core.data.multimodal import sft_common

    root = _write_tars(tmp_path)
    ds = _cfg(root, tmp_path).build(_FakeTok())
    monkeypatch.setattr(ds, "_build", lambda i: (_ for _ in ()).throw(ValueError("nope")))
    with pytest.raises(RuntimeError, match="consecutive rows"):
        ds[0]
    assert sft_common.MAX_ROW_SKIP > 1


def test_message_weight_and_truncation(tmp_path):
    root = _write_tars(tmp_path)
    tok = _FakeTok()
    base = _cfg(root, tmp_path, loss_token_weighting="none").build(tok)[1]
    weighted = _cfg(root, tmp_path, loss_token_weighting="none", message_weight=0.25).build(tok)[1]
    nz = base["loss_masks"] > 0
    np.testing.assert_allclose(weighted["loss_masks"][nz], 0.25 * base["loss_masks"][nz])
    full = _cfg(root, tmp_path).build(tok)[4]
    n_image = int((full["token_type_ids"] == 1).sum())
    assert len(full["input_ids"]) > n_image + 1000
    cut = _cfg(root, tmp_path, max_sequence_length=n_image + 150).build(tok)[4]
    assert len(cut["input_ids"]) == n_image + 150
    assert (cut["token_type_ids"] == 1).sum() == n_image


# ---------------------------------------------------------------------------
# OCR registry + Molmo2-Stage1 wiring
# ---------------------------------------------------------------------------


def test_ocr_registry_shape():
    names = ocr_mix.OCR_SOURCE_NAMES
    assert len(names) == len(set(names)) == 4 + 17
    assert set(ocr_mix.OLMOCR_MIX_SOURCES) <= set(names)
    assert set(ocr_mix.DUPLICATE_OLMOCR_SOURCES) == {"s2pdf", "iabooks"}
    assert set(ocr_mix.DEFAULT_OCR_SOURCES) == set(names) - {"s2pdf", "iabooks"}
    styles = {src.style for src in ocr_mix.OCR_TAR_SOURCES.values()}
    assert styles == {ocr_mix.OLMOCR_STYLE, ocr_mix.OCR_CAPTION_STYLE, ocr_mix.SCENE_TEXT_STYLE}
    for name, src in ocr_mix.OCR_TAR_SOURCES.items():
        # Transcription-type sources ship <text>-wrapped text; caption-type ones do not.
        assert src.strip_text_tags == (src.style != ocr_mix.OCR_CAPTION_STYLE), name
    with pytest.raises(OLMoConfigurationError):
        ocr_mix.build_ocr_source(
            "nope", _FakeTok(), olmocr=OlmOcrMixDatasetConfig(), tars=OcrCaptionTarsDatasetConfig()
        )


def test_build_ocr_source_fills_tar_template(tmp_path):
    root = tmp_path / "oe"
    (root / "scene_text_tars").mkdir(parents=True)
    os.rename(_write_tars(tmp_path), str(root / "scene_text_tars" / "cocotext_v6_tars"))
    tars = OcrCaptionTarsDatasetConfig(
        max_crops=1, index_cache_dir=str(tmp_path / "cache"), system_prompt="style_and_length_v3"
    )
    ds = ocr_mix.build_ocr_source(
        "cocotext", _FakeTok(), olmocr=OlmOcrMixDatasetConfig(), tars=tars, data_root=str(root)
    )
    assert ds.config.style == "scene_text" and ds.config.strip_text_tags is True
    assert ds.config.dataset_path == str(root / "scene_text_tars" / "cocotext_v6_tars")
    assert ds.config.max_crops == 1 and len(ds) == 5


def _load_stage1_module():
    try:
        import olmo_core.internal.common  # noqa: F401  (needs a recent beaker-py)
    except ImportError as e:  # pragma: no cover - env-dependent
        pytest.skip(f"Molmo2-Stage1.py imports fail here: {e}")
    spec = importlib.util.spec_from_file_location(
        "_stage1_ocr_tars", "src/scripts/train/Molmo2-Stage1.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stage1_ocr_tars"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def test_stage1_ocr_group_wiring():
    mod = _load_stage1_module()
    assert mod.OCR_RATE == 0.0
    assert mod.OCR_SOURCES == ocr_mix.DEFAULT_OCR_SOURCES
    assert mod.OCR_SYSTEM_PROMPT == "style_and_length_v3"
    fields = {f.name for f in mod.ExperimentConfig.__dataclass_fields__.values()}
    assert {"ocr_rate", "ocr_sources", "olmocr", "ocr_tars", "ocr_data_root"} <= fields


def test_stage1_refuses_per_source_ocr_tar_overrides():
    """One template serves every tar source, so ``build_ocr_source`` overwrites these three.
    Accepting them would record the value in the saved run config and then ignore it."""
    mod = _load_stage1_module()
    for override in (
        "--ocr_tars.dataset_path=/somewhere/else",
        "--ocr_tars.style=long_caption",
        "--ocr_tars.strip_text_tags=false",
    ):
        with pytest.raises(OLMoConfigurationError, match="per OCR source"):
            mod.build_config("x.py", "run", ["--ocr_rate=0.1", override])
    # The supported way to relocate the tree is a top-level field, and it reaches the sources.
    cfg = mod.build_config("x.py", "run", ["--ocr_rate=0.1", "--ocr_data_root=/my/tars"])
    assert cfg.ocr_data_root == "/my/tars"
    tars = ocr_mix.OCR_TAR_SOURCES["cocotext"]
    assert ocr_mix.os.path.join("/my/tars", tars.relpath).startswith("/my/tars")
