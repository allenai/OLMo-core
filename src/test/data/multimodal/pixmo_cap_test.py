"""
Tests for the PixMo-Cap adapter.

Two loading modes are covered:

- ``local``: reads a pre-downloaded on-disk HF ``Dataset``. Auto-skips if the
  local dataset path isn't present (e.g. running outside the AI2 cluster).
- ``hub``: streams ``allenai/pixmo-cap`` and downloads images per-URL. Tested
  with ``datasets.load_dataset`` and ``requests.get`` monkeypatched, so it runs
  deterministically without network access.
"""

import io
import os

import numpy as np
import pytest

from olmo_core.data.multimodal.pixmo_cap import PixmoCapDataset, PixmoCapDatasetConfig

# ---------------------------------------------------------------------------
# local mode (skipped without the on-disk dataset)
# ---------------------------------------------------------------------------


def _local_path_present() -> bool:
    return os.path.isdir(PixmoCapDatasetConfig(source="local").resolve_data_dir())


local_only = pytest.mark.skipif(
    not _local_path_present(),
    reason="local PixMo-Cap dataset directory not present",
)


@local_only
def test_local_yields_one_example():
    pytest.importorskip("datasets")
    pytest.importorskip("PIL")
    ds = PixmoCapDatasetConfig(source="local", limit=1).build()
    prompt, caption, image = next(iter(ds))
    assert isinstance(prompt, str) and len(prompt) > 0
    assert isinstance(caption, str) and len(caption) > 0
    assert hasattr(image, "size") and len(image.size) == 2


@local_only
def test_local_iteration_stops_at_limit():
    pytest.importorskip("datasets")
    pytest.importorskip("PIL")
    items = list(PixmoCapDatasetConfig(source="local", limit=3).build())
    assert len(items) == 3


# ---------------------------------------------------------------------------
# hub mode (monkeypatched — no network)
# ---------------------------------------------------------------------------


def _png_bytes(color=(123, 222, 64)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes, ok: bool = True):
        self.content = content
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("HTTP error")


def _patch_hub(monkeypatch, rows, *, dead_urls=()):
    """Patch datasets.load_dataset → rows, requests.get → fake PNG (or error
    for URLs in ``dead_urls``)."""
    pytest.importorskip("datasets")
    pytest.importorskip("requests")
    import datasets
    import requests

    def fake_load_dataset(dataset_id, split=None, streaming=False):
        assert streaming is True
        return list(rows)

    def fake_get(url, timeout=None):
        if url in dead_urls:
            return _FakeResponse(b"", ok=False)
        return _FakeResponse(_png_bytes())

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(requests, "get", fake_get)


def test_hub_yields_triples(monkeypatch):
    rows = [
        {"image_url": "http://x/0.jpg", "caption": "a cat", "transcripts": []},
        {"image_url": "http://x/1.jpg", "caption": "a dog", "transcripts": []},
    ]
    _patch_hub(monkeypatch, rows)
    items = list(PixmoCapDataset(PixmoCapDatasetConfig(source="hub")))
    assert len(items) == 2
    prompt, caption, image = items[0]
    assert prompt == PixmoCapDatasetConfig().prompt
    assert caption == "a cat"
    assert image.size == (8, 8)


def test_hub_skips_dead_urls(monkeypatch):
    rows = [
        {"image_url": "http://x/ok.jpg", "caption": "good", "transcripts": []},
        {"image_url": "http://x/dead.jpg", "caption": "bad", "transcripts": []},
        {"image_url": "http://x/ok2.jpg", "caption": "good2", "transcripts": []},
    ]
    _patch_hub(monkeypatch, rows, dead_urls={"http://x/dead.jpg"})
    captions = [c for _, c, _ in PixmoCapDataset(PixmoCapDatasetConfig(source="hub"))]
    assert captions == ["good", "good2"]


def test_hub_limit_counts_yielded_not_scanned(monkeypatch):
    """limit should count successfully-loaded examples, so dead URLs in front
    of the limit don't shrink the yielded count."""
    rows = [
        {"image_url": "http://x/dead.jpg", "caption": "skip", "transcripts": []},
        {"image_url": "http://x/a.jpg", "caption": "a", "transcripts": []},
        {"image_url": "http://x/b.jpg", "caption": "b", "transcripts": []},
        {"image_url": "http://x/c.jpg", "caption": "c", "transcripts": []},
    ]
    _patch_hub(monkeypatch, rows, dead_urls={"http://x/dead.jpg"})
    captions = [c for _, c, _ in PixmoCapDataset(PixmoCapDatasetConfig(source="hub", limit=2))]
    assert captions == ["a", "b"]


def test_invalid_source_raises():
    with pytest.raises(ValueError, match="source must be"):
        PixmoCapDataset(PixmoCapDatasetConfig(source="nope"))
