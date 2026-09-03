"""Webdataset-style caption tars for Molmo2 stage-1 (the oe-encoder ``*_v6_tars`` OCR sources).

Layout: a directory of ``.tar`` shards in which every sample is an image member (``<key>.jpg`` /
``<key>.png``) next to a ``<key>.json`` member. The JSON always carries ``caption`` (and a
``dense_caption`` duplicate of it); OCR-type sources wrap the transcription in ``<text>...</text>``,
and per-source extras (``url`` / ``page`` / ``n_words`` / ``image_classes``) are ignored. The
sources and their styles live in :mod:`olmo_core.data.multimodal.mixtures.ocr`.

A map-style dataset needs random access, but tar has no index: :class:`TarShardIndex` scans every
shard's headers once (seeking over member data, roughly a second per GB) and caches the member
offsets as an ``.npz`` keyed by the shards' names, sizes and mtimes, so later ranks and runs load
it instantly. Examples are built like olmOCR-mix pages: the user turn is the style tag only, the
assistant turn is the text, tail-truncated to ``max_sequence_length``.
"""

from __future__ import annotations

import glob
import hashlib
import io
import json
import logging
import os
import re
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .message_sequence import encode_sft_example
from .pixmo_cap import STYLE_TAG_FAMILIES, style_tag_prompt
from .sequence_builder import example_rng
from .sft_common import truncate_example

__all__ = [
    "TarShardIndex",
    "OcrCaptionTarsDatasetConfig",
    "OcrCaptionTarsDataset",
    "strip_text_tags",
    "default_index_cache_dir",
]

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
_TEXT_TAG_RE = re.compile(r"^\s*<text>\s*(.*?)\s*</text>\s*$", re.S)


def strip_text_tags(text: str) -> str:
    """Drop the ``<text>...</text>`` wrapper the OCR-type sources put around a transcription
    (only the outer pair; anything inside is kept verbatim)."""
    m = _TEXT_TAG_RE.match(text)
    return m.group(1) if m else text


def default_index_cache_dir() -> str:
    """Where shard indices are cached: ``$OLMO_CORE_TAR_INDEX_DIR``, else next to the HF datasets
    cache (``$HF_DATASETS_CACHE/olmo_core_tar_index``), else ``~/.cache/olmo_core/tar_index``."""
    explicit = os.environ.get("OLMO_CORE_TAR_INDEX_DIR")
    if explicit:
        return explicit
    hf = os.environ.get("HF_DATASETS_CACHE")
    if hf:
        return os.path.join(hf, "olmo_core_tar_index")
    return os.path.join(os.path.expanduser("~"), ".cache", "olmo_core", "tar_index")


def _scan_shard(path: str) -> Tuple[List[str], np.ndarray]:
    """One shard's complete ``(image, json)`` pairs, in tar order.

    :returns: ``(keys, offsets)`` with ``offsets`` an ``(n, 4)`` int64 array of
        ``image_offset, image_size, json_offset, json_size``. Members without a partner (or with
        an unexpected extension) are skipped.
    """
    images: Dict[str, Tuple[int, int]] = {}
    jsons: Dict[str, Tuple[int, int]] = {}
    order: List[str] = []
    with tarfile.open(path) as tf:  # random-access mode: `next()` seeks past member data
        for m in tf:
            if not m.isfile():
                continue
            stem, ext = os.path.splitext(m.name)
            ext = ext.lower()
            if ext == ".json":
                jsons[stem] = (m.offset_data, m.size)
            elif ext in IMAGE_EXTENSIONS:
                images[stem] = (m.offset_data, m.size)
            else:
                continue
            if stem not in order:
                order.append(stem)
    keys = [k for k in order if k in images and k in jsons]
    offsets = np.asarray([[*images[k], *jsons[k]] for k in keys], dtype=np.int64).reshape(
        len(keys), 4
    )
    return keys, offsets


@dataclass
class TarShardIndex:
    """Member offsets of every ``(image, json)`` sample across a directory of tar shards."""

    shards: List[str]
    """Absolute shard paths, sorted; ``shard_idx`` indexes into this list."""
    keys: np.ndarray
    """``(N,)`` sample keys (``bytes_`` dtype), for provenance."""
    shard_idx: np.ndarray
    """``(N,)`` int32 shard of each sample."""
    offsets: np.ndarray
    """``(N, 4)`` int64: image offset, image size, json offset, json size."""

    def __len__(self) -> int:
        return len(self.keys)

    @staticmethod
    def list_shards(dataset_path: str, shard_glob: str = "*.tar") -> List[str]:
        shards = sorted(glob.glob(os.path.join(dataset_path, shard_glob)))
        if not shards:
            raise FileNotFoundError(f"No shards matching {shard_glob!r} under {dataset_path}")
        return shards

    @staticmethod
    def fingerprint(shards: Sequence[str]) -> str:
        """Hash of the shards' basenames, sizes and mtimes: any change invalidates the cache."""
        h = hashlib.sha1()
        for s in shards:
            st = os.stat(s)
            h.update(f"{os.path.basename(s)}\t{st.st_size}\t{int(st.st_mtime)}\n".encode())
        return h.hexdigest()[:16]

    @classmethod
    def build(cls, shards: Sequence[str], *, num_threads: int = 8) -> "TarShardIndex":
        """Scan the shards' headers (in parallel: the cost is weka seek latency)."""
        shards = list(shards)
        with ThreadPoolExecutor(max(1, num_threads)) as ex:
            scanned = list(ex.map(_scan_shard, shards))
        keys: List[str] = []
        shard_idx: List[np.ndarray] = []
        offsets: List[np.ndarray] = []
        for i, (ks, offs) in enumerate(scanned):
            keys.extend(ks)
            shard_idx.append(np.full(len(ks), i, dtype=np.int32))
            offsets.append(offs)
        return cls(
            shards=shards,
            keys=np.asarray(keys, dtype=np.bytes_),
            shard_idx=np.concatenate(shard_idx) if shard_idx else np.zeros(0, dtype=np.int32),
            offsets=np.concatenate(offsets) if offsets else np.zeros((0, 4), dtype=np.int64),
        )

    def save(self, path: str) -> None:
        """Atomic write (temp file + ``os.replace``), so concurrent ranks never read a partial
        index."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp-{os.getpid()}"
        with open(tmp, "wb") as f:
            np.savez(
                f,
                shards=np.asarray(self.shards, dtype=np.str_),
                keys=self.keys,
                shard_idx=self.shard_idx,
                offsets=self.offsets,
            )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "TarShardIndex":
        with np.load(path) as z:
            return cls(
                shards=[str(s) for s in z["shards"]],
                keys=z["keys"],
                shard_idx=z["shard_idx"],
                offsets=z["offsets"],
            )

    @classmethod
    def load_or_build(
        cls,
        dataset_path: str,
        *,
        shard_glob: str = "*.tar",
        cache_dir: Optional[str] = None,
        num_threads: int = 8,
    ) -> "TarShardIndex":
        """The cached index for ``dataset_path``'s shards, scanning and caching it if missing."""
        shards = cls.list_shards(dataset_path, shard_glob)
        cache_dir = cache_dir or default_index_cache_dir()
        name = os.path.basename(os.path.normpath(dataset_path)) or "tars"
        cache_path = os.path.join(cache_dir, f"{name}-{cls.fingerprint(shards)}.npz")
        if os.path.exists(cache_path):
            index = cls.load(cache_path)
            if index.shards == shards:
                return index
        log.info("Indexing %d tar shard(s) under %s ...", len(shards), dataset_path)
        index = cls.build(shards, num_threads=num_threads)
        index.save(cache_path)
        log.info("Indexed %d samples -> %s", len(index), cache_path)
        return index

    def read_sample(self, i: int) -> Tuple[bytes, bytes]:
        """``(image_bytes, json_bytes)`` of sample ``i`` (one open + two seek-reads)."""
        img_off, img_size, js_off, js_size = (int(x) for x in self.offsets[i])
        with open(self.shards[int(self.shard_idx[i])], "rb") as f:
            f.seek(img_off)
            image = f.read(img_size)
            f.seek(js_off)
            meta = f.read(js_size)
        return image, meta


@dataclass
class OcrCaptionTarsDatasetConfig(Config):
    """One caption-tars source: image -> free text behind a style tag."""

    dataset_path: str = ""
    """Directory of ``.tar`` shards (see :mod:`.mixtures.ocr` for the known sources)."""

    style: str = "ocr_caption"
    """mm_olmo-style name shown in the user turn's tag; see :mod:`.mixtures.ocr` for the ones the
    OCR sources use (``ocr_caption`` / ``olmocr`` / ``scene_text``)."""

    text_field: str = "caption"
    """JSON field holding the target text."""

    strip_text_tags: bool = True
    """Drop the ``<text>...</text>`` wrapper of OCR-type sources so the target is the bare text,
    like olmOCR-mix's ``natural_text``."""

    shard_glob: str = "*.tar"
    index_cache_dir: Optional[str] = None
    """Where the shard index is cached; ``None`` -> :func:`default_index_cache_dir`."""
    index_threads: int = 8

    max_crops: int = 8
    max_sequence_length: Optional[int] = None
    """Tail-truncate the built sequence to this many tokens (see :func:`~.sft_common.truncate_example`)."""
    loss_token_weighting: str = "root_subsegments"
    message_weight: Optional[float] = None
    seed: int = 0
    system_prompt: str = "style_and_length_v3"
    """Prefix family for the style tag (:func:`~.pixmo_cap.style_tag_prompt`); the default renders the
    bare ``"<style>:"``, as mm_olmo's molmo3 stage 1 does for its OCR sources."""

    def validate(self):
        if not self.dataset_path:
            raise OLMoConfigurationError("dataset_path must point at a directory of tar shards")
        if not self.style:
            raise OLMoConfigurationError("style must be a non-empty style name")
        if not self.text_field:
            raise OLMoConfigurationError("text_field must be a JSON field name")
        if self.system_prompt not in STYLE_TAG_FAMILIES:
            raise OLMoConfigurationError(
                f"system_prompt={self.system_prompt!r} is not one of {sorted(STYLE_TAG_FAMILIES)}"
            )
        if self.index_threads < 1:
            raise OLMoConfigurationError("index_threads must be >= 1")

    def build(self, tokenizer) -> "OcrCaptionTarsDataset":
        self.validate()
        return OcrCaptionTarsDataset(self, tokenizer)


class OcrCaptionTarsDataset:
    """Map-style dataset over the ``(image, json)`` samples of a directory of tar shards."""

    def __init__(self, config: OcrCaptionTarsDatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.index = TarShardIndex.load_or_build(
            config.dataset_path,
            shard_glob=config.shard_glob,
            cache_dir=config.index_cache_dir,
            num_threads=config.index_threads,
        )
        self._lock = threading.Lock()
        log.info(
            "caption tars %s (style=%s): %d samples in %d shards",
            config.dataset_path,
            config.style,
            len(self.index),
            len(self.index.shards),
        )

    def __len__(self) -> int:
        return len(self.index)

    def key(self, i: int) -> str:
        return self.index.keys[i].decode()

    def text(self, meta: Dict) -> str:
        """The target text of a decoded JSON record."""
        cfg = self.config
        raw = meta.get(cfg.text_field)
        if not isinstance(raw, str):
            raise ValueError(f"JSON field {cfg.text_field!r} missing or not a string")
        text = strip_text_tags(raw) if cfg.strip_text_tags else raw
        if not text.strip():
            raise ValueError("empty target text")
        return text

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        from PIL import Image

        cfg = self.config
        image_bytes, json_bytes = self.index.read_sample(i)
        text = self.text(json.loads(json_bytes))
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rng = example_rng(cfg.seed, i)
        prompt = style_tag_prompt(cfg.style, text, rng, cfg.system_prompt)
        seq = encode_sft_example(
            self.tokenizer,
            image,
            [(prompt, text)],
            max_crops=cfg.max_crops,
            loss_token_weighting=cfg.loss_token_weighting,
            message_weight=cfg.message_weight,
            shuffle_rng=rng,
        )
        if cfg.max_sequence_length is not None:
            seq = truncate_example(seq, cfg.max_sequence_length)
        return seq
