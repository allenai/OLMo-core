"""The OCR source group for Molmo2 stage-1 (``Molmo2-Stage1.py --ocr_rate``).

Twenty image -> free-text sources of three kinds, each a separate dataset sharing the group's
rate (split by sqrt(size), mm_olmo's default ``root_size_factor``):

* **page transcription** (style ``olmocr``): the four olmOCR-mix-1025 subsets
  (:class:`~olmo_core.data.multimodal.olmocr.OlmOcrMixDatasetConfig`, rendered from PDFs), plus
  the oe-encoder ``olmocr_v6_tars`` ``s2pdf`` / ``iabooks`` (pre-rendered JPEGs, re-transcribed).
* **text-rich captions** (style ``ocr_caption``): synthetic charts / diagrams / documents /
  graphics / tables, the Cambrian OCR-heavy subsets (arxivqa, ocr_vqa, screen_qa, llavar, oodvqa)
  and TextCaps -- one dense natural-language caption per image.
* **scene text** (style ``scene_text``): TextOCR, HierText, COCO-Text and UberText, whose target
  is the text visible in the photo.

``s2pdf`` and ``iabooks`` are the SAME pages as olmOCR-mix ``documents`` / ``books`` train
(97.4% / 99.4% of their page ids, and every one of their documents; none of the eval pages), only
rendered and transcribed by a different pipeline. They are registered so either rendering can be
chosen, but :data:`DEFAULT_OCR_SOURCES` leaves them out so a page is not counted twice.

TextCaps' ``caption`` is its five reference captions concatenated into one string (``n_refs``),
which is what the tars ship; it stays in the default group as a caption source but is the one to
drop first if that target form is unwanted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

from olmo_core.data.multimodal.ocr_caption_tars import OcrCaptionTarsDatasetConfig
from olmo_core.data.multimodal.olmocr import OlmOcrMixDatasetConfig
from olmo_core.data.multimodal.paths import OE_ENCODER_DATA
from olmo_core.exceptions import OLMoConfigurationError

__all__ = [
    "OcrTarSource",
    "OCR_TAR_SOURCES",
    "OLMOCR_MIX_SOURCES",
    "OCR_SOURCE_NAMES",
    "DUPLICATE_OLMOCR_SOURCES",
    "DEFAULT_OCR_SOURCES",
    "OLMOCR_STYLE",
    "OCR_CAPTION_STYLE",
    "SCENE_TEXT_STYLE",
    "build_ocr_source",
]

OLMOCR_STYLE = "olmocr"
OCR_CAPTION_STYLE = "ocr_caption"
SCENE_TEXT_STYLE = "scene_text"


@dataclass(frozen=True)
class OcrTarSource:
    """One oe-encoder caption-tars source."""

    relpath: str
    """Shard directory under :data:`~olmo_core.data.multimodal.paths.OE_ENCODER_DATA`."""
    style: str
    strip_text_tags: bool
    """Whether the tars wrap the text in ``<text>...</text>`` (transcription-type sources)."""


OCR_TAR_SOURCES: Dict[str, OcrTarSource] = {
    # Synthetic text-rich images, one dense caption each (~800 chars).
    "text_rich_chart": OcrTarSource("text_rich_caption_v6_tars/chart", OCR_CAPTION_STYLE, False),
    "text_rich_diagram": OcrTarSource(
        "text_rich_caption_v6_tars/diagram", OCR_CAPTION_STYLE, False
    ),
    "text_rich_doc": OcrTarSource("text_rich_caption_v6_tars/doc", OCR_CAPTION_STYLE, False),
    "text_rich_graphic": OcrTarSource(
        "text_rich_caption_v6_tars/graphic", OCR_CAPTION_STYLE, False
    ),
    "text_rich_table": OcrTarSource("text_rich_caption_v6_tars/table", OCR_CAPTION_STYLE, False),
    # Cambrian's OCR-heavy subsets, re-captioned.
    "cambrian_arxivqa": OcrTarSource("cambrian_v6_tars/cambrian_arxivqa", OCR_CAPTION_STYLE, False),
    "cambrian_ocr_vqa": OcrTarSource("cambrian_v6_tars/cambrian_ocr_vqa", OCR_CAPTION_STYLE, False),
    "cambrian_screen_qa": OcrTarSource(
        "cambrian_v6_tars/cambrian_screen_qa", OCR_CAPTION_STYLE, False
    ),
    "cambrian_llavar": OcrTarSource("cambrian_v6_tars/cambrian_llavar", OCR_CAPTION_STYLE, False),
    "cambrian_oodvqa": OcrTarSource("cambrian_v6_tars/cambrian_oodvqa", OCR_CAPTION_STYLE, False),
    # olmOCR pages, pre-rendered; duplicates of olmOCR-mix documents / books (see module doc).
    "s2pdf": OcrTarSource("olmocr_v6_tars/s2pdf", OLMOCR_STYLE, True),
    "iabooks": OcrTarSource("olmocr_v6_tars/iabooks", OLMOCR_STYLE, True),
    # Scene text.
    "textocr": OcrTarSource("textocr_v6_tars", SCENE_TEXT_STYLE, True),
    "textcaps": OcrTarSource("textcaps_v6_tars", OCR_CAPTION_STYLE, False),
    "hiertext": OcrTarSource("scene_text_tars/hiertext_v6_tars", SCENE_TEXT_STYLE, True),
    "cocotext": OcrTarSource("scene_text_tars/cocotext_v6_tars", SCENE_TEXT_STYLE, True),
    "ubertext": OcrTarSource("scene_text_tars/ubertext_v6_tars", SCENE_TEXT_STYLE, True),
}

#: olmOCR-mix sources: group name -> ``OlmOcrMixDatasetConfig.subset``.
OLMOCR_MIX_SOURCES: Dict[str, str] = {
    "olmocr_documents": "documents",
    "olmocr_books": "books",
    "olmocr_loc_transcripts": "loc_transcripts",
    "olmocr_national_archives": "national_archives",
}

OCR_SOURCE_NAMES: Tuple[str, ...] = tuple(OLMOCR_MIX_SOURCES) + tuple(OCR_TAR_SOURCES)

#: Tar sources whose pages are already in an olmOCR-mix train subset (see module doc).
DUPLICATE_OLMOCR_SOURCES: Dict[str, str] = {"s2pdf": "olmocr_documents", "iabooks": "olmocr_books"}

DEFAULT_OCR_SOURCES: Tuple[str, ...] = tuple(
    n for n in OCR_SOURCE_NAMES if n not in DUPLICATE_OLMOCR_SOURCES
)


def build_ocr_source(
    name: str,
    tokenizer,
    *,
    olmocr: OlmOcrMixDatasetConfig,
    tars: OcrCaptionTarsDatasetConfig,
    data_root: str = OE_ENCODER_DATA,
):
    """Build one OCR source by name from the two template configs.

    :param olmocr: template for the olmOCR-mix sources; its ``subset`` is overridden.
    :param tars: template for the caption-tars sources; ``dataset_path``, ``style`` and
        ``strip_text_tags`` are overridden from :data:`OCR_TAR_SOURCES`.
    :param data_root: where the oe-encoder tar directories live.
    """
    if name in OLMOCR_MIX_SOURCES:
        return olmocr.replace(subset=OLMOCR_MIX_SOURCES[name]).build(tokenizer)
    if name in OCR_TAR_SOURCES:
        src = OCR_TAR_SOURCES[name]
        return tars.replace(
            dataset_path=os.path.join(data_root, src.relpath),
            style=src.style,
            strip_text_tags=src.strip_text_tags,
        ).build(tokenizer)
    raise OLMoConfigurationError(f"Unknown OCR source {name!r}; expected one of {OCR_SOURCE_NAMES}")
