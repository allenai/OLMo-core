"""
Shared definitions for the custom 50B "longmino-512k" mix.

This module is imported by both :mod:`download_longmino_512k_raw` and
:mod:`tokenize_longmino_512k` so that the stratum routing and the document filter are defined
exactly once. The two tokenizer runs (Qwen3 and Qwen3.5) read the same raw files and apply the same
rules from here, which is what makes their document sets identical by construction.
"""

import re
from typing import Optional

WEKA_ROOT = "/weka/oe-training-default/amandab/longmino_512k"

MIX_DATASET = "allenai/dolma3_longmino_mix-50B-1025"
POOL_DATASET = "allenai/dolma3_longmino_pool"

#: Target tokens per new pool bucket (15% each of a 17.17B long-context part).
POOL_TARGET_TOKENS = 2_580_000_000
#: Over-collect so the train-time mixing ratios have slack.
POOL_HEADROOM = 1.3

#: Total tokens per pool length bucket, from the pool dataset card. Used with the summed shard
#: sizes to convert a token budget into a byte budget, so shard selection needs no decompression.
POOL_BUCKET_TOKENS = {
    "2e16": 96_000_000_000,  # 64k-128k
    "2e17": 60_800_000_000,  # 128k-256k
    "2e18": 35_100_000_000,  # 256k-512k
}

POOL_BUCKETS = tuple(POOL_BUCKET_TOKENS)

#: Documents redacted after the fact keep their full metadata (``len_cl100k_base``,
#: ``pdf-total-pages``), so a metadata-driven length filter will not catch them.
REMOVED_PLACEHOLDER = "[REMOVED]"

_LC_REAL = re.compile(r"^olmocr_science_pdfs-high_quality-.*-(2e1[345])$")
_POOL_SUBSET = re.compile(r"^olmocr_science_pdfs-.*-(2e1[678])$")

_MIDTRAIN_GROUPS = (
    (re.compile(r"^common_crawl-high-quality_\d+_"), "common_crawl"),
    (re.compile(r"^stack_edu-fim_vigintile_\d+_"), "stack_edu"),
    (re.compile(r"^wiki_to_rcqa-part\d+$"), "wiki_to_rcqa"),
    (re.compile(r"^reddit_to_flashcards-"), "reddit_to_flashcards"),
)


def stratum_for_mix_subset(subset: str) -> str:
    """
    Map a subset directory name from the longmino mix to an output stratum path.

    :param subset: A directory name under ``data/`` in the mix repo, e.g.
        ``olmocr_science_pdfs-high_quality-health-2e15``.

    :returns: A relative stratum path such as ``lc/real_s2pdf/2e15`` or ``midtrain/common_crawl``.
    """
    m = _LC_REAL.match(subset)
    if m is not None:
        return f"lc/real_s2pdf/{m.group(1)}"
    if subset == "lc_synth-yake_s2pdf":
        return "lc/synth_rex/2e15"
    if subset == "lc_synth-cwe_s2pdf":
        return "lc/synth_cwe/2e15"
    for pattern, family in _MIDTRAIN_GROUPS:
        if pattern.match(subset):
            return f"midtrain/{family}"
    return f"midtrain/{subset}"


def pool_bucket_for_subset(subset: str) -> Optional[str]:
    """
    Return the length bucket (``2e16``/``2e17``/``2e18``) for a pool subset directory, or ``None``
    if the subset is not one of the buckets we pull.
    """
    m = _POOL_SUBSET.match(subset)
    if m is None:
        return None
    bucket = m.group(1)
    return bucket if bucket in POOL_BUCKET_TOKENS else None


def stratum_for_pool_bucket(bucket: str) -> str:
    """Map a pool length bucket to its output stratum path."""
    return f"lc/real_s2pdf/{bucket}"


def is_usable_text(text: Optional[str]) -> bool:
    """
    Whether a document's ``text`` field should be included.

    Filters empty/whitespace-only documents and the ``[REMOVED]`` redaction placeholder. Both
    tokenizer runs must apply this identically or their document sets diverge.
    """
    if not text:
        return False
    stripped = text.strip()
    return bool(stripped) and stripped != REMOVED_PLACEHOLDER
