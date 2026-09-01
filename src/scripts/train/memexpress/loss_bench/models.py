"""
Registry of the 8 checkpoints in the train/val loss-benchmark analysis, and the 5 distinct
training-data groups behind them (several checkpoints share a group verbatim -- see the
``data_group`` field on each entry in :data:`MODELS`).

The 8 checkpoints come from 4 results-hub comparison pairs:

- sparse-landmark vs fast-landmark, 32k window (``sparselm_32k`` / ``fastlm_32k``)
- dense vs fast-landmark/compressive, 256k-ish window (``dense_xlong5_256k`` / ``fastlm_33344_datamatch``)
- summary-token training: causal / decay / p50 mask-mixture arms, 256k packed data, IDENTICAL
  training data across all three (``summtok_causal`` / ``summtok_decay`` / ``summtok_p50``)
- doc-chunk vs dense, 32k window (``docchunk_bs128`` / ``dense_fixdata``)

All 9 target weka paths were verified to exist via a gantry job (2026-08-31); the 9th
(``q4b-dense-5task-32k-nocpt-fixdata``) does NOT have a ``step2000`` checkpoint -- only
``step10700`` exists, and that is what ``dense_fixdata`` below points at. That means the
doc-chunk pair is NOT step-matched (docchunk is step2000, dense is step10700) -- flag this in any
downstream comparison.

Data-source paths and mixture recipes below are read directly out of each run's training launcher
(``src/scripts/train/memexpress/sft_5task/_qwen35_5task_dolci25_32k_nocpt_common.py``,
``sft_xlong256k/_qwen35_xlong5_dolci25_256k_common.py``, ``sft_summtoken/_qwen35_summtoken_common.py``,
``sft_docchunk/_docchunk_5task_32k_nocpt_common.py``, and ``sft_5task/Qwen3-4B-dense-5task-32k-nocpt-SFT.py``
for the ``dense_fixdata`` recipe, whose *exact* launcher for this specific run name could not be
confirmed -- the data recipe (Qwen3, ``single_task_ladders_v2`` + p10 NQ, no Dolci) is shared
verbatim by every dense-5task-32k-nocpt Qwen3 launcher in that family, so it is high-confidence even
though the literal script is not).

IMPORTANT caveat surfaced during this mapping (see ``README.md``): the "landmark (compressive) vs
full attn" pair (``dense_xlong5_256k`` vs ``fastlm_33344_datamatch``) does NOT train on the same
data -- the dense arm uses the xlong5 2k->256k ladder, the landmark arm uses the plain 32k/33344
single-task ladder (``pair1_5task_dolci25_32k_qwen35``, the SAME group as ``sparselm_32k`` /
``fastlm_32k``). Train-loss numbers for that pair are each faithful to what that model actually
trained on, but are not on the same distribution as each other -- this was a deliberate, informed
choice (see the clarifying-question answers in the session that produced this script), not a bug.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------------------------
# Data groups: task name -> weka root of that task's `token_ids_part_*.npy` / `labels_mask_*.npy`
# shards. Every group's sources are read with NumpyDocumentSourceConfig(expand_glob=True) exactly
# as the training launchers build them -- see build_train_manifest.py.
# ---------------------------------------------------------------------------------------------

_SINGLE_TASK_V2_QWEN35 = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_v2_qwen35"
)
_NQ_P10_QWEN35 = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_p10_qwen35/nq"
)
_DOLCI_QWEN35 = "/weka/oe-training-default/amandab/dolci-instruct-sft/qwen35"
_XLONG5_SHARDS_FULL = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/xlong5_2k256k_qwen35/shards_full"
)
_SUMMTOK_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/amandab/summtoken_5task_xlong"
_DOCCHUNK_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/docchunk_5task_fixed40k"
_SINGLE_TASK_V2_QWEN3 = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_v2"
)
_NQ_P10_QWEN3 = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders_p10/nq"


@dataclasses.dataclass(frozen=True)
class DataGroup:
    tokenizer_family: str  # "qwen3_5" | "qwen3" -- passed to olmo_core.data.TokenizerConfig.{name}()
    sources: Dict[str, str]  # task label -> weka root dir (glob-expanded: token_ids_part_*.npy)


DATA_GROUPS: Dict[str, DataGroup] = {
    # Shared verbatim by sparselm_32k, fastlm_32k, AND fastlm_33344_datamatch (same launcher file,
    # same DATA_ROOT/NQ_DATA_ROOT/DOLCI_DATA_ROOT -- only ABLATION_SEQ_LEN/CP topology differ).
    "pair1_5task_dolci25_32k_qwen35": DataGroup(
        tokenizer_family="qwen3_5",
        sources={
            "contradiction": f"{_SINGLE_TASK_V2_QWEN35}/contradiction",
            "nq_retrieval": _NQ_P10_QWEN35,
            "oolong": f"{_SINGLE_TASK_V2_QWEN35}/oolong",
            "rerank": f"{_SINGLE_TASK_V2_QWEN35}/rerank",
            "outlier": f"{_SINGLE_TASK_V2_QWEN35}/outlier",
            "dolci_instruct_sft": _DOLCI_QWEN35,
        },
    ),
    # dense_xlong5_256k only. NOT the same tree as pair1 (xlong5 ladder, not single_task_ladders_v2).
    "pair2_dense_xlong5_qboth_256k": DataGroup(
        tokenizer_family="qwen3_5",
        sources={
            "contradiction": f"{_XLONG5_SHARDS_FULL}/contradiction_train",
            "nq_retrieval": f"{_XLONG5_SHARDS_FULL}/nq_train",
            "oolong": f"{_XLONG5_SHARDS_FULL}/oolong_train",
            "rerank": f"{_XLONG5_SHARDS_FULL}/rerank_train",
            "outlier": f"{_XLONG5_SHARDS_FULL}/outlier_train",
            "dolci_instruct_sft": _DOLCI_QWEN35,
        },
    ),
    # Shared verbatim by summtok_causal / summtok_decay / summtok_p50 -- confirmed identical data;
    # the three arms differ ONLY in the mask-mixture kwarg passed to enable_summary_token_attention.
    "pair3_summtoken_packed": DataGroup(
        tokenizer_family="qwen3_5",
        sources={
            "contradiction": f"{_SUMMTOK_ROOT}/contra_summary",
            "nq_retrieval": f"{_SUMMTOK_ROOT}/nq_summary",
            "oolong": f"{_SUMMTOK_ROOT}/oolong_summary",
            "rerank": f"{_SUMMTOK_ROOT}/rerank_summary",
            "outlier": f"{_SUMMTOK_ROOT}/outlier_summary",
        },
    ),
    "pair4_docchunk_5task": DataGroup(
        tokenizer_family="qwen3",
        sources={
            "contradiction": f"{_DOCCHUNK_ROOT}/contra_dense",
            "nq_retrieval": f"{_DOCCHUNK_ROOT}/nq_dense",
            "oolong": f"{_DOCCHUNK_ROOT}/oolong_dense",
            "rerank": f"{_DOCCHUNK_ROOT}/rerank_dense",
            "outlier": f"{_DOCCHUNK_ROOT}/outlier_dense",
        },
    ),
    "pair4_dense_5task_fixdata": DataGroup(
        tokenizer_family="qwen3",
        sources={
            "contradiction": f"{_SINGLE_TASK_V2_QWEN3}/contradiction",
            "nq_retrieval": _NQ_P10_QWEN3,
            "oolong": f"{_SINGLE_TASK_V2_QWEN3}/oolong",
            "rerank": f"{_SINGLE_TASK_V2_QWEN3}/rerank",
            "outlier": f"{_SINGLE_TASK_V2_QWEN3}/outlier",
        },
    ),
}

# Standard doubling ladder used to bucket BOTH train documents (by realized token length) and val
# rungs (by the ladder's own rung label) into comparable "context length" buckets.
LENGTH_LADDER: List[int] = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
LENGTH_LADDER_LABELS: List[str] = ["2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"]


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    architecture: str  # "dense" | "landmark" | "docchunk" | "summary_token"
    tokenizer_family: str  # "qwen3_5" | "qwen3"
    data_group: str  # key into DATA_GROUPS
    max_context_length: int  # cap val/train buckets at this (per the "own trained window" policy)
    query_position: str = (
        "both"  # "both" | "after" -- must match how the run's SFT shards were built
    )
    summary_mask_mode: Optional[str] = None  # summary_token models only: "causal" (project default)


_AB = "/weka/oe-training-default/ai2-llm/checkpoints/amandab"
_PR = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"

MODELS: Dict[str, ModelSpec] = {
    "sparselm_32k": ModelSpec(
        checkpoint=f"{_AB}/q35-4b-sparselm-5task-dolci25-32k-nocpt-cp4/step8550",
        architecture="landmark",
        tokenizer_family="qwen3_5",
        data_group="pair1_5task_dolci25_32k_qwen35",
        max_context_length=32768,
    ),
    "fastlm_32k": ModelSpec(
        checkpoint=f"{_AB}/q35-4b-fastlm-5task-dolci25-32k-nocpt/step8550",
        architecture="landmark",
        tokenizer_family="qwen3_5",
        data_group="pair1_5task_dolci25_32k_qwen35",
        max_context_length=32768,
    ),
    "dense_xlong5_256k": ModelSpec(
        checkpoint=f"{_AB}/q35-4b-dense-xlong5-qboth-dolci25-256k/step2240",
        architecture="dense",
        tokenizer_family="qwen3_5",
        data_group="pair2_dense_xlong5_qboth_256k",
        max_context_length=262144,
        query_position="both",  # explicit in the run name ("qboth")
    ),
    "fastlm_33344_datamatch": ModelSpec(
        checkpoint=f"{_AB}/q35-4b-fastlm-5task-dolci25-33344-datamatch/step10858",
        architecture="landmark",
        tokenizer_family="qwen3_5",
        data_group="pair1_5task_dolci25_32k_qwen35",  # NOT pair2's data -- see module docstring
        max_context_length=32768,  # trained at ABLATION_SEQ_LEN=33344; ladder caps at 32768 <= that
    ),
    "summtok_causal": ModelSpec(
        checkpoint=f"{_AB}/q35-4b-summ-causal-5task-packed/step1772",
        architecture="summary_token",
        tokenizer_family="qwen3_5",
        data_group="pair3_summtoken_packed",
        max_context_length=262144,
        summary_mask_mode="causal",
    ),
    "summtok_decay": ModelSpec(
        checkpoint=f"{_AB}/q35-4b-summ-decay-5task-packed/step1772",
        architecture="summary_token",
        tokenizer_family="qwen3_5",
        data_group="pair3_summtoken_packed",
        max_context_length=262144,
        summary_mask_mode="causal",  # eval-time mask ARM to serve, not the training mixture
    ),
    "summtok_p50": ModelSpec(
        checkpoint=f"{_AB}/q35-4b-summ-p50-5task-packed/step1772",
        architecture="summary_token",
        tokenizer_family="qwen3_5",
        data_group="pair3_summtoken_packed",
        max_context_length=262144,
        summary_mask_mode="causal",
    ),
    "docchunk_bs128": ModelSpec(
        checkpoint=f"{_PR}/q4b-docchunk-5task-32k-nocpt-bs128/step2000",
        architecture="docchunk",
        tokenizer_family="qwen3",
        data_group="pair4_docchunk_5task",
        max_context_length=32768,
    ),
    "dense_fixdata": ModelSpec(
        # step2000 does NOT exist on weka (verified) -- step10700 is the only complete checkpoint.
        checkpoint=f"{_PR}/q4b-dense-5task-32k-nocpt-fixdata/step10700",
        architecture="dense",
        tokenizer_family="qwen3",
        data_group="pair4_dense_5task_fixdata",
        max_context_length=32768,
    ),
}

# ---------------------------------------------------------------------------------------------
# Shared val dataset: the v3 eval bundle (contradiction/nq/outlier/rerank/oolong; contra+outlier are
# genuinely rebuilt for v3, nq/rerank/oolong are directory symlinks to v2_clean -- see
# records/v3-eval-howto.md). Base rungs (2k/3k..32k) always available; xlong rungs (64k-256k) are
# appended where the glob hits files, for models whose max_context_length reaches that far.
# ---------------------------------------------------------------------------------------------

V3_EVAL_BUNDLE_ROOT = (
    "/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v3"
)

# (ladder task key, seg_task alias used by segment_prompt_to_chunks / TASK_CFG, chunk_by, cot_mode)
VAL_TASKS = {
    "contradiction": dict(seg_task="contradiction", chunk_by="document", cot_mode="none"),
    "nq": dict(seg_task="retrieval", chunk_by="document", cot_mode="none"),
    "outlier": dict(seg_task="outlier", chunk_by="document", cot_mode="none"),
    "rerank": dict(seg_task="rerank", chunk_by="document", cot_mode="none"),
    "oolong": dict(seg_task="oolong", chunk_by="line", cot_mode="plan"),
}

WORK_DIR = "/weka/oe-training-default/ai2-llm/checkpoints/amandab/loss_bench_2026-08-31"
TRAIN_MANIFEST_DIR = f"{WORK_DIR}/train_manifests"
VAL_MANIFEST_PATH = f"{WORK_DIR}/val_manifest.json"
RESULTS_DIR = f"{WORK_DIR}/results"

SEED = 42
N_PER_BUCKET = 50
