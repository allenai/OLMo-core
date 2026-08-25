"""
NATIVE olmo_core eval for **document-chunked** Qwen3-4B models across the FULL long-context LADDER,
emitting a flat-keyed ``<task>_<rung>`` JSON (parsed directly by ``viz/collect_headline.py``).

This is the docchunk analogue of the standard native ladder eval ``ctc_eval/eval/eval_lc_native.py``:
it reuses the box-marker chunked prefill + bs=1 KV-cached greedy decode machinery of the single-task
doc-chunk evals (``eval_lc_native_docchunk.py``), but loops every task over its length ladder using the
**SAME rung -> eval-JSONL mapping + graders as ``eval_lc_native.py``** (v1/v2 LADDERS + LSPEC), so the
resulting JSON is directly comparable to the dense/landmark ladder JSONs.

Tasks (9): the 5 in-distribution v2 tasks
    contradiction, nq, oolong, rerank, outlier
plus the 4 OOD generalization ladders (graded by REUSING the in-distribution graders):
    fiqa           = retrieval-f1  (held-out BEIR)
    scifact        = retrieval-f1  (held-out BEIR)
    outlier_review = outlier-f1    (Amazon-review passages, different source than wiki100w)
    contra_fever   = contradiction-f1 (FEVER claims, different source than pubmed)

Per-task chunk granularity (matches the training converter):
    contradiction / contra_fever = (task=contradiction, chunk_by=document)
    nq / fiqa / scifact          = (task=retrieval,     chunk_by=document)
    outlier / outlier_review     = (task=outlier,       chunk_by=document)
    oolong                       = (task=oolong,        chunk_by=line, item_regex='\\|\\|')
    rerank                       = (task=rerank,        chunk_by=document)
``cot_mode='none'`` (NO-CoT) throughout.

The rung ladder files resolve against ``EVAL500_ROOT`` (env), exactly like ``eval_lc_native.py`` -- on
Beaker the runner sets ``EVAL500_ROOT=$PRASANNS/_eval_bundle_eval500_v2`` (v2) so ``{E5}/<sub>/<file>``
lands on the weka bundle. The relative base-rung / v1-oolong files resolve via ``--root`` (chdir to the
weka $BUNDLE). *** KEEP build_ladders()/LSPEC IN SYNC WITH eval_lc_native.py. ***

Variants:
  * ``dense``        -> DocumentChunkedAttention, dense emitter (box markers, no landmark tokens).
  * ``hierarchical`` -> loaded EXACTLY like dense (config.json drives the attention class).
  * ``full``         -> plain full attention baseline (no chunked mask; box markers ignored).
  * ``landmark``     -> DocumentLandmarkAttention, landmark emitter (a landmark token every --mem-freq).
  * ``summary``      -> SummaryTokenAttention, with a fixed ``<|summ|>`` run after every document.

Decode is the bs=1 greedy KV-cached loop from the single-task scripts. To stay inside a contended
Beaker wall-clock budget, SPLIT with ``--tasks`` and ``--rungs`` (the launcher fans out one Beaker job
per task); every invocation MERGES its ``<task>_<rung>`` keys into ``--out``.

    PYTHONPATH=<repo>/src/scripts:<repo>/src torchrun --nproc_per_node=8 \
      src/scripts/ctc_eval/eval/eval_lc_native_docchunk_ladder.py \
      --variant dense --model-path <step_dir> --ladder-version v2 \
      --out outputs/eval_results/<name>_ladder.json
"""

import argparse
import json
import math
import os
import sys
import time

import torch

# Reserved ids (match the converter + olmo_core.data.document_chunk_landmark defaults).
EOS_TOKEN_ID = 151643
# The SFT model ends its assistant turn with <|im_end|> (chat template), and 151643 is appended by the
# converter UNSUPERVISED (not in the loss mask) -- so the model reliably emits 151645 but often never
# emits 151643. Stopping ONLY on 151643 made the (stopkind=None) tasks (outlier/nq/...) decode the full
# max_new budget every example (huge slowdown; outlier@3k ~40min). Stop on 151645 too -> stops at the
# real answer end (answer is fully emitted before <|im_end|>), ~10x faster, results unchanged.
IM_END_ID = 151645  # <|im_end|>
LANDMARK_TOKEN_ID = 151860
DOC_START_ID = 151648  # <|box_start|>
DOC_END_ID = 151649  # <|box_end|>
PAD_TOKEN_ID = 151863

# Full task order (5 in-distribution v2 tasks + 4 OOD ladders).
ALL_TASKS = ["contradiction", "nq", "oolong", "rerank", "outlier",
             "fiqa", "scifact", "outlier_review", "contra_fever"]


def validate_summary_checkpoint_config(model_path, ids_set, n_summary_tokens):
    """Fail before model allocation if the eval layout disagrees with the trained checkpoint."""
    cfg_path = os.path.join(model_path, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read summary checkpoint config {cfg_path}: {exc}") from exc

    model = cfg.get("model") or {}
    summary_cfg = model.get("summary_token_attention")
    if not isinstance(summary_cfg, dict):
        raise ValueError(
            f"{cfg_path}: model.summary_token_attention is missing; refusing to run the summary "
            "emitter against a checkpoint that does not declare the matching role builder"
        )
    expected_ids = {
        "doc_start_id": ids_set.doc_start,
        "doc_end_id": ids_set.doc_end,
        "summary_token_id": ids_set.summary,
        "eos_id": ids_set.eos,
        "pad_id": ids_set.pad,
    }
    wrong = {
        key: (summary_cfg.get(key), expected)
        for key, expected in expected_ids.items()
        if summary_cfg.get(key) != expected
    }
    if wrong:
        raise ValueError(f"{cfg_path}: summary reserved-id mismatch (saved, eval)={wrong}")

    declared_counts = []

    def walk(node):
        if isinstance(node, dict):
            name = str(node.get("name", "")).lower()
            cls = str(node.get("_CLASS_", "")).lower()
            if "n_summary_tokens" in node and ("summary" in name or "summary" in cls):
                declared_counts.append(node["n_summary_tokens"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(model)
    if not declared_counts:
        raise ValueError(f"{cfg_path}: no SummaryTokenAttention n_summary_tokens setting found")
    if any(value != n_summary_tokens for value in declared_counts):
        raise ValueError(
            f"{cfg_path}: checkpoint n_summary_tokens={declared_counts}, but eval requested "
            f"{n_summary_tokens}"
        )
    return summary_cfg


# ----------------------------------------------------------------------------------------------------
# LADDER definition -- rung -> eval-JSONL mapping. This MIRRORS eval_lc_native.py's v1/v2 LADDERS EXACTLY
# (so the docchunk ladder JSON is directly comparable to the dense/landmark ladder JSONs). Keep in sync.
# ----------------------------------------------------------------------------------------------------
def build_ladders(args):
    E5 = os.environ.get("EVAL500_ROOT", "/scratch/users/prasann/cpt_data/eval500")
    # v1 ladders are DISABLED (2026-07-29). Each v1 rung drew its OWN questions, so every
    # rung-to-rung delta carried eval-set resampling noise on top of the length effect it was
    # meant to isolate. v2 fixes the question set across rungs and varies only the distractors.
    if args.ladder_version not in ("v2", "v3"):
        raise NotImplementedError(
            f"--ladder-version {args.ladder_version!r} is no longer supported: v2 and v3 are the "
            "ladders. Build what you need as v2 (build_v2_eval_ladders.py for 2k-32k, "
            "build_xlong_rungs.py for 64k-2M) and point EVAL500_ROOT at a v2 or v3 bundle."
        )
    if args.ladder_version in ("v2", "v3"):
        # v3 shares v2's relative layout, so the same table serves both -- the bundle root is what
        # differs (contra + outlier are rebuilt, nq/rerank/oolong are symlinks back to v2_clean, and
        # the OOD BEIR sets have no v3 build at all). Rungs whose file is absent under the selected
        # root are skipped with a `MISSING` line rather than faked, which is how a v3 run drops the
        # tasks v3 does not have.
        # v2: every rung of a task shares the SAME >=500 questions; only distractor docs differ. ALL
        # rungs live under $EVAL500_ROOT/<task>/ (point EVAL500_ROOT at the v2 bundle).
        # v2 -> `both`-mode contradiction gold; v3 -> `realistic`, matching the training generator.
        # KEEP IN SYNC with eval_lc_native.py, which got this v3 renaming first: with `both`
        # hardcoded, every v3 contra rung resolved to a filename the v3 bundle does not have, so the
        # whole task MISSING-skipped and the job exited 0 with an empty JSON (the 2026-08-16
        # summtoken v3c sweep lost all nine contra cells this way).
        _CM = "realistic" if args.ladder_version == "v3" else "both"
        ladders_v2 = {
            "contradiction": [("2k", f"{E5}/contra/contradiction_eval_pubmed_{_CM}_n100_k3.jsonl"),
                ("8k", f"{E5}/contra/contradiction_eval_pubmed_{_CM}_n190_k3.jsonl"),
                ("16k", f"{E5}/contra/contradiction_eval_pubmed_{_CM}_n385_k3.jsonl"),
                ("32k", f"{E5}/contra/contradiction_eval_pubmed_{_CM}_n765_k3.jsonl")],
            "nq": [("3k", f"{E5}/nq/nq_validation_k20_600.jsonl"),
                ("8k", f"{E5}/nq/nq_validation_k50_600.jsonl"),
                ("16k", f"{E5}/nq/nq_validation_k100_600.jsonl"),
                ("32k", f"{E5}/nq/nq_validation_k200_600.jsonl")],
            "outlier": [("3k", f"{E5}/outlier/outlier_wiki100w_n22_k3_eval_600.jsonl"),
                ("8k", f"{E5}/outlier/outlier_wiki100w_n55_k3_eval_600.jsonl"),
                ("16k", f"{E5}/outlier/outlier_wiki100w_n110_k3_eval_600.jsonl"),
                ("32k", f"{E5}/outlier/outlier_wiki100w_n220_k3_eval_600.jsonl")],
            "rerank": [("3k", f"{E5}/rerank/msmarco_trainhn_eval_k20_500.jsonl"),
                ("8k", f"{E5}/rerank/msmarco_trainhn_eval_k50_500.jsonl"),
                ("16k", f"{E5}/rerank/msmarco_trainhn_eval_k100_500.jsonl")],
            "oolong": [("8k", f"{E5}/oolong/oolong_test_synth_ctx8192_spliteval.jsonl"),
                ("16k", f"{E5}/oolong/oolong_test_synth_ctx16384_spliteval.jsonl"),
                ("32k", f"{E5}/oolong/oolong_test_synth_ctx32768_spliteval.jsonl")],
            # OOD generalization (held-out BEIR retrieval, graded as retrieval f1). Version-agnostic.
            "fiqa": [("2k", f"{E5}/beir/beir_fiqa_ce_ladder_k10_648.jsonl"),
                ("4k", f"{E5}/beir/beir_fiqa_ce_ladder_k20_648.jsonl"),
                ("8k", f"{E5}/beir/beir_fiqa_ce_ladder_k40_648.jsonl"),
                ("16k", f"{E5}/beir/beir_fiqa_ce_ladder_k80_648.jsonl")],
            "scifact": [("4k", f"{E5}/beir/beir_scifact_ladder_k11_299.jsonl"),
                ("8k", f"{E5}/beir/beir_scifact_ladder_k22_299.jsonl"),
                ("16k", f"{E5}/beir/beir_scifact_ladder_k44_299.jsonl"),
                ("32k", f"{E5}/beir/beir_scifact_ladder_k88_299.jsonl")],
            # OOD outlier + contradiction (review / FEVER source; graded identically via gold indices).
            "outlier_review": [("3k", f"{E5}/outlier/outlier_review_matched_n30_k3_eval_600.jsonl"),
                ("8k", f"{E5}/outlier/outlier_review_matched_n75_k3_eval_600.jsonl"),
                ("16k", f"{E5}/outlier/outlier_review_matched_n150_k3_eval_600.jsonl"),
                ("32k", f"{E5}/outlier/outlier_review_matched_n300_k3_eval_600.jsonl")],
            "contra_fever": [("2k", f"{E5}/contra/contradiction_eval_fever_plain_n100_k3.jsonl"),
                ("8k", f"{E5}/contra/contradiction_eval_fever_plain_n408_k3.jsonl"),
                ("16k", f"{E5}/contra/contradiction_eval_fever_plain_n820_k3.jsonl"),
                ("32k", f"{E5}/contra/contradiction_eval_fever_plain_n1642_k3.jsonl")],
        }
        # ---- OPT-IN ultra-long rungs (64k/128k/256k), OFF by default (v2 only) ----
        # KEEP IN SYNC with eval_lc_native.py: size-labelled glob so the calibrated doc-count in the
        # filename can drift without editing this file. Files staged on weka under EVAL500_ROOT/<sub>/
        # (NB contra subdir is "contra"). contra|nq|outlier only. Runner forces bs=1 + raises max_length.
        if getattr(args, "xlong", False):
            import glob as _glob

            _XL = {
                "contradiction": ("contra", f"contradiction_eval_pubmed_{_CM}_n*_k3_xlong_{{s}}.jsonl"),
                "nq": ("nq", "nq_validation_k*_xlong_{s}.jsonl"),
                "outlier": ("outlier", "outlier_wiki100w_n*_k3_eval_xlong_{s}.jsonl"),
            }
            for _t, (_sub, _pat) in _XL.items():
                for _s in ("64k", "128k", "256k"):
                    _hits = sorted(_glob.glob(os.path.join(E5, _sub, _pat.format(s=_s))))
                    if _hits:
                        ladders_v2[_t].append((_s, _hits[0]))
            print(f"[xlong] appended ultra-long rungs where files exist under {E5}", flush=True)
        return ladders_v2
    return {
        "contradiction": [("2k", args.contra_data),
            ("8k", f"{E5}/contra/contradiction_eval_pubmed_both_n190_k3.jsonl"),
            ("16k", f"{E5}/contra/contradiction_eval_pubmed_both_n385_k3.jsonl"),
            ("32k", f"{E5}/contra/contradiction_eval_pubmed_both_n765_k3.jsonl")],
        "nq": [("3k", args.nq_data),
            ("8k", f"{E5}/nq/nq_validation_k50_hn5_600.jsonl"),
            ("16k", f"{E5}/nq/nq_validation_k100_hn10_600.jsonl"),
            ("32k", f"{E5}/nq/nq_validation_k200_hn20_600.jsonl")],
        "oolong": [("1k", "data/oolong_test_synth_ctx1024_spliteval.jsonl"),
            ("2k", "data/oolong_test_synth_ctx2048_spliteval.jsonl"),
            ("4k", "data/oolong_test_synth_ctx4096_spliteval.jsonl"),
            ("8k", "data/oolong_test_synth_ctx8192_spliteval.jsonl"),
            ("16k", "data/oolong_test_synth_ctx16384_spliteval.jsonl"),
            ("32k", "data/oolong_test_synth_ctx32768_spliteval.jsonl")],
        "rerank": [("2k", args.rerank_data),
            ("8k", f"{E5}/rerank/msmarco_validation_rerank_k80_600.jsonl"),
            ("16k", f"{E5}/rerank/msmarco_validation_rerank_k158_597.jsonl"),
            ("32k", f"{E5}/rerank/msmarco_validation_rerank_k315_599.jsonl")],
        "outlier": [("3k", args.outlier_data),
            ("8k", f"{E5}/outlier/outlier_wiki100w_n55_k3_eval_600.jsonl"),
            ("16k", f"{E5}/outlier/outlier_wiki100w_n110_k3_eval_600.jsonl"),
            ("32k", f"{E5}/outlier/outlier_wiki100w_n220_k3_eval_600.jsonl")],
        "fiqa": [("2k", f"{E5}/beir/beir_fiqa_ce_ladder_k10_648.jsonl"),
            ("4k", f"{E5}/beir/beir_fiqa_ce_ladder_k20_648.jsonl"),
            ("8k", f"{E5}/beir/beir_fiqa_ce_ladder_k40_648.jsonl"),
            ("16k", f"{E5}/beir/beir_fiqa_ce_ladder_k80_648.jsonl")],
        "scifact": [("4k", f"{E5}/beir/beir_scifact_ladder_k11_299.jsonl"),
            ("8k", f"{E5}/beir/beir_scifact_ladder_k22_299.jsonl"),
            ("16k", f"{E5}/beir/beir_scifact_ladder_k44_299.jsonl"),
            ("32k", f"{E5}/beir/beir_scifact_ladder_k88_299.jsonl")],
        "outlier_review": [("3k", f"{E5}/outlier/outlier_review_matched_n30_k3_eval_600.jsonl"),
            ("8k", f"{E5}/outlier/outlier_review_matched_n75_k3_eval_600.jsonl"),
            ("16k", f"{E5}/outlier/outlier_review_matched_n150_k3_eval_600.jsonl"),
            ("32k", f"{E5}/outlier/outlier_review_matched_n300_k3_eval_600.jsonl")],
        "contra_fever": [("2k", f"{E5}/contra/contradiction_eval_fever_plain_n100_k3.jsonl"),
            ("8k", f"{E5}/contra/contradiction_eval_fever_plain_n408_k3.jsonl"),
            ("16k", f"{E5}/contra/contradiction_eval_fever_plain_n820_k3.jsonl"),
            ("32k", f"{E5}/contra/contradiction_eval_fever_plain_n1642_k3.jsonl")],
    }


# Per-task segmentation + scoring spec. loadtask/chunk_by/item_regex match the training converter; the
# grader + primary-metric key match eval_lc_native.py's LSPEC (OOD tasks reuse the base graders).
#   primary_key=None -> rerank: take the first "mrr*" metric.
def build_task_spec(args):
    return {
        # task:            (loadtask,        chunk_by,   item_regex, primary_key, max_new_tokens,            stopkind)
        "contradiction":  ("contradiction", "document", r"\|\|",   "f1",        args.contra_max_new_tokens,  "contra"),
        "nq":             ("retrieval",     "document", r"\|\|",   "f1",        args.nq_max_new_tokens,      None),
        "oolong":         ("oolong",        "line",     r"\|\|",   "score",     args.oolong_max_new_tokens,  "oolong"),
        "rerank":         ("rerank",        "document", r"\|\|",   None,        args.rerank_max_new_tokens,  None),
        "outlier":        ("outlier",       "document", r"\|\|",   "f1",        args.outlier_max_new_tokens, None),
        # ---- OOD ladders: reuse the in-distribution graders + segmentation of their base task ----
        "fiqa":           ("retrieval",     "document", r"\|\|",   "f1",        args.nq_max_new_tokens,      None),
        "scifact":        ("retrieval",     "document", r"\|\|",   "f1",        args.nq_max_new_tokens,      None),
        "outlier_review": ("outlier",       "document", r"\|\|",   "f1",        args.outlier_max_new_tokens, None),
        "contra_fever":   ("contradiction", "document", r"\|\|",   "f1",        args.contra_max_new_tokens,  "contra"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=["dense", "landmark", "full", "hierarchical", "summary"],
                    help="hierarchical loads EXACTLY like dense (DocumentChunkedAttention, dense "
                         "doc-chunk data, no landmarks) -- config.json drives the attention class.")
    ap.add_argument("--model-path", required=True, help="step dir: config.json + model_and_optim/")
    ap.add_argument("--out", required=True,
                    help="ladder JSON. MERGES <task>_<rung> keys into an existing file if present, so "
                         "per-task/per-rung split invocations accumulate into one ladder JSON.")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--tokenizer-family", choices=["qwen3", "qwen3_5"], default="qwen3")
    ap.add_argument("--num-summary-tokens", type=int, default=5,
                    help="summary variant only: <|summ|> tokens appended after every document; "
                         "must match the training shards and checkpoint config.")
    ap.add_argument("--ladder-version", choices=["v2", "v3"], default="v2",
                    help="v2 (DEFAULT): every rung of a task shares the SAME >=500 questions, only "
                         "distractors vary (reads the _eval_bundle_eval500_v2 bundle via EVAL500_ROOT). "
                         "v3: same layout, contradiction rebuilt in `realistic` mode; v3 contradiction "
                         "numbers are NOT comparable to v2 ones.")
    ap.add_argument("--summary-mask-mode", choices=["causal", "restricted"], default="causal",
                    help="summary variant only: which arm of the mask mixture to serve. causal "
                         "(DEFAULT) = plain causal attention with <|summ|> tokens present as ordinary "
                         "tokens, which is the arm a standard_mix_prob=1.0 or mix_end_p=1.0 model "
                         "trained on. restricted = the full summary mask, where the query reads only "
                         "summaries and its own document.")
    ap.add_argument("--mem-freq", type=int, default=63,
                    help="landmark variant: a landmark token every mem_freq content tokens (window=64).")
    ap.add_argument("--max-test-samples", type=int, default=400)
    ap.add_argument("--max-length", type=int, default=40960,
                    help="KV-cache length budget (covers the 32k rung + box/landmark token overhead).")

    # ---- task/rung filters (let the launcher split a slow ladder to dodge wall-clock timeouts) ----
    ap.add_argument("--tasks", default=",".join(ALL_TASKS),
                    help="comma list restricting which of the 9 tasks to run.")
    ap.add_argument("--rungs", default=None,
                    help="comma list restricting rungs (e.g. '16k,32k'); applied across tasks.")
    ap.add_argument("--xlong", action="store_true",
                    default=os.environ.get("LADDER_XLONG") == "1",
                    help="OPT-IN (v2 only): append the ultra-long 64k/128k/256k rungs (contra|nq|outlier) "
                         "by size-labelled glob under EVAL500_ROOT. Combine with --rungs 64k,128k to pick "
                         "which; the runner forces bs=1 and raises --max-length. Honors env LADDER_XLONG=1.")

    # ---- per-task max-new-tokens knobs ----
    ap.add_argument("--contra-max-new-tokens", type=int, default=96,
                    help="NO-CoT contradiction: the 'Contradicting pairs: [[...]]' line is short.")
    ap.add_argument("--nq-max-new-tokens", type=int, default=64)
    ap.add_argument("--oolong-max-new-tokens", type=int, default=200)
    ap.add_argument("--rerank-max-new-tokens", type=int, default=256)
    ap.add_argument("--outlier-max-new-tokens", type=int, default=200)

    # ---- per-task default (2k/3k) base eval JSONLs used only for --ladder-version v1 ----
    ap.add_argument("--contra-data", default="data/contradiction_eval_pubmed_both_n100_k3.jsonl")
    ap.add_argument("--nq-data", default="data/nq_validation_k20_hn2_600.jsonl")
    ap.add_argument("--rerank-data", default="data/msmarco_dev_rerank_k20_1000.jsonl")
    ap.add_argument("--outlier-data", default="data/outlier_wiki100w_n20_k3_eval_100.jsonl")
    ap.add_argument("--root", default=None,
                    help="chdir here before resolving relative data paths (on-cluster: the weka $BUNDLE, "
                         "so relative data/... base + v1-oolong files resolve).")
    # accepted for launcher parity; docchunk builds its own prefill via segment_prompt_to_chunks.
    ap.add_argument("--cot-mode", default="none", choices=["none", "plan"],
                    help="OOLONG CoT mode used to build the prefill prompt. 'none' (default) keeps the "
                         "no-CoT behavior byte-identical; 'plan' matches a CoT-trained checkpoint so the "
                         "model can externalize cross-item aggregation into (free/global) generated tokens.")
    ap.add_argument("--prompt-format", default="chat", choices=["chat", "raw", "alpaca"],
                    help="accepted for runner parity; docchunk prefill mirrors the training converter.")

    # ---- landmark-only top-k inference knobs (exact if unset; mirror the single-task scripts) ----
    ap.add_argument("--landmark-top-k-blocks", type=int, default=None,
                    help="landmark variant: keep only top-k landmark BLOCKS per query (exact if unset).")
    ap.add_argument("--landmark-top-k-fraction", type=float, default=None,
                    help="landmark variant: top-k = ceil(fraction * num_prompt_blocks) (reference: 0.1).")
    ap.add_argument("--save-generations", action=argparse.BooleanOptionalAction, default=True,
                    help="Dump per-example (prompt tail, generation, gold/pred detail) to "
                         "<out>.generations.jsonl for error inspection. On by default; "
                         "--no-save-generations to skip.")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.root:
        os.chdir(args.root)

    from ctc_eval.eval.evaluate import (
        _eval_contradiction,
        _eval_oolong,
        _eval_outlier,
        _eval_rerank,
        _eval_retrieval,
        load_unified_examples,
    )
    from transformers import AutoTokenizer

    from olmo_core.config import DType
    from olmo_core.data.document_chunk_landmark import (
        emit_document_chunk_dense,
        emit_document_chunk_landmark,
        emit_document_chunk_summary,
        reserved_ids,
        segment_prompt_to_chunks,
    )
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import (
        TransformerGenerationModuleConfig,
    )

    # Grader per task. OOD ladders reuse their base-task grader (fiqa/scifact->retrieval,
    # outlier_review->outlier, contra_fever->contradiction), matching eval_lc_native.py's LSPEC.
    SCORE_FNS = {
        "contradiction": _eval_contradiction,
        "nq": _eval_retrieval,
        "oolong": _eval_oolong,
        "rerank": _eval_rerank,
        "outlier": _eval_outlier,
        "fiqa": _eval_retrieval,
        "scifact": _eval_retrieval,
        "outlier_review": _eval_outlier,
        "contra_fever": _eval_contradiction,
    }

    # ---- data-parallel across ranks (torchrun) ----
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
    is_main = rank == 0
    if not is_main:
        sys.stdout = open(os.devnull, "w")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids_set = reserved_ids(args.tokenizer_family)
    if args.variant == "summary":
        validate_summary_checkpoint_config(args.model_path, ids_set, args.num_summary_tokens)
        print(
            f"[summary-layout] validated checkpoint config: family={args.tokenizer_family} "
            f"summary_token_id={ids_set.summary} tokens_per_document={args.num_summary_tokens}",
            flush=True,
        )
    eos_token_id = ids_set.eos
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    NEWLINE_ID = tok("\n", add_special_tokens=False).input_ids[-1]
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    is_landmark = args.variant == "landmark"
    is_full = args.variant == "full"
    use_dense_emit = not is_landmark

    t0 = time.time()
    # GenerationConfig requires pad != eos; Qwen3 has no pad and we decode bs=1 (pad unused).
    pad_id = tok.pad_token_id if tok.pad_token_id not in (None, eos_token_id) else im_end_id
    gen_cfg = GenerationConfig(
        eos_token_id=eos_token_id,
        pad_token_id=pad_id,
        max_length=args.max_length,
        use_cache=True,
    )
    gm = TransformerGenerationModuleConfig(
        gen_cfg, float8_config=None, dtype=DType("bfloat16"), compile_model=False
    ).build(checkpoint_dir=args.model_path, device=device)
    if not is_full and args.variant != "summary":
        gm.model.enable_document_chunk_attention(
            doc_start_id=ids_set.doc_start,
            doc_end_id=ids_set.doc_end,
            eos_id=eos_token_id,
            mode="chunked",
            pad_id=ids_set.pad if is_landmark else None,
        )
    print(f"[docchunk-ladder-{args.variant}] built from {args.model_path} in {time.time()-t0:.1f}s",
          flush=True)

    if args.variant == "summary":
        mode = gm.model.set_summary_eval_mask_mode(args.summary_mask_mode)
        print(f"[summary-mask] serving the '{mode}' arm of the mask mixture", flush=True)

    if is_landmark and args.landmark_top_k_blocks is not None:
        n_set = gm.model.set_landmark_eval_top_k(args.landmark_top_k_blocks)
        print(f"[topk] fixed top_k={args.landmark_top_k_blocks} on {n_set} landmark layers", flush=True)

    block_size = args.mem_freq + 1  # landmark window (64); eager landmark forward needs T % 64 == 0

    def build_prefill(raw_example, loadtask, chunk_by, item_regex):
        segs, _ids, _ = segment_prompt_to_chunks(
            tok, raw_example, loadtask, query_position="both", cot_mode=args.cot_mode,
            chunk_by=chunk_by, item_regex=item_regex, include_answer=False,
            doc_start_id=ids_set.doc_start, doc_end_id=ids_set.doc_end,
        )
        if args.variant == "summary":
            out, _ = emit_document_chunk_summary(
                segs, summary_token_id=ids_set.summary,
                n_summary_tokens=args.num_summary_tokens,
            )
        elif use_dense_emit:
            out, _ = emit_document_chunk_dense(segs)  # box markers present; full attention ignores them
        else:
            out, _ = emit_document_chunk_landmark(
                segs, mem_freq=args.mem_freq, mem_id=ids_set.landmark, pad_id=ids_set.pad
            )
        return out

    def strip_think(text):
        return text.split("</think>", 1)[1] if "</think>" in text else text

    def make_answer_complete(stopkind):
        if stopkind == "oolong":
            def _f(content_ids):
                return "answer:" in tok.decode(content_ids, skip_special_tokens=True).lower()
            return _f
        if stopkind == "contra":
            def _f(content_ids):
                txt = tok.decode(
                    [t for t in content_ids if t != ids_set.landmark], skip_special_tokens=True
                ).lower()
                return "contradicting pairs:" in txt
            return _f
        return None

    @torch.no_grad()
    def generate_one(prefill, max_new_tokens, answer_complete):
        # Size the KV cache to THIS example's actual need (prefill + decode budget), not the raised
        # --max-length. At the xlong 64k/128k rungs the runner raises --max-length to cover the longest
        # rung, so a full-max_length cache wastes ~10 GiB at the 64k rung and starves the FlexAttention
        # prefill kernel (the OOM that killed the docchunk xlong jobs). The landmark decode also inserts
        # a landmark token every mem_freq generated tokens, so budget for those extra cache slots. The
        # cache manager only reallocates when it needs to GROW (is_reusable keeps a larger buffer), so
        # this never shrinks a cache mid-rung. Capped at args.max_length as a hard ceiling.
        lm_overhead = (max_new_tokens // args.mem_freq + 2) if is_landmark else 0
        cache_len = min(len(prefill) + max_new_tokens + lm_overhead + 1, args.max_length)
        gm.prepare_inference_cache(1, cache_len)  # (re)set the cache cursor to 0 per example
        leftpad = torch.zeros(1, dtype=torch.int32, device=device)
        if not is_landmark:
            # Dense / hierarchical / full: prefill once (chunked mask applied + K,V cached for the
            # chunked variants), then single-token greedy decode over the cache (plain causal).
            logits = gm.model(
                torch.tensor([prefill], device=device), logits_to_keep=1, cache_leftpad=leftpad
            )
            nxt = int(logits[0, -1].argmax().item())
            new_content = []
            for _ in range(max_new_tokens):
                if nxt == eos_token_id or nxt == im_end_id:
                    break
                new_content.append(nxt)
                if nxt == NEWLINE_ID and answer_complete is not None and answer_complete(new_content):
                    break
                logits = gm.model(torch.tensor([[nxt]], device=device), logits_to_keep=1)
                nxt = int(logits[0, -1].argmax().item())
            return strip_think(tok.decode(new_content, skip_special_tokens=True))

        # Landmark: KV-cached decode. Prefill (block-aligned) runs once with the chunked grouped-softmax
        # mask + K,V cached; then each generated token is fed incrementally and decoded as a plain
        # landmark query (FREE -> chunk mask is a no-op). A real landmark token is inserted after every
        # mem_freq generated content tokens so the periodic is_mem structure matches the eager loop.
        logits = gm.model(
            torch.tensor([prefill], device=device), logits_to_keep=1, cache_leftpad=leftpad
        )
        nxt = int(logits[0, -1].argmax().item())
        new_content = []
        since_landmark = 0
        for _ in range(max_new_tokens):
            if nxt == eos_token_id or nxt == im_end_id:
                break
            new_content.append(nxt)
            logits = gm.model(torch.tensor([[nxt]], device=device), logits_to_keep=1)
            since_landmark += 1
            if since_landmark == args.mem_freq:
                logits = gm.model(torch.tensor([[ids_set.landmark]], device=device), logits_to_keep=1)
                since_landmark = 0
            if nxt == NEWLINE_ID and answer_complete is not None and answer_complete(new_content):
                break
            nxt = int(logits[0, -1].argmax().item())
        return strip_think(tok.decode(new_content, skip_special_tokens=True))

    # ---- output: MERGE into an existing ladder JSON so per-task/rung splits accumulate ----
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            summary = {}
    summary["model_path"] = args.model_path
    summary["variant"] = args.variant
    summary["ladder_version"] = args.ladder_version

    def flush():
        if is_main:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=2)

    # Per-example generation dump (for error inspection), same schema/convention as
    # eval_lc_native.py's _record_gens: {task, rung, idx, generation, prompt_tail, detail}, one
    # JSON object per line, appended incrementally after each rung so a preempted/OOM-killed job
    # doesn't lose everything already decoded. Deliberately does NOT merge with a pre-existing
    # dump (unlike the summary JSON above) -- a rerun must fully replace stale generations, not
    # silently inherit them, so the file is truncated once at process startup.
    gen_path = os.path.splitext(args.out)[0] + ".generations.jsonl"
    gen_dump = []
    gens_written = 0
    if is_main and args.save_generations:
        open(gen_path, "w").close()

    def flush_gens():
        nonlocal gens_written
        if not (is_main and args.save_generations) or len(gen_dump) <= gens_written:
            return
        with open(gen_path, "a") as gf:
            for rec in gen_dump[gens_written:]:
                gf.write(json.dumps(rec) + "\n")
        gens_written = len(gen_dump)

    ladders = build_ladders(args)
    task_spec = build_task_spec(args)
    task_filter = [t.strip() for t in args.tasks.split(",") if t.strip()]
    rung_filter = set(r.strip() for r in args.rungs.split(",")) if args.rungs else None

    for task in ALL_TASKS:
        if task not in task_filter or task not in ladders:
            continue
        loadtask, chunk_by, item_regex, primary_key, max_new_tokens, stopkind = task_spec[task]
        answer_complete = make_answer_complete(stopkind)
        score_fn = SCORE_FNS[task]
        cap = args.max_length - max_new_tokens
        rungs = ladders[task]
        if rung_filter:
            rungs = [(lab, p) for (lab, p) in rungs if lab in rung_filter]

        for label, path in rungs:
            if not path or not os.path.exists(path):
                print(f"[ladder:{task}@{label}] MISSING {path}, skipping", flush=True)
                continue
            examples = load_unified_examples(
                path, args.max_test_samples, task=loadtask,
                query_position="both", use_alpaca=True,
            )
            # bs=1 DP shard: this rank decodes global indices [rank, rank+world, ...].
            my_gidx = list(range(rank, len(examples), world))
            local = []
            for gi in my_gidx:
                raw = examples[gi].get("ex", examples[gi])
                prefill = build_prefill(raw, loadtask, chunk_by, item_regex)
                if len(prefill) > cap:
                    # An over-budget prefill used to be scored as an empty generation, i.e. graded
                    # WRONG on an example the model was never shown -- a config error laundered
                    # into the rung's metric, where it reads as a long-context collapse. Fail.
                    raise SystemExit(
                        f"[maxlen] {task}@{label} example {gi} builds a {len(prefill)}-token "
                        f"prefill, past the {cap}-token cap (--max-length {args.max_length} minus "
                        f"max_new_tokens {max_new_tokens}). Scoring it as empty would grade the "
                        f"model on an example it never saw. Re-run with --max-length >= "
                        f"{len(prefill) + max_new_tokens}."
                    )
                if is_landmark and args.landmark_top_k_fraction is not None:
                    n_blocks = max(1, len(prefill) // block_size)
                    gm.model.set_landmark_eval_top_k(
                        max(1, math.ceil(args.landmark_top_k_fraction * n_blocks))
                    )
                # Prompt tail for the generations dump: decode a bounded window of the ACTUAL
                # token-level prefill (box markers / landmark / summary tokens included), not the
                # raw example text -- that's what the model was really shown. Gathering the full
                # prefill (up to ~250k tokens at the long rungs) across ranks would be enormous;
                # a decoded tail is a few hundred chars.
                ptail = tok.decode(prefill[-512:], skip_special_tokens=False)[-1200:]
                local.append((gi, generate_one(prefill, max_new_tokens, answer_complete), ptail))

            full = [None] * len(examples)
            prompt_tails = [None] * len(examples)
            if world > 1:
                parts = [None] * world
                torch.distributed.all_gather_object(parts, local)
                for part in parts:
                    for gi, resp, ptail in part:
                        full[gi] = resp
                        prompt_tails[gi] = ptail
            else:
                for gi, resp, ptail in local:
                    full[gi] = resp
                    prompt_tails[gi] = ptail

            if is_main:
                res, det = score_fn(examples, full)
                if args.save_generations:
                    for i, resp in enumerate(full):
                        rec = {"task": task, "rung": label, "idx": i, "generation": resp,
                               "prompt_tail": prompt_tails[i]}
                        if det is not None and i < len(det):
                            rec["detail"] = det[i]
                        gen_dump.append(rec)
                    flush_gens()
                if primary_key is not None:
                    prim = res.get(primary_key)
                else:  # rerank -> first mrr* metric
                    prim = next((v for k, v in res.items() if k.startswith("mrr")), None)
                summary[f"{task}_{label}"] = (float(prim) if prim is not None else None)
                print(f"[ladder:{task}@{label}] {primary_key or 'mrr'}="
                      f"{prim if prim is None else round(float(prim), 3)} "
                      f"(n={len(examples)})", flush=True)
                flush()
            if world > 1:
                torch.distributed.barrier()

    flush()
    if is_main:
        print(f"\n[docchunk-ladder] WROTE {args.out} in {time.time()-t0:.1f}s", flush=True)
        keys = sorted(k for k in summary if any(k.startswith(t + "_") for t in ALL_TASKS))
        print(f"[docchunk-ladder] ladder keys present: {keys}", flush=True)
        if args.save_generations and gen_dump:
            print(f"[docchunk-ladder] wrote {len(gen_dump)} generations -> {gen_path}", flush=True)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
