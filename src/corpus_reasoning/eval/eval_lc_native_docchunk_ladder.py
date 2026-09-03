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
from olmo_core.data.document_chunk_landmark import (  # canonical ids -- never retype
    DOC_END_ID,
    DOC_START_ID,
    EOS_TOKEN_ID,
    LANDMARK_TOKEN_ID,
    PAD_TOKEN_ID,
    REAL_VOCAB_SIZE,
)

# Reserved ids (match the converter + olmo_core.data.document_chunk_landmark defaults).

# Full task order (5 in-distribution v2 tasks + 4 OOD ladders).
ALL_TASKS = ["contradiction", "nq", "oolong", "rerank", "outlier",
             "fiqa", "scifact", "outlier_review", "contra_fever"]


# ----------------------------------------------------------------------------------------------------
# LADDER definition -- rung -> eval-JSONL mapping. This MIRRORS eval_lc_native.py's v1/v2 LADDERS EXACTLY
# (so the docchunk ladder JSON is directly comparable to the dense/landmark ladder JSONs). Keep in sync.
# ----------------------------------------------------------------------------------------------------
def build_ladders(args):
    E5 = os.environ.get("EVAL500_ROOT", "/scratch/users/prasann/cpt_data/eval500")
    # v1 ladders are DISABLED (2026-07-29). Each v1 rung drew its OWN questions, so every
    # rung-to-rung delta carried eval-set resampling noise on top of the length effect it was
    # meant to isolate. v2 fixes the question set across rungs and varies only the distractors.
    if args.ladder_version != "v2":
        raise NotImplementedError(
            f"--ladder-version {args.ladder_version!r} is no longer supported: v2 is the only "
            "ladder. Build what you need as v2 (build_v2_eval_ladders.py for 2k-32k, "
            "build_xlong_rungs.py for 64k-2M) and point EVAL500_ROOT at a v2 bundle."
        )
    if args.ladder_version == "v2":
        # v2: every rung of a task shares the SAME >=500 questions; only distractor docs differ. ALL
        # rungs live under $EVAL500_ROOT/<task>/ (point EVAL500_ROOT at the v2 bundle).
        return {
            "contradiction": [("2k", f"{E5}/contra/contradiction_eval_pubmed_both_n100_k3.jsonl"),
                ("8k", f"{E5}/contra/contradiction_eval_pubmed_both_n190_k3.jsonl"),
                ("16k", f"{E5}/contra/contradiction_eval_pubmed_both_n385_k3.jsonl"),
                ("32k", f"{E5}/contra/contradiction_eval_pubmed_both_n765_k3.jsonl")],
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
    ap.add_argument("--variant", required=True, choices=["dense", "landmark", "full", "hierarchical"],
                    help="hierarchical loads EXACTLY like dense (DocumentChunkedAttention, dense "
                         "doc-chunk data, no landmarks) -- config.json drives the attention class.")
    ap.add_argument("--model-path", required=True, help="step dir: config.json + model_and_optim/")
    ap.add_argument("--out", required=True,
                    help="ladder JSON. MERGES <task>_<rung> keys into an existing file if present, so "
                         "per-task/per-rung split invocations accumulate into one ladder JSON.")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--ladder-version", choices=["v2"], default="v2",
                    help="v2 (DEFAULT): every rung of a task shares the SAME >=500 questions, only "
                         "distractors vary (reads the _eval_bundle_eval500_v2 bundle via EVAL500_ROOT).")
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
    ap.add_argument("--prompt-format", default="chat", choices=["chat", "raw", "alpaca"],
                    help="accepted for runner parity; docchunk prefill mirrors the training converter.")

    # ---- landmark-only top-k inference knobs (exact if unset; mirror the single-task scripts) ----
    ap.add_argument("--landmark-top-k-blocks", type=int, default=None,
                    help="landmark variant: keep only top-k landmark BLOCKS per query (exact if unset).")
    ap.add_argument("--landmark-top-k-fraction", type=float, default=None,
                    help="landmark variant: top-k = ceil(fraction * num_prompt_blocks) (reference: 0.1).")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.root:
        os.chdir(args.root)

    from transformers import AutoTokenizer

    from olmo_core.config import DType
    from olmo_core.data.document_chunk_landmark import (
        DOC_END_ID as _DE,
    )
    from olmo_core.data.document_chunk_landmark import (
        DOC_START_ID as _DS,
    )
    from olmo_core.data.document_chunk_landmark import (
        emit_document_chunk_dense,
        emit_document_chunk_landmark,
        segment_prompt_to_chunks,
    )
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import TransformerGenerationModuleConfig
    from corpus_reasoning.eval.evaluate import (
        _eval_contradiction,
        _eval_oolong,
        _eval_outlier,
        _eval_rerank,
        _eval_retrieval,
        load_unified_examples,
    )

    assert (_DS, _DE) == (DOC_START_ID, DOC_END_ID)

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
    NEWLINE_ID = tok("\n", add_special_tokens=False).input_ids[-1]
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    is_landmark = args.variant == "landmark"
    is_full = args.variant == "full"
    use_dense_emit = not is_landmark

    t0 = time.time()
    # GenerationConfig requires pad != eos; Qwen3 has no pad and we decode bs=1 (pad unused).
    pad_id = tok.pad_token_id if tok.pad_token_id not in (None, EOS_TOKEN_ID) else 151645
    gen_cfg = GenerationConfig(
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=pad_id,
        max_length=args.max_length,
        use_cache=True,
    )
    gm = TransformerGenerationModuleConfig(
        gen_cfg, float8_config=None, dtype=DType("bfloat16"), compile_model=False
    ).build(checkpoint_dir=args.model_path, device=device)
    if not is_full:
        gm.model.enable_document_chunk_attention(
            doc_start_id=DOC_START_ID,
            doc_end_id=DOC_END_ID,
            eos_id=EOS_TOKEN_ID,
            mode="chunked",
            pad_id=PAD_TOKEN_ID if is_landmark else None,
        )
    print(f"[docchunk-ladder-{args.variant}] built from {args.model_path} in {time.time()-t0:.1f}s",
          flush=True)

    if is_landmark and args.landmark_top_k_blocks is not None:
        n_set = gm.model.set_landmark_eval_top_k(args.landmark_top_k_blocks)
        print(f"[topk] fixed top_k={args.landmark_top_k_blocks} on {n_set} landmark layers", flush=True)

    block_size = args.mem_freq + 1  # landmark window (64); eager landmark forward needs T % 64 == 0

    def build_prefill(raw_example, loadtask, chunk_by, item_regex):
        segs, _ids, _ = segment_prompt_to_chunks(
            tok, raw_example, loadtask, query_position="both", cot_mode="none",
            chunk_by=chunk_by, item_regex=item_regex, include_answer=False,
            doc_start_id=DOC_START_ID, doc_end_id=DOC_END_ID,
        )
        if use_dense_emit:
            out, _ = emit_document_chunk_dense(segs)  # box markers present; full attention ignores them
        else:
            out, _ = emit_document_chunk_landmark(
                segs, mem_freq=args.mem_freq, mem_id=LANDMARK_TOKEN_ID, pad_id=PAD_TOKEN_ID
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
                    [t for t in content_ids if t != LANDMARK_TOKEN_ID], skip_special_tokens=True
                ).lower()
                return "contradicting pairs:" in txt
            return _f
        return None

    @torch.no_grad()
    def generate_one(prefill, max_new_tokens, answer_complete):
        gm.prepare_inference_cache(1, args.max_length)  # (re)set the cache cursor to 0 per example
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
                if nxt == EOS_TOKEN_ID:
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
            if nxt == EOS_TOKEN_ID:
                break
            new_content.append(nxt)
            logits = gm.model(torch.tensor([[nxt]], device=device), logits_to_keep=1)
            since_landmark += 1
            if since_landmark == args.mem_freq:
                logits = gm.model(torch.tensor([[LANDMARK_TOKEN_ID]], device=device), logits_to_keep=1)
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
    # reported-key -> EVAL EXAMPLES scored (never "n": `n` = corpus size).
    summary.setdefault("eval_size", {})

    def flush():
        if is_main:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=2)

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
                local.append((gi, generate_one(prefill, max_new_tokens, answer_complete)))

            full = [None] * len(examples)
            if world > 1:
                parts = [None] * world
                torch.distributed.all_gather_object(parts, local)
                for part in parts:
                    for gi, resp in part:
                        full[gi] = resp
            else:
                for gi, resp in local:
                    full[gi] = resp

            if is_main:
                res, _ = score_fn(examples, full)
                if primary_key is not None:
                    prim = res.get(primary_key)
                else:  # rerank -> first mrr* metric
                    prim = next((v for k, v in res.items() if k.startswith("mrr")), None)
                summary[f"{task}_{label}"] = (float(prim) if prim is not None else None)
                summary["eval_size"][f"{task}_{label}"] = len(examples)
                print(f"[ladder:{task}@{label}] {primary_key or 'mrr'}="
                      f"{prim if prim is None else round(float(prim), 3)} "
                      f"(eval_size={len(examples)})", flush=True)
                flush()
            if world > 1:
                torch.distributed.barrier()

    flush()
    if is_main:
        print(f"\n[docchunk-ladder] WROTE {args.out} in {time.time()-t0:.1f}s", flush=True)
        keys = sorted(k for k in summary if any(k.startswith(t + "_") for t in ALL_TASKS))
        print(f"[docchunk-ladder] ladder keys present: {keys}", flush=True)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
