"""
CTC task JSONL -> olmo-core SFT shards, as PLAIN COMPLETIONS (no chat template).

Why a second converter instead of ``convert_unified_to_sft.py``: that one targets instruction-tuned
Qwen checkpoints, so it wraps every instance in the Qwen3 chat template and hardcodes Qwen's EOS
(151643) and landmark id. The hybridish suite (Yashas' 275m-2.7b hybrid-attention models) are
**base** checkpoints with a different tokenizer and, in general, no chat template at all -- applying
one trains a wrapper the model has never seen and that the evaluator never emits.

The ctc-suite evaluator prompts a base model with the bare alpaca-style completion string
``spec.build_prompt(example)``. Verified byte-identical to
``corpus_reasoning_prompts.build_prompt(..., use_alpaca=True)``, which is what this script renders,
so training input and scoring input are the same string by construction. The gold answer comes from
the same builder, and ``--verify`` feeds it back through the *evaluator's own* parser and scorer:
if the target this script trains on does not score 1.0 under the grader that will judge the model,
the shards are wrong and the run is wasted.

Every id that is tokenizer-specific is read from ``--tokenizer`` rather than hardcoded.

Output (raw, headerless; matches ``NumpyPaddedFSLDataset``):

* ``token_ids_part_NNNNNN.npy``  -- uint32 ids, each instance EOS-terminated.
* ``labels_mask_NNNNNN.npy``     -- bool, True only on answer tokens.
* ``metadata.json``              -- counts, per-task and per-rung realized token-length stats.

Example::

    python src/scripts/data/hybridish/convert_ctc_to_sft_completion.py \\
        --input-jsonl /data/ctc_hybridish/short/qdmatch_nq/train.jsonl \\
        --task qdmatch --tokenizer allenai/dolma2-tokenizer \\
        --min-tokens 0 --max-seq-len 4096 --verify \\
        --out-dir /data/ctc_hybridish/shards_short/qdmatch_nq
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np

from olmo_core.data.corpus_reasoning_prompts import build_prompt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("convert_ctc_completion")

TOKEN_DTYPE = np.uint32
MASK_DTYPE = np.bool_

#: Ladder name -> the shared *spec* name, which is both the grading spec and the prompt-builder's
#: task vocabulary: nq and msmarco are both ``retrieval``, qdmatch_nq and qdmatch_hpqa both
#: ``qdmatch``. Passing a ladder name straight to ``build_prompt`` raises IndexError on every row
#: (it looks up a per-task renderer and finds none), which surfaces as a 100% build-error rate.
TASK_TO_SPEC = {
    "nq": "retrieval",
    "msmarco": "retrieval",
    "hotpotqa": "cot_retrieval",
    "fiqa": "retrieval",
    "scifact": "retrieval",
    "qdmatch_nq": "qdmatch",
    "qdmatch_hpqa": "qdmatch",
    "outlier_review": "outlier",
    "contra_fever": "contradiction",
    "grouping_labeled": "grouping_labeled",
}


def spec_name_for(task: str) -> str:
    """:returns: The shared spec/prompt name for a ladder name (identity when already a spec)."""
    return TASK_TO_SPEC.get(task, task)


def resolve_eos_id(tok, override: Optional[int]) -> int:
    """
    Pick the document-separator id from the tokenizer, never from a constant.

    :param tok: A loaded HF tokenizer.
    :param override: An explicit id, used verbatim when given.

    :returns: The EOS id to terminate every instance with.

    :raises SystemExit: If the tokenizer declares no EOS and none was passed.
    """
    if override is not None:
        return override
    eos = getattr(tok, "eos_token_id", None)
    if eos is None:
        raise SystemExit(
            "tokenizer declares no eos_token_id; pass --eos-token-id explicitly. Guessing an id "
            "here would silently split instances at the wrong place."
        )
    return int(eos)


def tokenize_completion(tok, prompt: str, answer: str, eos_id: int, train_on_eos: bool = True):
    """
    Tokenize one plain-completion instance.

    No chat template: the string is ``prompt + answer``, exactly what the evaluator prefills and
    then continues. The loss mask is derived from character offsets at the prompt/answer boundary,
    so it covers the answer and nothing else.

    :param tok: A fast HF tokenizer (offsets are required).
    :param prompt: The rendered alpaca prompt, ending at ``### Response:``.
    :param answer: The gold continuation.
    :param eos_id: Id appended to terminate the instance.
    :param train_on_eos: Include the terminating EOS in the loss, so the model learns to *stop*.
        On by default and it should stay on: with EOS masked out, a base model is never taught to
        end an answer, keeps generating past it, and the suite's graders score the ramble. That has
        already produced two artifacts in this project -- a no-CoT path that rambled until it was
        truncated at the first ``]]``, and a repetition loop that faked a 2-4x oolong collapse.
        The chat-template converter has the same behaviour by a different route: its mask covers
        the assistant turn *including* the closing ``<|im_end|>``.

    :returns: ``(token_ids, labels_mask)``, or ``None`` if the body already contains ``eos_id``
        (which would split the instance) or the answer tokenizes to nothing.
    """
    if not tok.is_fast:
        raise SystemExit("a fast tokenizer is required for offset-based mask derivation")
    text = prompt + answer
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    if eos_id in ids:
        return None
    boundary = len(prompt)
    mask = [bool(start >= boundary) for (start, _end) in enc["offset_mapping"]]
    if not any(mask):
        return None
    token_ids = np.asarray(ids + [eos_id], dtype=TOKEN_DTYPE)
    labels_mask = np.asarray(mask + [bool(train_on_eos)], dtype=MASK_DTYPE)
    return token_ids, labels_mask


def iter_examples(patterns: List[str], limit: int) -> Iterator[Tuple[dict, str, int]]:
    """Yield ``(example, source_file, row_index)`` over every JSONL path/glob, honouring ``--limit``.

    ``row_index`` counts from 0 across the concatenated inputs -- the order a consumer re-reading
    the same globs sees. It is what makes a dropped row recoverable: the shards are a subsequence
    of the source, and without the index nothing on disk says where the holes are.
    """
    paths: List[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched and os.path.exists(pattern):
            matched = [pattern]
        paths.extend(matched)
    if not paths:
        raise FileNotFoundError(f"no JSONL matched: {patterns}")
    log.info("reading %d JSONL file(s)", len(paths))
    n = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if "ex" in ex and "documents" not in ex:
                    ex = ex["ex"]
                yield ex, os.path.basename(path), n
                n += 1
                if limit and n >= limit:
                    return


class ShardWriter:
    """Buffers ``(token_ids, labels_mask)`` and flushes raw paired shard files."""

    def __init__(self, out_dir: str, flush_tokens: int):
        self.out_dir = out_dir
        self.flush_tokens = flush_tokens
        self.tok_buf: List[np.ndarray] = []
        self.mask_buf: List[np.ndarray] = []
        self.buffered = 0
        self.part = 0
        self.total_tokens = 0
        self.total_loss_tokens = 0

    def add(self, token_ids: np.ndarray, mask: np.ndarray) -> None:
        self.tok_buf.append(token_ids)
        self.mask_buf.append(mask)
        self.buffered += token_ids.size
        self.total_loss_tokens += int(mask.sum())
        if self.buffered >= self.flush_tokens:
            self.flush()

    def flush(self) -> None:
        if self.buffered == 0:
            return
        tokens = np.concatenate(self.tok_buf)
        masks = np.concatenate(self.mask_buf)
        assert tokens.size == masks.size
        tok_path = os.path.join(self.out_dir, f"token_ids_part_{self.part:06d}.npy")
        mask_path = os.path.join(self.out_dir, f"labels_mask_{self.part:06d}.npy")
        tokens.tofile(tok_path + ".tmp")
        masks.tofile(mask_path + ".tmp")
        os.replace(tok_path + ".tmp", tok_path)
        os.replace(mask_path + ".tmp", mask_path)
        self.total_tokens += int(tokens.size)
        log.info("wrote part %06d: %s tokens; total %s", self.part, f"{tokens.size:,}", f"{self.total_tokens:,}")
        self.part += 1
        self.tok_buf = []
        self.mask_buf = []
        self.buffered = 0


def make_verifier(task: str):
    """
    Build a callable that scores a gold answer under the evaluator's own parser and scorer.

    The point is not to check the data -- ``ctc-data build`` already audits that -- but to check
    *this script's target rendering* against the grader that will judge the trained model. A target
    the grader cannot parse trains the model to write unparseable answers, and the resulting score
    looks like a capability failure.

    :param task: The ladder/task name.

    :returns: ``(answer, example) -> primary metric`` , or ``None`` if the ``ctc`` package is not
        importable (``--verify`` then fails loudly rather than silently passing).
    """
    try:
        from ctc.format import registry
        from ctc.tasks import load_all
    except ImportError:
        return None
    load_all()
    spec = registry.get(spec_name_for(task))

    def verify(answer: str, example: dict) -> Optional[float]:
        n_docs = len(example.get("documents", []) or [])
        parsed = spec.parse(answer, n_docs)
        if parsed is None:
            return None
        gold = spec.gold(example) if hasattr(spec, "gold") else None
        if gold is None:
            # Skip keys that are PRESENT BUT EMPTY. oolong carries `gold_doc_indices: []` (its gold
            # is the answer string, not a document id), so a plain `key in example` check picks the
            # empty list, hands the scorer no gold at all, and the task reads as unverifiable.
            for key in ("gold_pairs", "gold_doc_indices", "gold_order", "answers"):
                value = example.get(key)
                if value:
                    gold = value
                    break
        scores = spec.score(parsed, gold)
        return float(scores.get(spec.primary_metric, next(iter(scores.values()))))

    return verify


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-jsonl", nargs="+", required=True, help="task JSONL path(s)/glob(s)")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--task",
        default=None,
        help="task for every row (single-task file). A per-row '_task' overrides it.",
    )
    p.add_argument("--cot-mode", default="none", help="cot_mode; per-row '_cot_mode' overrides")
    p.add_argument(
        "--query-position",
        default="both",
        choices=("before", "after", "both"),
        help="MUST match the eval setting; the ctc-suite evaluator uses 'both'",
    )
    p.add_argument("--tokenizer", required=True, help="HF tokenizer id or path")
    p.add_argument(
        "--eos-token-id",
        type=int,
        default=None,
        help="override the instance separator (default: the tokenizer's eos_token_id)",
    )
    p.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="drop instances shorter than this. The lower edge of a length band -- rung LABELS are "
        "not token counts (contradiction runs ~1.5x under its label, niah ~2.9x), so a band has to "
        "be enforced on measured length",
    )
    p.add_argument("--max-seq-len", type=int, default=4096, help="drop instances longer than this")
    p.add_argument(
        "--verify",
        action="store_true",
        help="score every gold target under the evaluator's parser+scorer and abort if any "
        "fall below --verify-min (requires the `ctc` package on PYTHONPATH)",
    )
    p.add_argument("--verify-min", type=float, default=1.0)
    p.add_argument("--verify-sample", type=int, default=200, help="0 = verify every row")
    p.add_argument(
        "--no-train-on-eos",
        action="store_true",
        help="exclude the terminating EOS from the loss. Almost always wrong: the model then never "
        "learns to stop and the graders score the ramble (see tokenize_completion)",
    )
    p.add_argument("--flush-tokens", type=int, default=100_000_000)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--print-examples", type=int, default=1)
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = resolve_eos_id(tok, args.eos_token_id)
    log.info("tokenizer=%s vocab=%d eos_id=%d", args.tokenizer, len(tok), eos_id)
    assert len(tok) <= np.iinfo(TOKEN_DTYPE).max

    verifiers: dict = {}
    if args.verify:
        probe = make_verifier(args.task or "nq")
        if probe is None:
            raise SystemExit(
                "--verify needs the `ctc` package importable (PYTHONPATH=<ctc>/src). Running "
                "without it would skip the one check that catches an unparseable target."
            )

    writer = ShardWriter(args.out_dir, args.flush_tokens)
    n_written = n_short = n_long = n_bad = n_build_err = 0
    by_task: dict = {}
    lengths: List[int] = []
    verify_scores: List[float] = []
    verify_fail: List[dict] = []
    build_err_by_task: dict = {}
    # Source row behind each written instance, and the rows that never made it. Graders need the
    # first to pair a generation with its OWN gold; the second is the audit trail for the drops.
    src_index: List[int] = []
    dropped_rows: List[int] = []

    for ex, src, row in iter_examples(args.input_jsonl, args.limit):
        task = ex.get("_task", args.task)
        cot_mode = ex.get("_cot_mode", args.cot_mode)
        if task is None:
            raise SystemExit("row has no '_task' and --task was not given")
        try:
            prompt, answer = build_prompt(
                ex,
                task=spec_name_for(task),
                query_position=args.query_position,
                use_alpaca=True,
                cot_mode=cot_mode,
            )
        except (KeyError, ValueError, TypeError, IndexError) as e:
            n_build_err += 1
            dropped_rows.append(row)
            build_err_by_task[task] = build_err_by_task.get(task, 0) + 1
            if build_err_by_task[task] <= 3:
                log.warning("build_prompt failed for _task=%s (%s: %s)", task, type(e).__name__, e)
            continue

        if args.verify and (not args.verify_sample or len(verify_scores) < args.verify_sample):
            if task not in verifiers:
                verifiers[task] = make_verifier(task)
            score = verifiers[task](answer, ex) if verifiers[task] else None
            if score is None or score < args.verify_min:
                verify_fail.append({"task": task, "src": src, "answer": answer[:200], "score": score})
            else:
                verify_scores.append(score)

        result = tokenize_completion(tok, prompt, answer, eos_id, not args.no_train_on_eos)
        if result is None:
            n_bad += 1
            dropped_rows.append(row)
            continue
        token_ids, mask = result
        if token_ids.size < args.min_tokens:
            n_short += 1
            dropped_rows.append(row)
            continue
        if token_ids.size > args.max_seq_len:
            n_long += 1
            dropped_rows.append(row)
            continue

        if n_written < args.print_examples:
            log.info(
                "EXAMPLE 0 [%s/%s] %d tokens, %d loss:\n%s\n<<<ANSWER>>>%s",
                task, cot_mode, token_ids.size, int(mask.sum()), prompt[:600], answer[:200],
            )
        writer.add(token_ids, mask)
        src_index.append(row)
        lengths.append(int(token_ids.size))
        by_task[task] = by_task.get(task, 0) + 1
        n_written += 1
        if n_written % 500 == 0:
            log.info("%s instances written", f"{n_written:,}")

    writer.flush()

    if args.verify and verify_fail:
        for row in verify_fail[:5]:
            log.error("VERIFY FAIL %s", json.dumps(row))
        raise SystemExit(
            f"{len(verify_fail)} gold target(s) scored below {args.verify_min} under the "
            f"evaluator's own parser/scorer. The shards would train unparseable answers."
        )

    arr = np.asarray(lengths) if lengths else np.zeros(1)
    meta = {
        "input_jsonl": args.input_jsonl,
        "task": args.task,
        "query_position": args.query_position,
        "tokenizer": args.tokenizer,
        "eos_token_id": eos_id,
        "chat_template": False,
        "train_on_eos": not args.no_train_on_eos,
        "prompt_style": "alpaca_completion",
        "dtype": "uint32",
        "mask_dtype": "bool",
        "min_tokens": args.min_tokens,
        "max_seq_len": args.max_seq_len,
        "num_instances": n_written,
        "num_instances_by_task": by_task,
        "num_tokens": writer.total_tokens,
        "num_loss_tokens": writer.total_loss_tokens,
        "num_parts": writer.part,
        "token_len": {
            "min": int(arr.min()), "p50": int(np.percentile(arr, 50)),
            "p90": int(np.percentile(arr, 90)), "max": int(arr.max()),
            "mean": float(arr.mean()),
        },
        "skipped_below_min_tokens": n_short,
        "skipped_above_max_seq_len": n_long,
        "skipped_bad": n_bad,
        "skipped_build_error": n_build_err,
        "skipped_build_by_task": build_err_by_task,
        "src_index_file": "src_index.json",
        "verified": len(verify_scores),
        "verify_mean_score": float(np.mean(verify_scores)) if verify_scores else None,
    }
    with open(os.path.join(args.out_dir, "src_index.json"), "w") as f:
        json.dump(
            {
                "note": "shard instance i was built from source row src_index[i]",
                "input_jsonl": args.input_jsonl,
                "num_instances": len(src_index),
                "src_index": src_index,
                "dropped": dropped_rows,
            },
            f,
        )
    assert len(src_index) == n_written, "src_index must have one entry per written instance"
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("DONE: %s", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
