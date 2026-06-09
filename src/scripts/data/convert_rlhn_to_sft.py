"""
Convert the ``rlhn/rlhn-100K`` retrieval dataset into an OLMo-core SFT dataset for the
"list the relevant document IDs" task, tokenized with the Qwen3 chat template.

Each source row has a ``query`` plus ``positive_passages`` and ``negative_passages`` (each a
list of ``{docid, title, text}``). We turn each row into a single-turn chat instance::

    <user>
    {doc_id_1}: {doc 1 text}

    {doc_id_2}: {doc 2 text}
    ...
    For the query below, list the IDs of all documents that contain the information necessary
    to answer the query.
    Query: {query}
    Relevant document IDs:
    <assistant>
    {doc_id_a}, {doc_id_b}        # the IDs of the positive passages, in display order

Document IDs are random short alphanumeric tags (e.g. ``doc_a8f3``) assigned per-instance and
shuffled in with the negatives, so the model must bind ID<->content rather than memorize positions.

Output format (matches what :class:`olmo_core.data.composable.NumpyDocumentSource` /
``NumpyPackedFSLDataset`` read for SFT, i.e. what the Qwen3-4B landmark SFT script consumes):

  * ``token_ids_part_NNNNNN.npy``  -- raw (headerless) ``uint32`` token IDs, documents
    concatenated and each terminated by the EOS id ``151643``.
  * ``labels_mask_NNNNNN.npy``     -- raw (headerless) ``bool``, parallel to the token file
    (same length), ``True`` only on the assistant-response tokens (loss is computed there).

Both files are written with ``ndarray.tofile`` (NOT ``np.save``) because the reader memmaps them
as raw arrays. The labels-mask file must have exactly the same number of items as its paired
token file; the loader pairs them by sorted filename position.

CRITICAL -- EOS alignment:
    ``TokenizerConfig.qwen3().eos_token_id`` is ``151643`` (``<|endoftext|>``), and OLMo-core finds
    document boundaries by splitting on that id. The Qwen3 *chat template* ends every turn with
    ``<|im_end|>`` = ``151645``, which is NOT a document boundary for OLMo-core. So we explicitly
    append ``<|endoftext|>`` (151643) after each conversation. Without this, OLMo-core would never
    find a boundary and would concatenate the whole dataset into a single document.

Run (CPU, e.g. via gantry with the weka bucket mounted)::

    python src/scripts/data/convert_rlhn_to_sft.py \\
        --out-dir /weka/oe-training-default/ai2-llm/checkpoints/$USER/rlhn_sft_qwen \\
        --max-seq-len 8192

Then point the landmark SFT script's ``DATASET_PATH`` at ``--out-dir``.
"""

import argparse
import json
import logging
import os
import random
from typing import List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("convert_rlhn")

DATASET = "rlhn/rlhn-100K"
# Any Qwen3 tokenizer is identical for our purposes; matches olmo_core.data.TokenizerConfig.qwen3()
# (vocab 151936, eos/bos/pad 151643).
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"

EOS_TOKEN_ID = 151643  # <|endoftext|> -- OLMo-core document separator (see module docstring)
IM_END_ID = 151645  # <|im_end|> -- Qwen3 chat turn terminator (informational)
LANDMARK_TOKEN_ID = 151860  # reserved id inserted later by LandmarkInstanceSource; must NOT appear

TOKEN_DTYPE = np.uint32
MASK_DTYPE = np.bool_

INSTRUCTION = (
    "For the query below, list the IDs of all documents that contain the information "
    "necessary to answer the query."
)

# Special-token strings stripped from passage text so they can't be re-parsed as control tokens.
_SPECIAL_STRINGS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>")
_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def sanitize(text: Optional[str]) -> str:
    text = text or ""
    for s in _SPECIAL_STRINGS:
        text = text.replace(s, " ")
    return text.strip()


def render_doc(docid: str, title: str, text: str, max_chars: int) -> str:
    body = f"{title}\n{text}" if title else text
    if max_chars > 0 and len(body) > max_chars:
        body = body[:max_chars]
    return f"{docid}: {body}"


def assign_ids(n: int, id_length: int, rng: random.Random) -> List[str]:
    ids: set = set()
    out: List[str] = []
    while len(out) < n:
        cand = "doc_" + "".join(rng.choice(_ID_ALPHABET) for _ in range(id_length))
        if cand in ids:
            continue
        ids.add(cand)
        out.append(cand)
    return out


def render_chat(tok, messages: List[dict], add_generation_prompt: bool) -> str:
    """Render a chat to a string with the Qwen3 template (``enable_thinking`` left at its default)."""
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def apply_template(tok, messages: List[dict], add_generation_prompt: bool) -> List[int]:
    """Tokenize a chat with the Qwen3 template (token-count estimates only; see :func:`select_docs`)."""
    return tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )


def build_user_content(doc_blocks: List[str], query: str) -> str:
    doc_section = "".join(f"{block}\n\n" for block in doc_blocks)
    return f"{doc_section}{INSTRUCTION}\nQuery: {query}\nRelevant document IDs:"


def tokenize_instance(
    tok, doc_blocks: List[str], query: str, answer: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Tokenize one instance and build its (token_ids, labels_mask) arrays, or ``None`` if the body
    contains the EOS id.

    The loss mask is derived from **character offsets** rather than token-prefix matching: we render
    the prompt (the conversation up to the assistant header) and the full conversation as strings,
    confirm the prompt is a string prefix, then tokenize the full string once with
    ``return_offsets_mapping`` and mark every token that *starts* at or after the prompt boundary.
    This is robust to the Qwen3 template injecting a ``<think>\\n\\n</think>\\n\\n`` block before the
    final assistant turn (which makes the prompt/full token sequences non-prefix even though the
    strings are prefix-consistent). Loss therefore covers the whole assistant turn after the header:
    the (empty) think block, the answer, and the closing ``<|im_end|>``.
    """
    user_content = build_user_content(doc_blocks, query)
    messages = [{"role": "user", "content": user_content}]
    prompt_str = render_chat(tok, messages, add_generation_prompt=True)
    full_str = render_chat(
        tok, messages + [{"role": "assistant", "content": answer}], add_generation_prompt=False
    )

    if not full_str.startswith(prompt_str):
        # The user turn must render identically with/without the trailing assistant turn; if not,
        # the chat template is doing something unexpected and the mask can't be derived.
        raise RuntimeError(
            "Rendered prompt is not a string prefix of the full conversation; cannot derive the "
            "loss mask. Check the tokenizer's chat template."
        )

    if not tok.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset-based mask derivation.")

    enc = tok(full_str, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # The appended EOS is the OLMo-core document separator; it must not occur inside the body.
    if EOS_TOKEN_ID in ids:
        return None

    boundary = len(prompt_str)
    token_ids = np.asarray(list(ids) + [EOS_TOKEN_ID], dtype=TOKEN_DTYPE)
    if LANDMARK_TOKEN_ID in token_ids:
        return None
    # Loss on response tokens (start char >= boundary); prompt and trailing separator are masked.
    mask = np.zeros(token_ids.shape, dtype=MASK_DTYPE)
    for i, (start, _end) in enumerate(offsets):
        if start >= boundary:
            mask[i] = True
    return token_ids, mask


def select_docs(
    tok,
    positives: List[str],
    negatives: List[str],
    answer_ids: List[str],
    query: str,
    budget: int,
    rng: random.Random,
) -> Optional[List[str]]:
    """
    Greedily choose document blocks that fit within ``budget`` total tokens. All positives are
    kept; negatives are sampled in until adding another would exceed the budget. Returns the
    chosen blocks in shuffled display order, or ``None`` if the positives alone don't fit.

    A fast additive estimate picks the negative count; the caller then does an exact tokenization
    and trims if the estimate was optimistic, so the final instance never exceeds ``budget``.
    """
    # Per-block token estimates (independent tokenization; cross-block merges only ever reduce the
    # true count, so the sum is a safe over-estimate). One batched call.
    all_blocks = positives + negatives
    block_lens = [len(ids) for ids in tok([b + "\n\n" for b in all_blocks], add_special_tokens=False)["input_ids"]]
    pos_lens = block_lens[: len(positives)]
    neg_lens = block_lens[len(positives) :]

    # Fixed frame cost: template + instruction + query + answer + appended EOS, with no documents.
    frame_ids = apply_template(
        tok,
        [
            {"role": "user", "content": build_user_content([], query)},
            {"role": "assistant", "content": ", ".join(answer_ids)},
        ],
        add_generation_prompt=False,
    )
    frame_len = len(frame_ids) + 1  # +1 for the appended EOS separator

    running = frame_len + sum(pos_lens)
    if running > budget:
        return None  # positives + frame already over budget -> skip this example

    chosen = list(positives)
    neg_order = list(range(len(negatives)))
    rng.shuffle(neg_order)
    for j in neg_order:
        if running + neg_lens[j] > budget:
            continue  # skip this one, a smaller later negative may still fit
        chosen.append(negatives[j])
        running += neg_lens[j]

    rng.shuffle(chosen)
    return chosen


class ShardWriter:
    """Buffers (token_ids, labels_mask) arrays and flushes raw paired shard files."""

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
        log.info(
            f"wrote part {self.part:06d}: {tokens.size:,} tokens "
            f"({int(masks.sum()):,} with loss); total {self.total_tokens:,}"
        )
        self.part += 1
        self.tok_buf = []
        self.mask_buf = []
        self.buffered = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Total token budget per instance (template + docs + query + answer + EOS). Never "
        "exceeded. Must be <= the packer sequence length used in training so instances aren't "
        "truncated (truncation would drop the answer at the end).",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=2000,
        help="Truncate each passage's (title+text) to this many characters (0 = no limit).",
    )
    parser.add_argument("--id-length", type=int, default=4, help="Length of the random doc-ID suffix.")
    parser.add_argument("--flush-tokens", type=int, default=100_000_000, help="Tokens per shard file.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N rows (0 = all).")
    parser.add_argument("--print-examples", type=int, default=2, help="Print the first N assembled instances.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tok.vocab_size <= np.iinfo(TOKEN_DTYPE).max
    if getattr(tok, "eos_token_id", None) not in (None, EOS_TOKEN_ID):
        log.warning(
            f"Tokenizer eos_token_id={tok.eos_token_id} != expected OLMo-core separator "
            f"{EOS_TOKEN_ID}; the appended separator is hard-coded to {EOS_TOKEN_ID}."
        )

    log.info(f"loading {DATASET} ...")
    ds = load_dataset(DATASET, split="train")
    n_rows = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    log.info(f"{n_rows:,} rows to process; budget={args.max_seq_len} tokens/instance")

    writer = ShardWriter(args.out_dir, args.flush_tokens)
    n_written = 0
    n_skipped_no_pos = 0
    n_skipped_too_long = 0
    n_skipped_bad = 0

    try:
        from tqdm import tqdm

        row_iter = tqdm(range(n_rows), desc="convert")
    except ImportError:
        row_iter = range(n_rows)

    for i in row_iter:
        row = ds[i]
        positives = row["positive_passages"] or []
        negatives = row["negative_passages"] or []
        if not positives:
            n_skipped_no_pos += 1
            continue

        rng = random.Random(args.seed + i)
        query = sanitize(row["query"])

        pos_ids = assign_ids(len(positives), args.id_length, rng)
        neg_ids = assign_ids(len(negatives), args.id_length, rng)
        # Re-roll negative ids until globally unique within the instance (rare).
        while set(neg_ids) & set(pos_ids):
            neg_ids = assign_ids(len(negatives), args.id_length, rng)

        pos_blocks = [
            render_doc(pid, sanitize(p.get("title")), sanitize(p.get("text")), args.max_doc_chars)
            for pid, p in zip(pos_ids, positives)
        ]
        neg_blocks = [
            render_doc(nid, sanitize(p.get("title")), sanitize(p.get("text")), args.max_doc_chars)
            for nid, p in zip(neg_ids, negatives)
        ]
        # Map a rendered block back to its (id) so we can build the answer after shuffling.
        block_to_id = {block: pid for block, pid in zip(pos_blocks, pos_ids)}
        positive_block_set = set(pos_blocks)

        chosen = select_docs(
            tok, pos_blocks, neg_blocks, pos_ids, query, args.max_seq_len, rng
        )
        if chosen is None:
            n_skipped_too_long += 1
            continue

        # Answer = positive IDs in display (chosen) order.
        answer = ", ".join(block_to_id[b] for b in chosen if b in positive_block_set)

        result = tokenize_instance(tok, chosen, query, answer)
        # Trim trailing negatives if the exact tokenization overshot the budget.
        while result is not None and result[0].size > args.max_seq_len:
            # drop the last non-positive block in display order
            drop_idx = next(
                (k for k in range(len(chosen) - 1, -1, -1) if chosen[k] not in positive_block_set),
                None,
            )
            if drop_idx is None:
                result = None  # only positives left and still too long
                break
            chosen.pop(drop_idx)
            answer = ", ".join(block_to_id[b] for b in chosen if b in positive_block_set)
            result = tokenize_instance(tok, chosen, query, answer)

        if result is None:
            n_skipped_too_long += 1
            continue
        token_ids, mask = result
        if token_ids.size > args.max_seq_len or not mask.any():
            n_skipped_bad += 1
            continue

        if n_written < args.print_examples:
            log.info(
                "EXAMPLE %d (%d tokens, %d with loss):\n%s",
                n_written,
                token_ids.size,
                int(mask.sum()),
                tok.decode(token_ids.tolist()),
            )

        writer.add(token_ids, mask)
        n_written += 1

    writer.flush()

    meta = {
        "dataset": DATASET,
        "tokenizer": args.tokenizer,
        "eos_token_id": EOS_TOKEN_ID,
        "dtype": "uint32",
        "mask_dtype": "bool",
        "max_seq_len": args.max_seq_len,
        "max_doc_chars": args.max_doc_chars,
        "id_length": args.id_length,
        "seed": args.seed,
        "num_instances": n_written,
        "num_tokens": writer.total_tokens,
        "num_loss_tokens": writer.total_loss_tokens,
        "num_parts": writer.part,
        "skipped_no_positives": n_skipped_no_pos,
        "skipped_too_long": n_skipped_too_long,
        "skipped_bad": n_skipped_bad,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("DONE: %s", json.dumps(meta))


if __name__ == "__main__":
    main()
