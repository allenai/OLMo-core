"""
Convert ``allenai/Dolci-Instruct-SFT`` into an OLMo-core SFT dataset tokenized with a Qwen
chat template (Qwen3 by default, Qwen3.5 via ``--tokenizer`` + the id overrides below).

Each row has a ``messages`` field: a list of ``{role, content}`` turns (``system``/``user``/
``assistant``, possibly multi-turn). We render the whole conversation once with the Qwen3 chat
template and derive a loss mask that is ``True`` on every assistant turn's content (and its
closing ``<|im_end|>``), ``False`` everywhere else (system/user turns and assistant turn headers).

Output format (matches what :class:`olmo_core.data.composable.PackingInstanceSourceConfig.from_npy`
reads for SFT, i.e. what the Qwen3-4B SFT scripts under ``src/scripts/train/sft`` consume):

  * ``token_ids_part_NNNNNN.npy``  -- raw (headerless) ``uint32`` token IDs, conversations
    concatenated and each terminated by the EOS id ``151643``.
  * ``labels_mask_NNNNNN.npy``     -- raw (headerless) ``bool``, parallel to the token file
    (same length), ``True`` only on assistant-turn tokens (loss is computed there).

Both files are written with ``ndarray.tofile`` (NOT ``np.save``) because the reader memmaps them
as raw arrays. The labels-mask file must have exactly the same number of items as its paired
token file; the loader pairs them by sorted filename position.

CRITICAL -- EOS alignment:
    ``TokenizerConfig.qwen3().eos_token_id`` is ``151643`` (``<|endoftext|>``), and OLMo-core finds
    document boundaries by splitting on that id. The Qwen3 *chat template* ends every turn with
    ``<|im_end|>`` = ``151645``, which is NOT a document boundary for OLMo-core. So we explicitly
    append ``<|endoftext|>`` (151643) after each conversation. Without this, OLMo-core would never
    find a boundary and would concatenate the whole dataset into a single document.

MASK DERIVATION:
    We render the whole conversation once (``add_generation_prompt=False``) and locate each
    turn's span in that single string by scanning, in order, for the literal
    ``<|im_start|>{role}\\n ... <|im_end|>`` markers the Qwen3 template emits per turn (message
    content is pre-sanitized so these control strings can't appear inside it). For assistant
    turns the span runs from just after the role header to just after the closing
    ``<|im_end|>``, so it also covers any empty ``<think>\\n\\n</think>\\n\\n`` boilerplate the
    template auto-inserts.

    NOTE: we deliberately do NOT derive spans by re-rendering message slices (e.g.
    ``messages[:i+1]``) the way the ``rlhn`` converter does for its single trailing assistant
    turn. For multi-turn conversations that trick breaks: the Qwen3 template only auto-inserts
    the empty think block for an assistant message that is the *actual last* message being
    rendered, so slicing to ``i+1`` makes a mid-conversation assistant turn look like the last
    one and injects a think block that isn't present in the true full render, breaking the
    prefix relationship. Scanning the full render for literal turn markers sidesteps this
    entirely. We then tokenize the *full* string once with ``return_offsets_mapping=True`` and
    mark every token whose start offset falls inside an assistant span.

TOOL-USE ROLES:
    Dolci-Instruct-SFT's Tool Use subset (~9% of rows) carries ``environment``/``tool`` turns that
    neither Qwen chat template renders. They are filtered out by role before templating, so the row
    is dropped whole rather than rendered with a hole in the dialogue. This reproduces what the
    Qwen3 build already did by accident -- that template drops the unknown turn, after which the
    turn-marker scan fails and the conversation is skipped -- and keeps the two builds comparable.
    Qwen3.5's template instead raises, which is why the filter is explicit rather than incidental.

OTHER TOKENIZERS:
    ``--tokenizer`` alone is not enough to retarget this script: the EOS separator and the
    reserved landmark id are tokenizer-specific and default to the Qwen3 values. Override both
    when converting for another vocabulary. For Qwen3.5 (``TokenizerConfig.qwen3_5()``, vocab
    248320) that is ``--eos-token-id 248044 --landmark-token-id 248200``, matching the ids the
    Qwen3.5 landmark training scripts use. The turn-marker scan itself is vocabulary-independent
    -- it works on the rendered string -- but the script asserts the template really does emit
    ``<|im_start|>``/``<|im_end|>`` rather than silently producing an all-False mask.

Run (CPU, e.g. via gantry with the weka bucket mounted)::

    python src/scripts/data/convert_dolci_instruct_sft.py \\
        --out-dir /weka/oe-training-default/amandab/dolci-instruct-sft/qwen3

    python src/scripts/data/convert_dolci_instruct_sft.py \\
        --tokenizer Qwen/Qwen3.5-0.8B --eos-token-id 248044 --landmark-token-id 248200 \\
        --out-dir /weka/oe-training-default/amandab/dolci-instruct-sft/qwen35

Then point an SFT script's ``DATASET_PATH`` at ``--out-dir``.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("convert_dolci_instruct_sft")

DATASET = "allenai/Dolci-Instruct-SFT"
# Any Qwen3 tokenizer is identical for our purposes; matches olmo_core.data.TokenizerConfig.qwen3()
# (vocab 151936, eos/bos/pad 151643).
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"

# Defaults are the Qwen3 values; both are overridable per tokenizer (see module docstring).
DEFAULT_EOS_TOKEN_ID = 151643  # <|endoftext|> -- OLMo-core document separator
DEFAULT_LANDMARK_TOKEN_ID = 151860  # reserved id inserted later by LandmarkInstanceSource

# Set from the CLI in main() before any conversion work runs.
EOS_TOKEN_ID = DEFAULT_EOS_TOKEN_ID
LANDMARK_TOKEN_ID = DEFAULT_LANDMARK_TOKEN_ID

TOKEN_DTYPE = np.uint32
MASK_DTYPE = np.bool_

# Special-token strings stripped from message content so they can't be re-parsed as control tokens.
_SPECIAL_STRINGS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>")

# Roles the plain chat rendering handles. Dolci-Instruct-SFT's Tool Use subset (~9% of rows) also
# carries 'environment'/'tool' turns, and the two Qwen templates disagree about them: Qwen3 silently
# drops the turn (so the span scan then fails and the conversation is skipped), while Qwen3.5 raises
# a jinja2 TemplateError and takes the whole job down mid-run. Filtering here makes the drop explicit
# and identical across tokenizers, so the two builds stay comparable.
_SUPPORTED_ROLES = frozenset({"system", "user", "assistant"})


def sanitize(text: Optional[str]) -> str:
    text = text or ""
    for s in _SPECIAL_STRINGS:
        text = text.replace(s, " ")
    return text


def normalize_messages(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {"role": m["role"], "content": sanitize(m.get("content"))}
        for m in raw_messages
        if m.get("role") and m.get("content") is not None
    ]


def find_turn_spans(
    full_str: str, messages: List[Dict[str, str]]
) -> Optional[List[Tuple[int, int]]]:
    """
    Scan ``full_str`` in order for each message's ``<|im_start|>{role}\\n ... <|im_end|>`` marker
    pair and return the assistant-turn spans (start just after the role header, end just after
    the closing ``<|im_end|>``). Returns ``None`` if the markers aren't found in the expected
    order (unexpected template structure).
    """
    spans: List[Tuple[int, int]] = []
    pos = 0
    for m in messages:
        start_marker = f"<|im_start|>{m['role']}\n"
        marker_pos = full_str.find(start_marker, pos)
        if marker_pos == -1:
            return None
        content_start = marker_pos + len(start_marker)
        end_pos = full_str.find("<|im_end|>", content_start)
        if end_pos == -1:
            return None
        content_end = end_pos + len("<|im_end|>")
        if m["role"] == "assistant":
            spans.append((content_start, content_end))
        pos = content_end
    return spans


def tokenize_instance(
    tok, messages: List[Dict[str, str]]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Tokenize one multi-turn conversation and build its (token_ids, labels_mask) arrays."""
    if not any(m["role"] == "assistant" for m in messages):
        return None

    # Belt and braces: callers filter unsupported roles up front, but a template can reject a
    # conversation for other reasons too, and one bad row must not kill a multi-hour job.
    try:
        full_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception as exc:  # noqa: BLE001 -- any template failure degrades to a skipped row
        log.debug("chat template rejected a conversation (%s); skipping", exc)
        return None

    spans = find_turn_spans(full_str, messages)
    if not spans:
        return None

    if not tok.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset-based mask derivation.")

    enc = tok(full_str, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # The appended EOS is the OLMo-core document separator; it must not occur inside the body.
    if EOS_TOKEN_ID in ids:
        return None

    token_ids = np.asarray(list(ids) + [EOS_TOKEN_ID], dtype=TOKEN_DTYPE)
    if LANDMARK_TOKEN_ID in token_ids:
        return None

    starts = np.asarray([s for s, _e in offsets], dtype=np.int64)
    mask = np.zeros(token_ids.shape, dtype=MASK_DTYPE)
    for start, end in spans:
        mask[: len(offsets)] |= (starts >= start) & (starts < end)

    if not mask.any():
        return None
    return token_ids, mask


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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--eos-token-id",
        type=int,
        default=DEFAULT_EOS_TOKEN_ID,
        help="Document separator appended after every conversation. Must equal the training "
        "TokenizerConfig's eos_token_id (Qwen3: 151643, Qwen3.5: 248044).",
    )
    parser.add_argument(
        "--landmark-token-id",
        type=int,
        default=DEFAULT_LANDMARK_TOKEN_ID,
        help="Reserved id the landmark instance sources insert; conversations containing it are "
        "dropped (Qwen3: 151860, Qwen3.5: 248200).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="Skip conversations whose tokenized length exceeds this (0 = no limit; the packer's "
        "long_doc_strategy truncates long docs at pack time regardless).",
    )
    parser.add_argument(
        "--flush-tokens", type=int, default=100_000_000, help="Tokens per shard file."
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N rows (0 = all).")
    parser.add_argument(
        "--print-examples", type=int, default=2, help="Print the first N assembled instances."
    )
    args = parser.parse_args()

    global EOS_TOKEN_ID, LANDMARK_TOKEN_ID
    EOS_TOKEN_ID = args.eos_token_id
    LANDMARK_TOKEN_ID = args.landmark_token_id
    log.info(
        f"tokenizer={args.tokenizer} eos_token_id={EOS_TOKEN_ID} "
        f"landmark_token_id={LANDMARK_TOKEN_ID}"
    )

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

    # The mask derivation scans the rendered string for ChatML turn markers. If this tokenizer's
    # chat template emits something else, every span lookup would miss and *every* conversation
    # would be silently dropped as a template mismatch -- fail here instead.
    _probe = tok.apply_chat_template(
        [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if "<|im_start|>assistant\n" not in _probe or "<|im_end|>" not in _probe:
        raise RuntimeError(
            f"{args.tokenizer}'s chat template does not use the ChatML turn markers this "
            f"converter scans for. Rendered probe:\n{_probe!r}"
        )
    log.info("chat-template probe (ChatML markers present):\n%s", _probe)

    log.info(f"loading {DATASET} ...")
    ds = load_dataset(DATASET, split="train")
    n_rows = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    log.info(f"{n_rows:,} rows to process")

    writer = ShardWriter(args.out_dir, args.flush_tokens)
    n_written = 0
    n_skipped_no_assistant = 0
    n_skipped_unsupported_role = 0
    n_skipped_template = 0
    n_skipped_too_long = 0
    n_skipped_bad = 0

    try:
        from tqdm import tqdm

        row_iter = tqdm(range(n_rows), desc="convert")
    except ImportError:
        row_iter = range(n_rows)

    for i in row_iter:
        row = ds[i]
        raw_messages = row["messages"] or []
        messages = normalize_messages(raw_messages)
        if any(m["role"] not in _SUPPORTED_ROLES for m in messages):
            n_skipped_unsupported_role += 1
            continue
        if not any(m["role"] == "assistant" for m in messages):
            n_skipped_no_assistant += 1
            continue

        result = tokenize_instance(tok, messages)
        if result is None:
            n_skipped_template += 1
            continue

        token_ids, mask = result
        if args.max_seq_len > 0 and token_ids.size > args.max_seq_len:
            n_skipped_too_long += 1
            continue
        if not mask.any():
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
        "landmark_token_id": LANDMARK_TOKEN_ID,
        "dtype": "uint32",
        "mask_dtype": "bool",
        "max_seq_len": args.max_seq_len,
        "num_instances": n_written,
        "num_tokens": writer.total_tokens,
        "num_loss_tokens": writer.total_loss_tokens,
        "num_parts": writer.part,
        "skipped_no_assistant": n_skipped_no_assistant,
        "skipped_unsupported_role": n_skipped_unsupported_role,
        "skipped_template_mismatch": n_skipped_template,
        "skipped_too_long": n_skipped_too_long,
        "skipped_bad": n_skipped_bad,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("DONE: %s", json.dumps(meta))


if __name__ == "__main__":
    main()
