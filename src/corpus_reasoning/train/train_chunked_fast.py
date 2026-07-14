"""Training with chunked document attention.

Chunked attention isolates documents from each other during training: each
document can only attend to itself and to "free" tokens (instruction, query,
answer), not to other documents.

Backends:
  - "flex" (default): Block-sparse FlexAttention via a compiled BlockMask. Fast,
    uses FlashAttention-class kernels, does not materialize the O(N^2) mask.
  - "sdpa": Legacy path — materializes a dense 4D float mask and calls SDPA.
    Kept for debugging / correctness comparison. Much slower.

The `standard_attention: true` config option forces plain causal attention for
ablations (no doc boundary tokens, no chunked mask).

Usage:
    accelerate launch --num_processes 4 scripts/train/train_chunked_fast.py configs/foo.yml
    python scripts/train/train_chunked_fast.py configs/foo.yml --attn sdpa    # legacy
"""

import argparse
import os
import random
import yaml
import torch
from pathlib import Path
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

import json as _json

from corpus_reasoning.lib.io import load_jsonl
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.lib.chunked_attention import (
    DOC_START, DOC_END, setup_tokenizer, wrap_documents,
    find_chunk_spans,
    PAD_CHUNK_ID, FREE_CHUNK_ID,
    AttentionPattern, build_flex_mask_mod, build_is_anchor,
    build_random_doc_edges, build_random_token_keep,
    install_flex_chunked_attention,
)


# FlexAttention block alignment — pad seq_len to a multiple of this so the
# compiled block mask has uniform tiling.
FLEX_BLOCK = 128


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChunkedDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, doc_start_id, doc_end_id,
                 query_position="after", train_on_inputs=False, task="retrieval",
                 use_titles=True, before_dummy=0, after_dummy=0,
                 unified_prompt=False, cot_mode="label",
                 standard_mix_prob=0.0, mix_seed=42):
        self.examples = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.doc_start_id = doc_start_id
        self.doc_end_id = doc_end_id
        self.query_position = query_position
        self.train_on_inputs = train_on_inputs
        self.task = task
        self.use_titles = use_titles
        self.before_dummy = before_dummy
        self.after_dummy = after_dummy
        self.unified_prompt = unified_prompt
        self.cot_mode = cot_mode
        # Per-example mix: with prob standard_mix_prob, skip wrap_documents()
        # so the example has no doc-boundary tokens. Under the "chunked"
        # mask_mod, all-FREE tokens collapse to plain causal — i.e. that
        # example behaves like standard attention while the rest stay chunked.
        self.standard_mix_prob = standard_mix_prob
        self.mix_seed = mix_seed
        self.response_marker = tokenizer.encode("### Response:\n", add_special_tokens=False)

    def __len__(self):
        return len(self.examples)

    def _find_response_start(self, input_ids):
        ids = input_ids.tolist()
        marker = self.response_marker
        for i in range(len(ids) - len(marker) + 1):
            if ids[i:i + len(marker)] == marker:
                return i + len(marker)
        return len(ids)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt, output = build_prompt(
            ex, task=self.task, query_position=self.query_position,
            use_titles=self.use_titles, before_dummy=self.before_dummy,
            after_dummy=self.after_dummy, use_alpaca=True,
            unified_prompt=self.unified_prompt, cot_mode=self.cot_mode,
        )

        if self.standard_mix_prob > 0.0:
            # Deterministic per-idx so the chunked/standard split is
            # reproducible and stable across epochs.
            rng = random.Random(self.mix_seed + idx)
            present_as_standard = rng.random() < self.standard_mix_prob
        else:
            present_as_standard = False

        if not present_as_standard:
            prompt = wrap_documents(prompt)
        full_text = prompt + output + self.tokenizer.eos_token

        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len,
            return_tensors="pt", padding=False,
        )
        input_ids = encoding.input_ids.squeeze(0)

        labels = input_ids.clone()
        if not self.train_on_inputs:
            response_start = self._find_response_start(input_ids)
            labels[:response_start] = -100

        expected_num_docs = len(ex.get("documents", []))

        return {
            "input_ids": input_ids,
            "labels": labels,
            "expected_num_docs": expected_num_docs,
        }


# ---------------------------------------------------------------------------
# Chunk ID construction
# ---------------------------------------------------------------------------

def build_chunk_ids(input_ids, doc_start_id, doc_end_id):
    """Return int32 chunk ID per token. Doc i -> i; free tokens -> -1."""
    seq_len = len(input_ids)
    spans = find_chunk_spans(input_ids, doc_start_id, doc_end_id)
    chunk_id = torch.full((seq_len,), FREE_CHUNK_ID, dtype=torch.int32)
    for idx, (s, e) in enumerate(spans):
        chunk_id[s:e] = idx
    return chunk_id


def count_rectangular_chunks(chunk_id_row):
    """Count contiguous runs of non-negative chunk IDs.

    This equals the number of rectangular document blocks along the diagonal
    of the attention mask: each doc span produces one K_i × K_i block.
    """
    count = 0
    prev = None
    for v in chunk_id_row.tolist():
        if v >= 0 and v != prev:
            count += 1
        prev = v
    return count


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------

def _round_up(x, m):
    return ((x + m - 1) // m) * m


class FlexChunkedCollator:
    """Collator that emits per-token chunk IDs (+ optional anchor/random-edge
    tensors) for FlexAttention.

    The actual BlockMask is built in `FlexTrainer._prepare_inputs` — the
    collator only produces cheap CPU tensors.
    """

    def __init__(self, doc_start_id, doc_end_id, pad_token_id, pattern: AttentionPattern):
        self.doc_start_id = doc_start_id
        self.doc_end_id = doc_end_id
        self.pad_token_id = pad_token_id
        self.pattern = pattern

    def __call__(self, features):
        raw_lens = [f["input_ids"].size(0) for f in features]
        max_len = _round_up(max(raw_lens), FLEX_BLOCK)

        batch_ids, batch_labels, batch_chunks = [], [], []
        batch_anchor = [] if self.pattern.needs_anchor_tensor() else None
        # For bigbird random edges we also collect per-example doc counts to
        # decide the max_docs padding after the loop.
        per_ex_num_docs = [] if self.pattern.needs_random_edges() else None
        per_ex_idx = []

        for ex_idx, (f, orig_len) in enumerate(zip(features, raw_lens)):
            ids = f["input_ids"]
            labs = f["labels"]
            pad_len = max_len - orig_len

            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)])

            if self.pattern.name == "standard":
                # No chunking — treat every non-padding token as "free" so the
                # mask_mod collapses to plain causal + padding-aware.
                chunk_id = torch.full((max_len,), FREE_CHUNK_ID, dtype=torch.int32)
                chunk_id[orig_len:] = PAD_CHUNK_ID
                num_docs = 0
            else:
                chunk_id_orig = build_chunk_ids(f["input_ids"], self.doc_start_id, self.doc_end_id)
                # Structural check: number of rectangular doc blocks equals the
                # number of documents in the source example. Catches tokenizer
                # drift, missing boundary tokens, truncation mid-doc, etc.
                n_rect = count_rectangular_chunks(chunk_id_orig)
                n_unique = int((chunk_id_orig[chunk_id_orig >= 0].unique()).numel())
                expected = f["expected_num_docs"]
                assert n_rect == n_unique, (
                    f"Non-contiguous chunk IDs: {n_rect} runs vs {n_unique} unique. "
                    f"Doc spans are not forming clean rectangular blocks."
                )
                assert n_rect <= expected, (
                    f"Found {n_rect} doc chunks but example only has {expected} documents."
                )

                chunk_id = torch.full((max_len,), PAD_CHUNK_ID, dtype=torch.int32)
                chunk_id[:orig_len] = chunk_id_orig
                num_docs = n_unique

            batch_ids.append(ids)
            batch_labels.append(labs)
            batch_chunks.append(chunk_id)
            per_ex_idx.append(ex_idx)

            if batch_anchor is not None:
                # is_anchor marks doc_end positions; doc_end_id is None for
                # "standard" mode, but standard doesn't use anchors anyway.
                anchor = torch.zeros((max_len,), dtype=torch.bool)
                if self.doc_end_id is not None:
                    anchor[:orig_len] = (f["input_ids"] == self.doc_end_id)
                batch_anchor.append(anchor)

            if per_ex_num_docs is not None:
                per_ex_num_docs.append(num_docs)

        out = {
            "input_ids": torch.stack(batch_ids),
            "labels": torch.stack(batch_labels),
            "chunk_ids": torch.stack(batch_chunks),
        }
        if batch_anchor is not None:
            out["is_anchor"] = torch.stack(batch_anchor)
        if per_ex_num_docs is not None:
            max_docs = max(per_ex_num_docs) if per_ex_num_docs else 0
            # Need at least 1 to avoid empty tensor.
            max_docs = max(max_docs, 1)
            doc_random = torch.zeros((len(features), max_docs, max_docs), dtype=torch.bool)
            for b, nd in enumerate(per_ex_num_docs):
                adj = build_random_doc_edges(
                    num_docs=nd,
                    num_edges=self.pattern.num_random_doc_edges,
                    seed=self.pattern.random_seed + b,
                    max_docs=max_docs,
                )
                doc_random[b] = adj
            out["doc_random"] = doc_random
        if self.pattern.needs_random_token_mask():
            random_keep = torch.zeros(
                (len(features), max_len, max_len), dtype=torch.bool,
            )
            for b, f in enumerate(features):
                # Per-example seed derived from input_ids content so each
                # unique training example sees its own frozen mask. Otherwise
                # micro_batch_size=1 would give every example the same seed
                # (= base + 0), collapsing to a single shared mask.
                content_hash = int(f["input_ids"].sum().item())
                random_keep[b] = build_random_token_keep(
                    seq_len=max_len,
                    keep_prob=self.pattern.keep_prob,
                    seed=self.pattern.random_seed + content_hash,
                )
            out["random_keep"] = random_keep
        return out


class SdpaChunkedCollator:
    """Legacy dense 4D mask collator (kept for `--attn sdpa` comparison).

    Builds masks via chunked_attention.build_dense_bool_mask for every pattern
    so the SDPA and Flex paths stay in lockstep.
    """

    def __init__(self, doc_start_id, doc_end_id, pad_token_id, pattern: AttentionPattern):
        self.doc_start_id = doc_start_id
        self.doc_end_id = doc_end_id
        self.pad_token_id = pad_token_id
        self.pattern = pattern

    def __call__(self, features):
        from corpus_reasoning.lib.chunked_attention import build_dense_bool_mask

        max_len = max(f["input_ids"].size(0) for f in features)
        dtype = torch.bfloat16
        min_val = torch.finfo(dtype).min

        batch_ids, batch_labels, batch_masks = [], [], []
        for b, f in enumerate(features):
            ids = f["input_ids"]
            labs = f["labels"]
            orig_len = ids.size(0)
            pad_len = max_len - orig_len

            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)])

            if self.pattern.name == "standard":
                chunk_id = torch.full((max_len,), FREE_CHUNK_ID, dtype=torch.int32)
                chunk_id[orig_len:] = PAD_CHUNK_ID
            else:
                chunk_id_orig = build_chunk_ids(f["input_ids"], self.doc_start_id, self.doc_end_id)
                chunk_id = torch.full((max_len,), PAD_CHUNK_ID, dtype=torch.int32)
                chunk_id[:orig_len] = chunk_id_orig

            kwargs = {}
            if self.pattern.needs_anchor_tensor():
                anchor = torch.zeros((max_len,), dtype=torch.bool)
                if self.doc_end_id is not None:
                    anchor[:orig_len] = (f["input_ids"] == self.doc_end_id)
                kwargs["is_anchor"] = anchor.unsqueeze(0)
            if self.pattern.needs_random_edges():
                n_docs = int((chunk_id >= 0).unique().numel()) if self.pattern.name != "standard" else 0
                max_docs = max(n_docs, 1)
                adj = build_random_doc_edges(
                    num_docs=n_docs,
                    num_edges=self.pattern.num_random_doc_edges,
                    seed=self.pattern.random_seed + b,
                    max_docs=max_docs,
                )
                kwargs["doc_random"] = adj.unsqueeze(0)
            if self.pattern.needs_random_token_mask():
                rk = build_random_token_keep(
                    seq_len=max_len,
                    keep_prob=self.pattern.keep_prob,
                    seed=self.pattern.random_seed + b,
                )
                kwargs["random_keep"] = rk.unsqueeze(0)

            bool_mask = build_dense_bool_mask(self.pattern, chunk_id.unsqueeze(0), **kwargs)[0]
            mask = torch.full((max_len, max_len), min_val, dtype=dtype)
            mask = torch.where(bool_mask, torch.tensor(0.0, dtype=dtype), mask)

            batch_ids.append(ids)
            batch_labels.append(labs)
            batch_masks.append(mask.unsqueeze(0))

        return {
            "input_ids": torch.stack(batch_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_masks),
        }


# ---------------------------------------------------------------------------
# FlexAttention BlockMask construction
# ---------------------------------------------------------------------------

def build_block_mask(pattern, chunk_ids, is_anchor=None, doc_random=None,
                     random_keep=None, compile_mask=True):
    """Build a FlexAttention BlockMask for the given attention pattern."""
    from torch.nn.attention.flex_attention import create_block_mask

    B, S = chunk_ids.shape
    mask_mod = build_flex_mask_mod(
        pattern, chunk_ids, is_anchor=is_anchor, doc_random=doc_random,
        random_keep=random_keep,
    )
    return create_block_mask(
        mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S,
        device=chunk_ids.device, _compile=compile_mask,
    )


def verify_mask_structure(pattern, chunk_ids, is_anchor=None, doc_random=None,
                          random_keep=None):
    """First-batch sanity check: materialize the pattern's mask_mod on a
    meshgrid and confirm the structural invariants.

    Always-checked invariants (every pattern):
      - Padding rows/cols are fully False.
      - FREE (Q/A) tokens attend to all non-pad causal KV and are attended
        to by all non-pad causal Q.
      - The compiled FlexAttention BlockMask is constructable.

    Chunked-specific invariant (only when pattern.name == "chunked"):
      - Each doc-block [s, e) is a clean causal rectangle; no cross-doc edges.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    B, S = chunk_ids.shape
    device = chunk_ids.device

    mask_mod = build_flex_mask_mod(
        pattern, chunk_ids, is_anchor=is_anchor, doc_random=doc_random,
        random_keep=random_keep,
    )

    q = torch.arange(S, device=device)
    kv = torch.arange(S, device=device)
    Q, KV = torch.meshgrid(q, kv, indexing="ij")

    for b in range(B):
        b_t = torch.full_like(Q, b)
        h_t = torch.zeros_like(Q)
        dense = mask_mod(b_t, h_t, Q, KV)  # (S, S) bool

        row_chunks = chunk_ids[b]

        # (Universal) padding rows/cols fully masked.
        pad_rows = (row_chunks == PAD_CHUNK_ID)
        assert not dense[pad_rows, :].any().item(), (
            f"Example {b}: padding row is attending."
        )
        assert not dense[:, pad_rows].any().item(), (
            f"Example {b}: padding col is being attended to."
        )

        # (Universal) Q/A invariant — free rows see all non-pad causal kv, and
        # free cols are seen by all non-pad causal q.
        free_mask = (row_chunks == FREE_CHUNK_ID)
        non_pad_mask = (row_chunks != PAD_CHUNK_ID)
        causal_allowed = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
        expected_free_rows = causal_allowed & non_pad_mask.unsqueeze(0)  # (S, S)
        actual_free_rows = dense[free_mask]
        expected_slice = expected_free_rows[free_mask]
        assert (actual_free_rows == expected_slice).all().item(), (
            f"Example {b}: FREE row does not see all non-pad causal KV."
        )
        expected_free_cols = causal_allowed & non_pad_mask.unsqueeze(1)
        actual_free_cols = dense[:, free_mask]
        expected_slice_cols = expected_free_cols[:, free_mask]
        assert (actual_free_cols == expected_slice_cols).all().item(), (
            f"Example {b}: FREE col is not seen by all non-pad causal Q."
        )

        if pattern.name == "chunked":
            qc = row_chunks.unsqueeze(1)
            kc = row_chunks.unsqueeze(0)
            cross_doc = (qc >= 0) & (kc >= 0) & (qc != kc)
            assert not dense[cross_doc].any().item(), (
                f"Example {b}: cross-document attention leaked in chunked pattern."
            )
            # Each doc block is a clean causal rectangle.
            spans = []
            start = None
            prev = None
            for i, v in enumerate(row_chunks.tolist()):
                if v >= 0 and v != prev:
                    if start is not None:
                        spans.append((start, i))
                    start = i
                elif v < 0 and start is not None:
                    spans.append((start, i))
                    start = None
                prev = v
            if start is not None:
                spans.append((start, S))
            for (s, e) in spans:
                block = dense[s:e, s:e]
                tri = torch.tril(torch.ones_like(block))
                assert (block == tri.bool()).all().item(), (
                    f"Example {b}: doc block [{s},{e}) is not a clean causal rectangle."
                )

    # Try to actually build the compiled BlockMask on this device.
    _ = create_block_mask(mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=device)
    print(
        f"[mask check] ok — pattern={pattern.tag()} B={B} S={S} "
        f"chunks/example={[count_rectangular_chunks(chunk_ids[b]) for b in range(B)]}"
    )


# ---------------------------------------------------------------------------
# Trainer subclass
# ---------------------------------------------------------------------------

class FlexTrainer(Trainer):
    """Trainer that converts `chunk_ids` (+ optional is_anchor / doc_random)
    into a FlexAttention BlockMask per batch, using the configured pattern.

    Two delivery modes for the BlockMask:

    - Default (`mask_via_state=False`): put the BlockMask in
      `inputs["attention_mask"]`. HF's flex_attention path picks it up. Used
      for architectures HF natively registers for FlexAttention (Llama etc.).

    - State-dict (`mask_via_state=True`): stash the BlockMask in
      `mask_state["block_mask"]`. The trainer keeps `attention_mask` out of
      `inputs` entirely, so the model's mask preparation does nothing. The
      mask reaches the softmax layers via the custom `flex_chunked` attn
      function installed by `install_flex_chunked_attention`. Used for
      hybrid architectures like Qwen3.5 whose linear-attn layers can't
      accept a BlockMask through the normal channel.
    """

    _mask_verified = False

    def __init__(self, *args, attention_pattern: AttentionPattern = None,
                 mask_via_state: bool = False, mask_state: dict = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_pattern = attention_pattern or AttentionPattern()
        self.mask_via_state = mask_via_state
        self.mask_state = mask_state if mask_state is not None else {}

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        chunk_ids = inputs.pop("chunk_ids", None)
        is_anchor = inputs.pop("is_anchor", None)
        doc_random = inputs.pop("doc_random", None)
        random_keep = inputs.pop("random_keep", None)
        if chunk_ids is not None:
            if not FlexTrainer._mask_verified:
                verify_mask_structure(
                    self.attention_pattern, chunk_ids,
                    is_anchor=is_anchor, doc_random=doc_random,
                    random_keep=random_keep,
                )
                FlexTrainer._mask_verified = True
            block_mask = build_block_mask(
                self.attention_pattern, chunk_ids,
                is_anchor=is_anchor, doc_random=doc_random,
                random_keep=random_keep,
            )
            if self.mask_via_state:
                self.mask_state["block_mask"] = block_mask
                # Drop any stale attention_mask the dataloader might have
                # produced — model.forward must see attention_mask=None so
                # Qwen3.5's linear-attn layers stay on their no-mask path.
                inputs.pop("attention_mask", None)
            else:
                inputs["attention_mask"] = block_mask
        return inputs


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class LogFileCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 4), **logs}
            with open(self.log_path, "a") as f:
                f.write(_json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Fast chunked attention training")
    parser.add_argument("config", help="YAML config file (Axolotl format)")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Use torch.compile on the model")
    parser.add_argument("--attn", choices=["flex", "sdpa"], default=None,
                        help="Attention backend. Defaults to cfg['attn_backend'] "
                             "or 'flex' if unset. For 'flex': architectures HF "
                             "registers for FlexAttention (e.g. Llama) use HF's "
                             "native flex path. Architectures it doesn't "
                             "register (e.g. Qwen3.5) automatically fall back "
                             "to a custom `flex_chunked` attn function that "
                             "delivers the BlockMask via a state dict — same "
                             "FlexAttention kernel, just plumbed differently. "
                             "Pass --attn sdpa to force the legacy dense-mask "
                             "path.")
    parser.add_argument("--disable-flex-chunked", action="store_true",
                        default=None,
                        help="Escape hatch for the new flex_chunked path. When "
                             "set (or cfg['disable_flex_chunked']: true), if the "
                             "architecture refuses native flex_attention, fall "
                             "back to the legacy SDPA dense-mask path instead of "
                             "installing the custom flex_chunked impl. Use this "
                             "if you suspect the new path has a bug and want to "
                             "revert without flipping --attn on every config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.attn is None:
        # Default flipped from "sdpa" -> "flex" once the flex_chunked path
        # was validated on Qwen3.5 (~4x speedup, identical losses to 3
        # decimals). Configs that need the old SDPA dense-mask path
        # explicitly should set `attn_backend: sdpa` or pass --attn sdpa.
        args.attn = cfg.get("attn_backend", "flex")
    # CLI takes precedence over config when set; otherwise read from cfg.
    if args.disable_flex_chunked is None:
        args.disable_flex_chunked = bool(cfg.get("disable_flex_chunked", False))
    data_path = cfg["datasets"][0]["path"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attention_pattern = AttentionPattern.from_config(cfg)
    standard_attention = (attention_pattern.name == "standard")

    if standard_attention:
        doc_start_id, doc_end_id = None, None
    else:
        doc_start_id, doc_end_id = setup_tokenizer(tokenizer)

    # Try the requested backend first. For args.attn=="flex", some
    # architectures (notably hybrid Qwen3.5) refuse `flex_attention`
    # because HF hasn't registered it. By default we transparently fall
    # back to loading with sdpa and installing the custom `flex_chunked`
    # impl below — the block-sparse FlexAttention kernel is still used,
    # only the mask delivery channel differs.
    #
    # If --disable-flex-chunked (or cfg.disable_flex_chunked) is set,
    # we instead fall back to the *legacy* SDPA dense-mask path (which
    # is what `--attn sdpa` does directly). Useful as an escape hatch
    # if the flex_chunked impl turns out to misbehave.
    flex_via_state = False
    if args.attn == "flex":
        try:
            print(f"Loading {cfg['base_model']} with attn_implementation=flex_attention")
            model = AutoModelForCausalLM.from_pretrained(
                cfg["base_model"],
                attn_implementation="flex_attention",
                torch_dtype=torch.bfloat16,
            )
        except ValueError as e:
            if "flex_attention" not in str(e):
                raise
            if args.disable_flex_chunked:
                print("  -> flex_attention unsupported for this architecture; "
                      "--disable-flex-chunked is set, falling back to the "
                      "legacy SDPA dense-mask path (--attn flex effectively "
                      "becomes --attn sdpa).")
                model = AutoModelForCausalLM.from_pretrained(
                    cfg["base_model"],
                    attn_implementation="sdpa",
                    torch_dtype=torch.bfloat16,
                )
                # Switch the rest of the script onto the SDPA collator/trainer.
                args.attn = "sdpa"
            else:
                print("  -> flex_attention unsupported for this architecture; "
                      "falling back to sdpa+flex_chunked (state-dict BlockMask "
                      "delivery; same FlexAttention kernel). Pass "
                      "--disable-flex-chunked to revert to the legacy SDPA "
                      "dense-mask path.")
                model = AutoModelForCausalLM.from_pretrained(
                    cfg["base_model"],
                    attn_implementation="sdpa",
                    torch_dtype=torch.bfloat16,
                )
                flex_via_state = True
    else:
        print(f"Loading {cfg['base_model']} with attn_implementation=sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )

    # State dict shared between the trainer (writes block_mask each step)
    # and the custom attn function (reads it). Empty/unused when the model
    # supports HF's native flex_attention path.
    flex_state = {}
    if flex_via_state:
        install_flex_chunked_attention(model, flex_state)

    if not standard_attention:
        from corpus_reasoning.lib.olmo3_mask_patch import maybe_patch_olmo3_sliding_to_full
        maybe_patch_olmo3_sliding_to_full(model)
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            mean_emb = model.get_input_embeddings().weight[:-2].mean(dim=0)
            model.get_input_embeddings().weight[-2] = mean_emb
            model.get_input_embeddings().weight[-1] = mean_emb

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    if cfg.get("adapter") == "lora":
        lora_config = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        if cfg.get("freeze_embed", False):
            for name, param in model.named_parameters():
                if "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad = False
                    print(f"  Frozen: {name} ({param.numel():,} params)")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full fine-tuning: {trainable_params:,} / {total_params:,} parameters ({trainable_params/total_params:.1%})")

    query_position = cfg.get("query_position", "after")
    train_on_inputs = cfg.get("train_on_inputs", False)
    task = cfg.get("task", "retrieval")
    use_titles = not cfg.get("no_titles", False)
    before_dummy = cfg.get("before_dummy", 0)
    after_dummy = cfg.get("after_dummy", 0)
    standard_mix_prob = float(cfg.get("standard_mix_prob", 0.0))
    if standard_mix_prob > 0.0 and standard_attention:
        raise ValueError(
            "standard_mix_prob requires a non-standard attention_pattern "
            "(typically 'chunked'); set standard_attention: false."
        )
    mix_seed = int(cfg.get("mix_seed", cfg.get("random_seed", 42)))
    dataset = ChunkedDataset(
        data_path, tokenizer, cfg.get("sequence_len", 8192),
        doc_start_id, doc_end_id, query_position=query_position,
        train_on_inputs=train_on_inputs, task=task, use_titles=use_titles,
        before_dummy=before_dummy, after_dummy=after_dummy,
        unified_prompt=cfg.get("unified_prompt", False),
        cot_mode=cfg.get("cot_mode", "label"),
        standard_mix_prob=standard_mix_prob, mix_seed=mix_seed,
    )
    print(f"Loaded {len(dataset)} training examples from {data_path}")
    if standard_mix_prob > 0.0:
        n_std = sum(
            1 for i in range(len(dataset))
            if random.Random(mix_seed + i).random() < standard_mix_prob
        )
        print(
            f"  Mixed attention: {n_std}/{len(dataset)} examples "
            f"({n_std/len(dataset):.1%}) presented as standard "
            f"(unwrapped); rest use chunked. mix_seed={mix_seed}"
        )

    if args.attn == "flex":
        collator = FlexChunkedCollator(
            doc_start_id, doc_end_id, tokenizer.pad_token_id, attention_pattern,
        )
        trainer_cls = FlexTrainer
    else:
        collator = SdpaChunkedCollator(
            doc_start_id, doc_end_id, tokenizer.pad_token_id, attention_pattern,
        )
        trainer_cls = Trainer

    world_size = max(torch.cuda.device_count(), 1)
    batch_size = cfg.get("micro_batch_size", 1)
    grad_acc = cfg.get("gradient_accumulation_steps", 8)
    total_steps = len(dataset) // (batch_size * grad_acc * world_size)
    saves = cfg.get("saves_per_epoch", 1)
    save_steps = max(total_steps // saves, 1) if saves > 0 else total_steps

    output_dir = cfg.get("output_dir", "./outputs/chunked-lora")
    data_stem = Path(data_path).stem
    lr = float(cfg.get("learning_rate", 5e-4))
    epochs = cfg.get("num_epochs", 1)
    lora_r = cfg.get("lora_r", 16)
    seq_len = cfg.get("sequence_len", 8192)
    qpos = query_position[:1]
    attn_tag = f"{attention_pattern.tag()}-{args.attn}"
    if standard_mix_prob > 0.0:
        attn_tag = f"mix-std{int(round(standard_mix_prob * 100))}-{attn_tag}"
    adapter_tag = f"r{lora_r}" if cfg.get("adapter") == "lora" else "fullft"
    run_name = (
        cfg.get("wandb_name")
        or f"fast_{attn_tag}_{data_stem}_q{qpos}_lr{lr}_ep{epochs}_{adapter_tag}_seq{seq_len}_ga{grad_acc}"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=epochs,
        max_steps=int(cfg.get("max_steps", -1)),
        learning_rate=lr,
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        bf16=True,
        logging_steps=cfg.get("logging_steps", 1),
        logging_dir=f"{output_dir}/logs",
        save_steps=save_steps,
        save_total_limit=cfg.get("save_total_limit", 2),
        save_only_model=bool(cfg.get("save_only_model", False)),
        save_strategy=cfg.get("save_strategy", "steps"),
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        deepspeed=cfg.get("deepspeed"),
        report_to=["wandb"] if cfg.get("wandb_project") else [],
        run_name=run_name,
        tf32=True,
    )

    if cfg.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = cfg["wandb_project"]

    log_file = f"{output_dir}/train_log.jsonl"
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[LogFileCallback(log_file)],
    )
    if trainer_cls is FlexTrainer:
        trainer_kwargs["attention_pattern"] = attention_pattern
        trainer_kwargs["mask_via_state"] = flex_via_state
        trainer_kwargs["mask_state"] = flex_state
    trainer = trainer_cls(**trainer_kwargs)

    backend_label = args.attn
    if flex_via_state:
        backend_label = "flex (custom flex_chunked / state-dict BlockMask)"
    print(f"\nStarting training: {run_name}")
    print(f"  Attention pattern: {attention_pattern.tag()} ({attention_pattern})")
    print(f"  Attention backend: {backend_label}")
    print(f"  Gradient checkpointing: {cfg.get('gradient_checkpointing', True)}")
    print(f"  torch.compile: {args.compile}")
    print(f"  Monitor: tail -f {log_file}")
    print(f"  Total examples: {len(dataset)}, steps/epoch: {total_steps}")
    print(f"  Batch: {batch_size} × {grad_acc} grad_accum × {world_size} GPUs = {batch_size * grad_acc * world_size} effective")

    # Optional: force the SDPA backend to mem-efficient (+ cudnn when
    # available), excluding the math backend. The math backend materializes
    # the full B*H*S*S attention-scores tensor, which OOMs around S>~70k on
    # 144 GB H200s with our 4D float chunked mask. Mem-efficient (xFormers)
    # supports broadcastable float masks and runs in O(S) memory.
    # Off by default (existing runs unchanged); opt in with
    # `force_efficient_sdpa: true` in the YAML.
    if cfg.get("force_efficient_sdpa", False):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        backends = [SDPBackend.EFFICIENT_ATTENTION]
        if hasattr(SDPBackend, "CUDNN_ATTENTION"):
            backends.append(SDPBackend.CUDNN_ATTENTION)
        print(f"  force_efficient_sdpa=true: restricting SDPA backends to {[b.name for b in backends]}")
        with sdpa_kernel(backends):
            trainer.train()
    else:
        trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
