"""
Native backend: grade an olmo-core checkpoint directly, with no export step.

This is the reference backend. It runs the same model code that produced the checkpoint, so when
the vLLM or HuggingFace path disagrees with it, this one is right by definition -- there is no
conversion between training and grading for a bug to hide in.

It is also the only backend that can grade the non-standard attention masks, because they exist
only in olmo-core. ``--attn chunked`` enables ``DocumentChunkedAttention`` on the loaded model;
``--attn full`` forces plain causal attention even on a checkpoint whose config requests the mask.

.. warning::
   **The word "dense" is a trap in the pre-migration code.** The evaluators call the chunked mask
   ``--variant dense`` (it is dense *within* a chunk), while the suite driver's ``--variant dense``
   means plain causal attention, with a one-line dict translating between them. Two layers, one
   word, opposite meanings. This module uses ``full`` / ``chunked`` / ``landmark`` throughout and
   never says "dense", which is the only reason the translation is not needed.

Requires ``pip install 'ctc[native]'`` -- torch, transformers and olmo-core itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..prefill import Prefill, build_prefills, plain_prefill, structural_prefill
from ..prefix_cache import generate_group_with_shared_prefix, group_by_corpus, measure_reuse
from ..stopping import StopCondition
from ..stopping import apply as apply_stop
from ..stopping import should_stop, strip_think

__all__ = ["NativeBackend", "build"]

ATTENTION_MODES = ("full", "chunked", "landmark")

#: Generation pad id used when the tokenizer's pad collides with eos, which every Qwen tokenizer
#: here does. Matches the pre-migration ``--pad-fallback-id`` default, so a reproduction run uses
#: the same value the original did. Any distinct reserved id would work; this one is
#: ``<|im_end|>`` in the Qwen3 vocabulary.
PAD_FALLBACK_ID = 151645

#: Tokens between host-side stop-rule checks during decode. Each check decodes the whole completion
#: so far, which is O(n) per call and quadratic over a generation -- checking every token was the
#: dominant cost of this loop. The interval cannot affect the result: :func:`ctc.eval.stopping.apply`
#: performs the authoritative truncation afterwards over the full string, so a late check only means
#: a few extra tokens were generated and discarded.
STOP_CHECK_INTERVAL = 8


class NativeBackend:
    """
    Greedy decoding against an olmo-core checkpoint.

    :param ckpt: Checkpoint directory (the one containing ``config.json`` and ``model_and_optim``,
        or a ``stepN`` subdirectory).
    :param tokenizer: HuggingFace tokenizer id or path.
    :param attn: One of :data:`ATTENTION_MODES`.
    :param max_length: Total sequence budget (prompt + generation).
    :param device: Torch device string.
    :param dtype: Model dtype.
    :param pad_token_id: Generation pad id. Qwen3 tokenizers set ``pad == eos``, which breaks
        left-padding, so it is set explicitly rather than read from the tokenizer.
    :param share_prefix: Prefill each ``corpus_id`` group's shared token prefix once and reuse its
        KV (:mod:`ctc.eval.prefix_cache`). **Off by default and not yet validated end to end here**
        -- wired so the path exists, but no scored run has gone through it in this repo. It is also
        inert for the current checkpoints: they train with ``query_position="both"``, which puts the
        per-query question before the corpus, so there is almost no shared token prefix to reuse.
    """

    def __init__(
        self,
        ckpt: Path,
        *,
        tokenizer: str = "Qwen/Qwen3-4B",
        attn: str = "full",
        max_length: int = 4096,
        device: str = "cuda",
        dtype: str = "bfloat16",
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        query_position: str = "both",
        mem_freq: int = 63,
        doc_start_id: Optional[int] = None,
        doc_end_id: Optional[int] = None,
        share_prefix: bool = False,
    ):
        if attn not in ATTENTION_MODES:
            raise ValueError(f"attn must be one of {ATTENTION_MODES}, got {attn!r}")

        import torch  # noqa: F401 -- imported for its side effect on device init
        from transformers import AutoTokenizer

        from olmo_core.config import DType
        from olmo_core.generate.generation_module.config import GenerationConfig
        from olmo_core.generate.generation_module.transformer import (
            TransformerGenerationModuleConfig,
        )

        self.ckpt = Path(ckpt)
        self.attn = attn
        self.max_length = max_length
        self.share_prefix = share_prefix
        self.device = device
        # Needed by build_prefill: the segmenter rebuilds the prompt itself, so it must be told the
        # same layout ctc.format used, or the two disagree silently.
        self.query_position = query_position
        self.mem_freq = mem_freq
        self.doc_start_id = doc_start_id
        self.doc_end_id = doc_end_id
        self._prefill: Optional[Prefill] = None
        self._prefill_task: Optional[str] = None

        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        self.eos_id = eos_token_id if eos_token_id is not None else self.tok.eos_token_id

        # Qwen tokenizers set pad == eos, and olmo_core's GenerationConfig rejects that outright:
        # left-padding with eos would make the model read padding as a real end-of-sequence. Any
        # DISTINCT reserved id works, and at batch size 1 no pad token ever enters the stream --
        # but the config still validates it, so a real one has to be supplied.
        pad = pad_token_id if pad_token_id is not None else self.tok.pad_token_id
        if pad is None or pad == self.eos_id:
            pad = PAD_FALLBACK_ID
        if pad == self.eos_id:
            raise ValueError(
                f"pad_token_id and eos_token_id are both {pad}; pass a distinct pad_token_id"
            )

        gen_cfg = GenerationConfig(
            pad_token_id=pad,
            eos_token_id=self.eos_id,
            max_length=max_length,
            do_sample=False,  # greedy: eval must be reproducible run to run
        )
        self.gm = TransformerGenerationModuleConfig(
            gen_cfg, float8_config=None, dtype=DType(dtype), compile_model=False
        ).build(checkpoint_dir=str(self.ckpt), device=device)

        self._check_tokenizer_matches_model()
        self._configure_attention()
        if self.share_prefix:
            # At load time, not at first generate: refusing after a rung has already been graded
            # wastes the run, and the answer depends only on the checkpoint.
            self._check_share_prefix_supported()

    def _check_tokenizer_matches_model(self) -> None:
        """
        Refuse a tokenizer whose vocabulary does not fit the checkpoint's.

        The two Qwen families in this project have different vocabularies -- Qwen3 is 151,936 and
        Qwen3.5 is 248,320 -- and ``--tokenizer`` defaults to a Qwen3 id. Point that at a Qwen3.5
        checkpoint and every token id is wrong, the EOS is wrong (151,645 against 248,044), and the
        run completes normally and reports **f1 = 0.000**, which reads as a dead model rather than
        as a mis-set flag. Measured on ``ctc-s5-contra-full-4b``: f1 0.000 at parse_rate 0.000.

        Checked against the embedding matrix rather than the config, because the embedding is what
        the ids actually index.

        :raises ValueError: If the tokenizer's vocabulary exceeds the model's, or the two differ
            enough that they cannot be the same family.
        """
        embeddings = getattr(self.gm.model, "embeddings", None)
        weight = getattr(embeddings, "weight", None)
        if weight is None:  # pragma: no cover - depends on the model class
            return
        model_vocab = int(weight.shape[0])
        tok_vocab = int(getattr(self.tok, "vocab_size", 0) or 0)
        if not tok_vocab:  # pragma: no cover - tokenizer without a declared size
            return

        # Padded vocabularies are normal (the model rounds up for kernel alignment), so a model
        # LARGER than the tokenizer by a little is fine. A tokenizer larger than the model, or a
        # gap far past any padding, means they are different families.
        if tok_vocab > model_vocab or model_vocab - tok_vocab > 4096:
            raise ValueError(
                f"tokenizer/checkpoint mismatch: the tokenizer has {tok_vocab:,} tokens but this "
                f"checkpoint's embedding is {model_vocab:,}. These are different tokenizers, so "
                f"every id -- including EOS -- would be wrong and the run would score ~0.0 rather "
                f"than fail.\n"
                f"  Qwen3   (vocab 151,936): --tokenizer Qwen/Qwen3-4B\n"
                f"  Qwen3.5 (vocab 248,320): --tokenizer Qwen/Qwen3.5-0.8B-Base "
                f"(one tokenizer across all Qwen3.5 sizes)"
            )

    def _recurrent_layers(self) -> int:
        """
        :returns: How many of this checkpoint's layers carry recurrent (GatedDeltaNet) state.
        """
        try:
            from olmo_core.nn.attention.recurrent import GatedDeltaNet
        except ImportError:  # a build without the recurrent mixer cannot have any
            return 0
        return sum(1 for m in self.gm.model.modules() if isinstance(m, GatedDeltaNet))

    def _check_share_prefix_supported(self) -> None:
        """
        Refuse ``--share-prefix`` on a checkpoint whose state a reused prefix cannot carry.

        Prefix reuse assumes a layer's whole history lives in a KV cache that can be snapshotted
        and restored. That is true of attention and false of GatedDeltaNet, whose history is a
        recurrent state plus a causal-conv window. :mod:`ctc.eval.prefix_cache` does snapshot and
        restore both, but ``GatedDeltaNet.forward`` only READS them on a single-token call
        (``use_precomputed = ... and T_og == 1``, ``recurrent.py``). A reused suffix is multi-token,
        so it takes the other branch, which overwrites the conv window from the suffix's own inputs
        and calls the chunk kernel with no ``initial_state`` -- the restored state is discarded
        before it is ever read, and those layers silently resume from zero.

        Measured on Qwen3.5-4B (24 of 32 layers GatedDeltaNet), contradiction 2k, 500 rows:
        **132 generations (26.4%) changed** and 69 got a different per-example f1. The aggregate
        barely moved (0.8085 -> 0.8034, paired SE 0.0054), which is exactly what makes this
        dangerous -- a quarter of the answers are resampled while the headline number looks fine.

        :raises SystemExit: If the checkpoint has recurrent layers.
        """
        n = self._recurrent_layers()
        if not n:
            return
        raise SystemExit(
            f"--share-prefix is not sound on this checkpoint: {n} of its layers are GatedDeltaNet, "
            "whose recurrent and conv state a reused prefix does not carry into a multi-token "
            "suffix (they resume from zero). Measured cost on a 24/32-GDN model: 26.4% of "
            "generations changed while the aggregate f1 moved only -0.005, so the damage does not "
            "show up in the headline number. Re-run without --share-prefix. Lifting this needs "
            "GatedDeltaNet.forward to honour a restored state on a multi-token call -- "
            "`initial_state` is already a supported argument of dispatch_chunk_gated_delta_rule, "
            "but CausalConv1d also needs a multi-token forward seeded from the cached window."
        )

    def _marker_family(self) -> str:
        """
        :returns: ``"qwen3"`` or ``"qwen3_5"``, decided by the checkpoint's embedding height.

        Read off the embedding rather than the config or the tokenizer argument, for the same
        reason :meth:`_check_tokenizer_matches_model` does: the embedding is what the ids actually
        index, and a checkpoint's ``config.json`` does not reliably carry
        ``document_chunk_attention`` (verified absent on the shipped 4B checkpoints).
        """
        embeddings = getattr(self.gm.model, "embeddings", None)
        weight = getattr(embeddings, "weight", None)
        rows = int(weight.shape[0]) if weight is not None else 0
        return "qwen3" if rows and rows <= 200_000 else "qwen3_5"

    def _document_chunk_ids(self) -> Dict[str, int]:
        """
        :returns: The ``doc_start_id`` / ``doc_end_id`` / ``eos_id`` the chunked mask needs.

        Explicit constructor arguments win; otherwise they come from the family's reserved set.
        The mask derives every token's chunk role from these three ids, so a wrong one does not
        raise -- it yields a mask that silently degrades toward plain causal.
        """
        from olmo_core.data.document_chunk_landmark import reserved_ids

        ids = reserved_ids(self._marker_family())
        return {
            "doc_start_id": self.doc_start_id if self.doc_start_id is not None else ids.doc_start,
            "doc_end_id": self.doc_end_id if self.doc_end_id is not None else ids.doc_end,
            "eos_id": ids.eos,
        }

    def _configure_attention(self) -> None:
        """
        Put the model into the requested attention mode.

        :raises RuntimeError: If a mode is requested that this checkpoint cannot provide. Silently
            falling back would grade one arm while reporting the other.
        """
        model = self.gm.model
        if self.attn == "chunked":
            if not hasattr(model, "enable_document_chunk_attention"):
                raise RuntimeError(
                    "this checkpoint's model has no enable_document_chunk_attention; it cannot be "
                    "graded with --attn chunked"
                )
            # The three ids are REQUIRED positional arguments. Calling this bare raised TypeError
            # at model load, which made `--attn chunked --backend native` -- the suite's primary
            # arm -- unreachable: every such run died before its first prompt.
            model.enable_document_chunk_attention(**self._document_chunk_ids())
        elif self.attn == "full":
            # A checkpoint trained with the mask still carries it in config.json. Forcing plain
            # causal here is what makes the "full" arm mean full.
            disable = getattr(model, "disable_document_chunk_attention", None)
            if disable is not None:
                disable()

    def prefill_for(self, task: str) -> Prefill:
        """
        The prefill builder for a task, constructed once and reused.

        Shared with the vLLM and HuggingFace backends via :mod:`ctc.eval.prefill`, which is what
        makes a cross-backend comparison a comparison of *decoding* rather than of two different
        token streams.

        The chunk layout comes from the task's own spec rather than from a flag. It has to match how
        the training shards were converted, and for one task in the suite it is not the default:
        oolong's items are lines inside a single context block. A caller who had to know that could
        get it wrong, and the failure is silent -- the model is simply graded against a token stream
        it never trained on.

        :param task: Task name.

        :returns: The builder.
        """
        if self._prefill is None or self._prefill_task != task:
            from ...format import registry

            extra = registry.get(task).extra if task in registry.names() else {}
            self._prefill = structural_prefill(
                self.tok,
                task=task,
                attn=self.attn,
                query_position=self.query_position,
                mem_freq=self.mem_freq,
                doc_start_id=self.doc_start_id,
                doc_end_id=self.doc_end_id,
                chunk_by=extra.get("chunk_by", "document"),
                item_regex=extra.get("item_regex", r"\|\|"),
            )
            self._prefill_task = task
        return self._prefill

    def count_tokens(self, text: str) -> int:
        """
        :param text: A prompt.

        :returns: Its length in tokens, for the runner's length audit.
        """
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def generate(
        self,
        prompts: Sequence[str],
        examples: Optional[Sequence[Mapping[str, Any]]] = None,
        *,
        stop: StopCondition,
        task: Optional[str] = None,
        progress_every: int = 25,
    ) -> List[str]:
        """
        Greedily decode a continuation for each prompt.

        Decodes one prompt at a time. Batching left-pads, and left-padding interacts with the
        chunked mask's chunk-id reconstruction, so the unbatched path is the one that is known
        correct; batching belongs in a later change with its own parity check.

        :param prompts: Prompts, already assembled by the task spec. Used only when ``examples``
            is absent -- plain tokenization omits the document markers the model was trained with,
            so it is a fallback, not the intended path.
        :param examples: The unified-format examples the prompts came from. When given, each goes
            through the shared structural prefill so the token stream carries markers.
        :param task: Task name, required alongside ``examples``.
        :param stop: The task's stop condition, checked after each token so generation ends at the
            answer rather than at the budget.
        :param progress_every: Print progress every N examples. Long rungs are slow enough that a
            silent run is indistinguishable from a hung one.

        :returns: One generation per prompt, already truncated and think-stripped.
        """
        import torch

        if examples is not None and task is None:
            raise ValueError("passing examples requires task=")

        prefill = self.prefill_for(task) if examples is not None else plain_prefill(self.tok)
        all_ids = build_prefills(prefill, prompts, examples)

        if self.share_prefix:
            return self._generate_with_shared_prefixes(torch, all_ids, examples, stop)

        out: List[str] = []
        for i, ids in enumerate(all_ids):
            out.append(self._generate_one(torch, ids, stop))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"[ctc-eval] {i + 1}/{len(all_ids)}", flush=True)
        return out

    def _generate_with_shared_prefixes(
        self,
        torch: Any,
        all_ids: Sequence[Sequence[int]],
        examples: Optional[Sequence[Mapping[str, Any]]],
        stop: StopCondition,
    ) -> List[str]:
        """
        Decode via :mod:`ctc.eval.prefix_cache`, prefilling each corpus group's shared prefix once.

        **Not yet validated end to end on a GPU** -- the module's rewind logic is tested and the
        wiring below is, but no scored run has gone through this path in this repo. Treat a number
        produced with ``share_prefix=True`` as unverified until a byte-identity run against the
        plain path exists.

        The reuse is measured and printed rather than assumed. Shared documents are not shared
        tokens: with ``query_position="both"`` the per-query question precedes the corpus, and the
        measured reusable prefix was 0.5-0.9% on the multiplexed tasks. When there is nothing to
        reuse this costs a grouping pass and falls back.

        :param torch: The torch module.
        :param all_ids: Tokenized prompts.
        :param examples: The examples they came from, read for ``corpus_id``.
        :param stop: The task's stop condition.

        :returns: One generation per prompt, in the order of ``all_ids``.
        """
        groups = group_by_corpus(examples or [{} for _ in all_ids])
        estimate = measure_reuse([[all_ids[i] for i in rows] for rows in groups.values()])
        print(f"[ctc-eval] prefill reuse: {estimate}", flush=True)

        out: List[Optional[str]] = [None] * len(all_ids)
        for rows in groups.values():
            texts, _ = generate_group_with_shared_prefix(
                self.gm,
                [all_ids[i] for i in rows],
                device=self.device,
                max_length=self.max_length,
                decode_fn=lambda logits: self._decode_from(torch, logits, stop),
            )
            for row, text in zip(rows, texts):
                out[row] = text
        return [t or "" for t in out]

    @staticmethod
    def _decode_step(torch: Any, logits) -> int:
        """:returns: The greedy next-token id."""
        return int(logits[0, -1].argmax().item())

    def _generate_one(self, torch: Any, prompt_ids: Sequence[int], stop: StopCondition) -> str:
        """
        Decode one prompt.

        :param torch: The torch module.
        :param prompt_ids: Tokenized prompt.
        :param stop: The task's stop condition.

        :returns: The generation, truncated and cleaned exactly as
            :func:`ctc.eval.stopping.apply` would -- so this path and a batched or remote one
            produce the same string, which is what makes cross-backend parity meaningful.
        """
        with torch.no_grad():
            self.gm.prepare_inference_cache(1, self.max_length)
            leftpad = torch.zeros(1, dtype=torch.int32, device=self.device)
            logits = self.gm.model(
                torch.tensor([list(prompt_ids)], device=self.device),
                logits_to_keep=1,
                cache_leftpad=leftpad,
            )
        return self._decode_from(torch, logits, stop)

    def _decode_from(self, torch: Any, logits: Any, stop: StopCondition) -> str:
        """
        Run the decode loop from a prefilled state.

        Split out from :meth:`_generate_one` so the prefill-reuse path can supply the logits from a
        shared prefix instead. Both paths therefore share every stop rule and truncation detail --
        the reuse path differs only in where the prompt's KV came from, which is the only way it
        can be argued to preserve scores.

        :param torch: The torch module.
        :param logits: Logits for the last prompt token.
        :param stop: The task's stop condition.

        :returns: The generation, truncated and cleaned.
        """
        with torch.no_grad():
            nxt = self._decode_step(torch, logits)
            produced: List[int] = []
            for step in range(stop.max_new_tokens):
                if stop.eos and nxt == self.eos_id:
                    break
                produced.append(nxt)

                # The stop rule is checked against decoded TEXT, not token ids -- "]]" and "\n" are
                # text facts and a tokenizer may split them anywhere. But decoding is host-side and
                # costs O(len(produced)) each time, so doing it every token is quadratic in the
                # generation length and dominated this loop.
                #
                # Checking on an interval cannot change the OUTPUT, only the work: the authoritative
                # truncation is the apply_stop below, which sees the whole string. A late check just
                # means a few extra tokens were generated and then discarded.
                if (step + 1) % STOP_CHECK_INTERVAL == 0:
                    text = self.tok.decode(produced, skip_special_tokens=True)
                    if stop.strip_think:
                        text = strip_think(text)
                    if should_stop(text, stop) is not None:
                        break

                logits = self.gm.model(torch.tensor([[nxt]], device=self.device), logits_to_keep=1)
                nxt = self._decode_step(torch, logits)

        return apply_stop(self.tok.decode(produced, skip_special_tokens=True), stop)


def build(ckpt: str, **kwargs: Any) -> NativeBackend:
    """
    Construct the native backend.

    Present so :func:`ctc.eval.backends.base.load` can build any backend by name, with the same
    call shape for all three. The constructor takes the arguments; this only adapts the path type.

    :param ckpt: Checkpoint directory.
    :param kwargs: See :class:`NativeBackend`.

    :returns: The backend.
    """
    return NativeBackend(Path(ckpt), **kwargs)
