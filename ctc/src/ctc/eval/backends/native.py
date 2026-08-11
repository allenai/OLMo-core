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

from ..stopping import StopCondition
from ..stopping import apply as apply_stop
from ..stopping import should_stop, strip_think

__all__ = ["NativeBackend"]

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
        query_position: str = "after",
        mem_freq: int = 63,
        doc_start_id: Optional[int] = None,
        doc_end_id: Optional[int] = None,
    ):
        if attn not in ATTENTION_MODES:
            raise ValueError(f"attn must be one of {ATTENTION_MODES}, got {attn!r}")

        import torch  # noqa: F401 -- imported for its side effect on device init
        from olmo_core.config import DType
        from olmo_core.generate.generation_module.config import GenerationConfig
        from olmo_core.generate.generation_module.transformer import (
            TransformerGenerationModuleConfig,
        )
        from transformers import AutoTokenizer

        self.ckpt = Path(ckpt)
        self.attn = attn
        self.max_length = max_length
        self.device = device
        # Needed by build_prefill: the segmenter rebuilds the prompt itself, so it must be told the
        # same layout ctc.format used, or the two disagree silently.
        self.query_position = query_position
        self.mem_freq = mem_freq
        self.doc_start_id = doc_start_id
        self.doc_end_id = doc_end_id

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

        self._configure_attention()

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
            model.enable_document_chunk_attention()
        elif self.attn == "full":
            # A checkpoint trained with the mask still carries it in config.json. Forcing plain
            # causal here is what makes the "full" arm mean full.
            disable = getattr(model, "disable_document_chunk_attention", None)
            if disable is not None:
                disable()

    def build_prefill(self, example: Mapping[str, Any], task: str) -> List[int]:
        """
        Tokenize one example the way the model was trained, not as flat text.

        **The token stream carries document markers even in the ``full`` arm.** ``full`` means the
        attention *mask* is plain causal; the tokens still wrap each document in
        ``<|box_start|>``/``<|box_end|>``, exactly as training did. Plain-tokenizing the prompt
        string instead produces a stream the model has never seen, which is worth roughly -0.01 f1
        on contradiction @2k -- small enough to be mistaken for numerical noise, which is what makes
        it dangerous.

        This is why the structure has to come from the example rather than from the assembled text:
        the segmenter needs document boundaries, and recovering them from a flat string means
        re-deriving what the generator already knew.

        :param example: A unified-format example.
        :param task: Task name.

        :returns: Prompt token ids, markers included.
        """
        from olmo_core.data.document_chunk_landmark import (
            DOC_END_ID,
            DOC_START_ID,
            LANDMARK_TOKEN_ID,
            PAD_TOKEN_ID,
            emit_document_chunk_dense,
            emit_document_chunk_landmark,
            segment_prompt_to_chunks,
        )

        segs, _, _ = segment_prompt_to_chunks(
            self.tok,
            example,
            task,
            query_position=self.query_position,
            cot_mode="none",
            chunk_by="document",
            item_regex=r"\|\|",
            include_answer=False,
            doc_start_id=self.doc_start_id if self.doc_start_id is not None else DOC_START_ID,
            doc_end_id=self.doc_end_id if self.doc_end_id is not None else DOC_END_ID,
        )
        if self.attn in ("full", "chunked"):
            # Both arms share one token stream; they differ only in the MASK, which
            # _configure_attention set at load time.
            ids, _ = emit_document_chunk_dense(segs)
        else:
            ids, _ = emit_document_chunk_landmark(
                segs, mem_freq=self.mem_freq, mem_id=LANDMARK_TOKEN_ID, pad_id=PAD_TOKEN_ID
            )
        return list(ids)

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
        :param examples: The unified-format examples the prompts came from. When given, each is
            tokenized through :meth:`build_prefill` so the token stream carries markers.
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

        out: List[str] = []
        for i, prompt in enumerate(prompts):
            if examples is not None:
                ids = self.build_prefill(examples[i], task)
            else:
                ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
            out.append(self._generate_one(torch, ids, stop))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"[ctc-eval] {i + 1}/{len(prompts)}", flush=True)
        return out

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
