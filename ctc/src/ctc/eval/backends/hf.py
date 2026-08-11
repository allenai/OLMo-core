"""
HuggingFace backend: widest model coverage, slowest.

Its real job is to be the *third opinion*. When vLLM and the native path disagree, the question is
which one is wrong, and a plain ``transformers`` forward pass -- no paged KV cache, no custom
kernels, no export step of its own -- is the cheapest way to break the tie. It is not the backend
to grade a full suite with.

Like the vLLM backend it is fed token ids from :mod:`ctc.eval.prefill` rather than prompt strings,
and like it, its stop handling is an early-exit optimisation over which
:func:`ctc.eval.stopping.apply` has the final word. Both properties exist so the three backends can
be compared at all.

Requires ``pip install 'ctc[hf]'``.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from ..prefill import Prefill, build_prefills, plain_prefill, structural_prefill
from ..stopping import StopCondition
from ..stopping import apply as apply_stop

__all__ = ["HFBackend", "build"]


class HFBackend:
    """
    Greedy decoding through ``transformers.generate``.

    :param model: A loaded causal LM, already on its device and in eval mode.
    :param tokenizer: The matching tokenizer.
    :param prefill: How to turn an example into token ids.
    :param device: Torch device string.
    :param eos_token_id: Override the tokenizer's eos.
    :param batch_size: Prompts per ``generate`` call. Defaults to 1 because batching left-pads,
        and left-padding interacts with the chunked mask's chunk-id reconstruction -- the same
        reason the native backend decodes one at a time.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        prefill: Optional[Prefill] = None,
        device: str = "cuda",
        eos_token_id: Optional[int] = None,
        batch_size: int = 1,
    ):
        self.model = model
        self.tok = tokenizer
        self.prefill = prefill or plain_prefill(tokenizer)
        self.device = device
        self.eos_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
        if batch_size != 1:
            raise ValueError(
                "batch_size > 1 is not supported yet. Batching left-pads, which the chunked mask's "
                "chunk-id reconstruction is sensitive to; it needs its own parity check before "
                "being switched on, not a flag."
            )
        self.batch_size = batch_size

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
        :param prompts: Rendered prompts, used only to build the prefill.
        :param examples: The examples they came from, required by a structural prefill.
        :param stop: The task's stop condition.
        :param task: Accepted for signature parity with the other backends; the prefill builder
            already carries the task.
        :param progress_every: Print progress every N examples. This backend is slow enough that a
            silent run is indistinguishable from a hung one.

        :returns: One generation per prompt, truncated and think-stripped, in input order.
        """
        del task
        import torch

        ids = build_prefills(self.prefill, prompts, examples)
        out: List[str] = []
        with torch.no_grad():
            for i, prompt_ids in enumerate(ids):
                tensor = torch.tensor([prompt_ids], device=self.device)
                generated = self.model.generate(
                    tensor,
                    max_new_tokens=stop.max_new_tokens,
                    do_sample=False,  # greedy: eval must be reproducible run to run
                    eos_token_id=self.eos_id if stop.eos else None,
                    pad_token_id=self.eos_id,
                )
                # Slice off the prompt: generate() returns prompt + continuation, and decoding the
                # whole thing would hand the parser a copy of the question.
                completion = generated[0, tensor.shape[1] :]
                out.append(apply_stop(self.tok.decode(completion, skip_special_tokens=True), stop))
                if progress_every and (i + 1) % progress_every == 0:
                    print(f"[ctc-eval] {i + 1}/{len(ids)}", flush=True)
        return out


def build(
    ckpt: str,
    *,
    tokenizer: Optional[str] = None,
    attn: str = "full",
    task: Optional[str] = None,
    query_position: str = "after",
    structural: bool = True,
    device: str = "cuda",
    dtype: str = "bfloat16",
    eos_token_id: Optional[int] = None,
    **model_kwargs: Any,
) -> HFBackend:
    """
    Load a HuggingFace checkpoint.

    :param ckpt: Model directory or hub id.
    :param tokenizer: Tokenizer id or path. Defaults to ``ckpt``.
    :param attn: ``full``, ``chunked`` or ``landmark`` -- affects the prefill only. transformers
        cannot install our masks, so ``chunked`` here grades the chunked token stream under full
        attention.
    :param task: Task name, required when ``structural`` is set.
    :param query_position: Must match what assembled the prompts.
    :param structural: Build the marker scaffold.
    :param device: Torch device string.
    :param dtype: Model dtype.
    :param eos_token_id: Override the tokenizer's eos.
    :param model_kwargs: Passed through to ``from_pretrained``.

    :returns: The backend.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer or ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=getattr(torch, dtype), **model_kwargs
    )
    model.to(device)
    model.eval()

    if structural and not task:
        raise ValueError(
            "structural=True needs task=; the segmenter dispatches on the task name. Pass "
            "structural=False only for a checkpoint trained on flat text."
        )
    prefill = (
        structural_prefill(tok, task=task or "", attn=attn, query_position=query_position)
        if structural
        else plain_prefill(tok)
    )
    return HFBackend(model, tok, prefill=prefill, device=device, eos_token_id=eos_token_id)
