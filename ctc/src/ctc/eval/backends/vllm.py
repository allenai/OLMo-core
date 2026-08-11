"""
vLLM backend: batched generation, by far the fastest for prefill-heavy long-context grading.

Two things about this backend are load-bearing and easy to get wrong.

**It is fed token ids, not prompt strings.** ``TokensPrompt(prompt_token_ids=…)`` with the prefill
built by :mod:`ctc.eval.prefill`, the same builder the native backend uses. Handing vLLM the prompt
*string* would let it tokenize independently, which drops the document-marker scaffold and makes
any comparison against the native backend a comparison of two different inputs.

**vLLM's own stop conditions are an optimisation, never the definition.** They are set where they
can end generation early, but :func:`ctc.eval.stopping.apply` runs afterwards over the full decoded
string and is what actually determines the answer. That ordering is what makes cross-backend parity
possible at all: vLLM's ``stop`` semantics, HuggingFace's ``StoppingCriteria`` and the native
decode loop each cut at slightly different points, and only a host-side truncation over the final
text is identical across all three.

Loading a Qwen3.5 checkpoint is its own problem -- see :func:`build`.

Requires ``pip install 'ctc[vllm]'``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..prefill import Prefill, build_prefills, plain_prefill, structural_prefill
from ..stopping import StopCondition
from ..stopping import apply as apply_stop

__all__ = ["VLLMBackend", "build"]

#: Marker of a text-only olmo export, which vLLM cannot load directly -- see :func:`build`.
TEXT_ONLY_MODEL_TYPES = ("qwen3_5_text",)


class VLLMBackend:
    """
    Greedy batched decoding through vLLM.

    :param llm: A constructed ``vllm.LLM``.
    :param tokenizer: HuggingFace tokenizer, used for the prefill and for decoding.
    :param prefill: How to turn an example into token ids. Defaults to plain tokenization; pass a
        structural builder to grade a document-chunked checkpoint.
    :param eos_token_id: Override the tokenizer's eos.
    :param allow_early_text_stops: Push text stops down to vLLM even when the stop condition says
        the first occurrence might be premature. This is a **speed-for-safety trade**: it can cut
        decode time by an order of magnitude on short-answer tasks, and it can also return the
        preamble instead of the answer. Only enable it against a checkpoint you have checked does
        not emit a leading newline or preamble -- and then compare a rung against the default
        before trusting the number.
    """

    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        *,
        prefill: Optional[Prefill] = None,
        eos_token_id: Optional[int] = None,
        allow_early_text_stops: bool = False,
    ):
        self.llm = llm
        self.tok = tokenizer
        self.prefill = prefill or plain_prefill(tokenizer)
        self.eos_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
        self.allow_early_text_stops = allow_early_text_stops

    def count_tokens(self, text: str) -> int:
        """
        :param text: A prompt.

        :returns: Its length in tokens, for the runner's length audit.
        """
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def sampling_kwargs(self, stop: StopCondition) -> Dict[str, Any]:
        """
        Translate a :class:`StopCondition` into vLLM's knobs.

        Pure, and separate from constructing ``SamplingParams``, so the translation is testable
        without vLLM installed -- which matters because this is where a backend most easily stops
        agreeing with the others.

        **An early stop is only safe if it cuts no earlier than the host-side rule would.** vLLM
        stops at the first literal occurrence of a stop string; :class:`StopCondition` suppresses
        the first occurrence under two conditions vLLM cannot express:

        * ``require_before`` -- oolong's answer follows a templated marker, so an earlier newline
          is preamble. vLLM stopping there returns the preamble and discards the answer.
        * ``require_content`` -- a leading formatting newline before any content. vLLM stopping
          there returns an empty string for every example, which reads as total model failure.

        In both cases the text is *gone* by the time :func:`ctc.eval.stopping.apply` runs, so it
        cannot repair the damage. The stops are therefore withheld and generation runs to eos or
        the budget. Every preset sets ``require_content``, so today that means no text stop is ever
        pushed down -- a real decode cost, paid deliberately. See :data:`allow_early_text_stops`
        for the opt-out.

        :param stop: The task's stop condition.

        :returns: Keyword arguments for ``vllm.SamplingParams``.
        """
        unsafe = bool(stop.require_before) or bool(stop.require_content)
        push_down = stop.text_stops and (self.allow_early_text_stops or not unsafe)

        return dict(
            temperature=0.0,  # greedy: eval must be reproducible run to run
            max_tokens=stop.max_new_tokens,
            stop=list(stop.text_stops) if push_down else None,
            stop_token_ids=[self.eos_id] if stop.eos else None,
            include_stop_str_in_output=True,  # apply_stop decides what to keep, not vLLM
        )

    def generate(
        self,
        prompts: Sequence[str],
        examples: Optional[Sequence[Mapping[str, Any]]] = None,
        *,
        stop: StopCondition,
        task: Optional[str] = None,
    ) -> List[str]:
        """
        :param prompts: Rendered prompts, used only to build the prefill.
        :param examples: The examples they came from, required by a structural prefill.
        :param stop: The task's stop condition.
        :param task: Task name. Unused here -- the prefill builder already carries it -- and
            accepted so every backend has one signature.

        :returns: One generation per prompt, truncated and think-stripped, in input order.
        """
        del task
        from vllm import SamplingParams, TokensPrompt

        ids = build_prefills(self.prefill, prompts, examples)
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=i) for i in ids],
            SamplingParams(**self.sampling_kwargs(stop)),
        )
        return [apply_stop(o.outputs[0].text, stop) for o in _in_submission_order(outputs)]


def _in_submission_order(outputs: Sequence[Any]) -> List[Any]:
    """
    Restore submission order.

    vLLM's ``llm.generate`` returns results in submission order today, and this does not distrust
    that so much as decline to depend on it: the scheduler is free to complete requests in any
    order, and if the list ever came back permuted, every generation would be graded against
    another example's gold answer -- producing a plausible, wrong, and completely silent score.

    :param outputs: ``RequestOutput`` objects.

    :returns: The same objects, ordered by numeric ``request_id`` when every id is numeric,
        otherwise unchanged.
    """
    ids = [str(getattr(o, "request_id", "")) for o in outputs]
    if all(i.isdigit() for i in ids) and len({int(i) for i in ids}) == len(ids):
        return [o for _, o in sorted(zip((int(i) for i in ids), outputs), key=lambda p: p[0])]
    return list(outputs)


def build(
    ckpt: str,
    *,
    tokenizer: str = "Qwen/Qwen3-4B",
    attn: str = "full",
    max_length: int = 4096,
    max_new_tokens: int = 512,
    task: Optional[str] = None,
    query_position: str = "after",
    structural: bool = True,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.6,
    model_family: str = "qwen3_5",
    eos_token_id: Optional[int] = None,
    **llm_kwargs: Any,
) -> VLLMBackend:
    """
    Construct the backend, applying the Qwen3.5 serving recipe.

    **Point this at a serving copy, not at a raw olmo export.** ``export_olmo_to_hf.py`` writes a
    text-only checkpoint (``model_type: qwen3_5_text``, no ``vision_config``), while vLLM resolves
    any ``Qwen3_5*`` architecture to a *multimodal* class whose ``__init__`` reads
    ``config.vision_config``. The load then dies at model construction, before any memory is
    touched, and no amount of ``hf_overrides`` fixes it -- building a serving copy takes three
    separate scripts (a wrapper config, a ``model.*`` -> ``model.language_model.*`` key rename, and
    ~297 synthesized ``visual.*`` params). Prebuilt ones already exist; this function checks and
    refuses rather than letting you discover it from a confusing AttributeError.

    :param ckpt: Serving-copy directory.
    :param tokenizer: Tokenizer id or path.
    :param attn: ``full``, ``chunked`` or ``landmark``. Only affects the prefill here -- vLLM has no
        way to install our masks, so ``chunked`` grades the chunked *token stream* under full
        attention unless the chunked vLLM patch is in play.
    :param max_length: Sequence budget. Raised automatically if a prompt needs more.
    :param max_new_tokens: Decode budget, used only to size ``max_length``.
    :param task: Task name, required when ``structural`` is set.
    :param query_position: Must match what assembled the prompts.
    :param structural: Build the marker scaffold. Turn off only for a model trained on flat text.
    :param tensor_parallel_size: GPUs to shard over.
    :param gpu_memory_utilization: Long rungs need more KV cache.
    :param model_family: ``qwen3_5`` applies the architecture override and
        ``limit_mm_per_prompt`` that make vLLM load only the language model of the multimodal
        wrapper. ``qwen3`` is a plain dense causal LM and needs neither.
    :param eos_token_id: Override the tokenizer's eos.
    :param llm_kwargs: Passed through to ``vllm.LLM``.

    :returns: The backend.

    :raises ValueError: If ``ckpt`` looks like a raw text-only export.
    """
    import json

    from transformers import AutoTokenizer
    from vllm import LLM

    _refuse_text_only_export(Path(ckpt), json)

    tok = AutoTokenizer.from_pretrained(tokenizer)
    prefill = (
        structural_prefill(
            tok, task=_require_task(task, structural), attn=attn, query_position=query_position
        )
        if structural
        else plain_prefill(tok)
    )

    extra: Dict[str, Any] = dict(llm_kwargs)
    if model_family == "qwen3_5":
        extra.setdefault("hf_overrides", {"architectures": ["Qwen3_5ForCausalLM"]})
        extra.setdefault("limit_mm_per_prompt", {"image": 0, "video": 0})

    llm = LLM(
        model=str(ckpt),
        tokenizer=tokenizer,
        max_model_len=max_length,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,  # Qwen3.5's hybrid GDN blocks crash under CUDA-graph capture
        **extra,
    )
    del max_new_tokens  # sizing is the caller's job; recorded in the signature for symmetry
    return VLLMBackend(llm, tok, prefill=prefill, eos_token_id=eos_token_id)


def _require_task(task: Optional[str], structural: bool) -> str:
    if not structural:
        return ""
    if not task:
        raise ValueError(
            "structural=True needs task=; the segmenter dispatches on the task name. Pass "
            "structural=False only for a checkpoint trained on flat text."
        )
    return task


def _refuse_text_only_export(ckpt: Path, json_mod: Any) -> None:
    """
    :param ckpt: Checkpoint directory.
    :param json_mod: The ``json`` module, injected so this stays importable without vLLM.

    :raises ValueError: If the directory is a text-only export rather than a serving copy.
    """
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = json_mod.loads(cfg_path.read_text())
    except ValueError:
        return
    if cfg.get("model_type") in TEXT_ONLY_MODEL_TYPES and "vision_config" not in cfg:
        raise ValueError(
            f"{ckpt} is a TEXT-ONLY olmo export (model_type={cfg.get('model_type')!r}, no "
            "vision_config). vLLM resolves Qwen3_5* to a multimodal class that reads "
            "config.vision_config and will die at model construction.\n"
            "Point this at a serving copy instead -- one should already exist under "
            "/data/prasann/ctc_suite/vllm_serving_4b_v3/<ckpt-name>/. Building a new one takes "
            "three scripts (wrapper config, model.* -> model.language_model.* rename, ~297 dummy "
            "visual.* params); a valid serving dir has ~426 model.language_model.* keys, ~297 "
            "visual.* keys, and vision_config in config.json."
        )
