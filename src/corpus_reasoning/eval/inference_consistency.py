"""
Machinery for **inference-consistency** checks: does the distribution a model assigns to a
continuation under a single teacher-forced forward pass match the distribution it assigns to the
same continuation, token for token, while actually generating it?

The two paths are supposed to compute the same conditional distribution, but they run through
different code: the forward pass masks and attends over the whole sequence at once, while generation
prefills, then decodes one token at a time against a KV cache -- and for every attention variant in
this repo except plain dense, the decode step uses a *different mask builder* than the forward pass
(see :meth:`~olmo_core.nn.attention.summary_token.SummaryTokenAttention._sdpa_cached` vs
:meth:`~olmo_core.nn.attention.summary_token.SummaryTokenAttention._sdpa_masked`, or
:meth:`~olmo_core.generate.generation_module.transformer.TransformerGenerationModule._set_landmark_eval_decode`
vs :func:`~olmo_core.nn.attention.landmark.build_landmark_masks`).

**Teacher forcing is what makes the comparison mean anything.** If the generation path free-runs, its
context diverges from the forward pass's context at the first token where the model's argmax differs
from the gold continuation, and every later position is then conditioned on different history -- a
mismatch no longer tells you the two code paths disagree. So the decode loop here is forced: at each
step the real logits are recorded, and then the **gold** token is fed in regardless of what the model
would have picked. Any difference that survives is attributable to *how* the distribution was
computed, not to what was generated.

Note that "the two paths must agree" is a claim about the *dense* variant only. For the structured
variants the decode path deliberately serves different semantics than a full teacher-forced forward
(landmark decode modes, hard top-k retrieval, the summary-token serving mask), so a gap there is a
measurement, not a failure -- see :class:`~test.inference_consistency.variants.VariantSpec` and the
``expect`` field it carries.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence
from unittest import mock

import torch
import torch.nn.functional as F

from olmo_core.generate.generation_module import TransformerGenerationModule
from olmo_core.generate.generation_module.transformer import generation_module as gm_mod

__all__ = [
    "DecodeTrace",
    "ConsistencyReport",
    "forced_generate_batch",
    "forced_manual_decode",
    "reference_forward",
    "reference_step_logits",
    "compare_paths",
]


@dataclass
class DecodeTrace:
    """
    A record of one forced-decode run: the exact token stream the model was fed, the logits it
    produced at each forced step, and the position whose query produced each of those logits.

    Reporting ``fed_ids`` rather than reconstructing it from the prompt is what keeps the comparison
    exact for variants whose generation path rewrites the token stream -- landmark generation inserts
    memory tokens into the prompt, and the document-chunked eval loop injects a landmark token after
    every ``mem_freq`` generated tokens. The reference forward runs over precisely this stream.

    :param fed_ids: The full model-space token stream, shape ``(1, L)``.
    :param step_logits: Logits recorded at each forced decode step, shape ``(S, vocab_size)``.
    :param step_query_pos: For each step, the index into ``fed_ids`` whose query produced that
        step's logits. ``step_logits[s]`` predicts the token at ``step_query_pos[s] + 1``.
    :param gold: The gold token forced at each step, length ``S``.
    """

    fed_ids: torch.Tensor
    step_logits: torch.Tensor
    step_query_pos: List[int]
    gold: List[int]

    def __post_init__(self):
        s = len(self.gold)
        if self.step_logits.shape[0] != s or len(self.step_query_pos) != s:
            raise ValueError(
                f"DecodeTrace is ragged: {s} gold tokens, {self.step_logits.shape[0]} logit rows, "
                f"{len(self.step_query_pos)} query positions."
            )
        # Every recorded query position must actually predict the gold token that was forced there.
        for s_i, (pos, tok) in enumerate(zip(self.step_query_pos, self.gold)):
            if self.fed_ids[0, pos + 1].item() != tok:
                raise ValueError(
                    f"step {s_i}: fed_ids[{pos + 1}]={int(self.fed_ids[0, pos + 1])} but the forced "
                    f"gold token was {tok}; the trace's alignment is wrong."
                )


@dataclass
class ConsistencyReport:
    """
    Per-token and aggregate agreement between the teacher-forced forward pass and the generation
    path, over the same gold continuation.

    ``ce_forward`` and ``ce_generate`` are the two answers to "what cross-entropy does this model
    assign to this continuation?" -- the number a loss/perplexity script would report, and the number
    implied by the distributions the model actually produced while generating. They are the headline
    comparison; the distribution-level fields say whether an apparent agreement is real or a
    coincidence of the gold token's logprob.
    """

    variant: str
    n_steps: int
    ce_forward: float
    ce_generate: float
    max_abs_logprob_delta: float
    mean_kl: float
    max_kl: float
    top1_agreement: float
    first_divergent_step: Optional[int]
    per_step_kl: List[float] = field(default_factory=list)
    per_step_logprob_delta: List[float] = field(default_factory=list)

    @property
    def ce_delta(self) -> float:
        """Signed difference in mean cross-entropy, ``generate - forward``."""
        return self.ce_generate - self.ce_forward

    def summary(self) -> str:
        """A one-block human-readable summary, suitable for a test failure message or a report."""
        div = "none" if self.first_divergent_step is None else str(self.first_divergent_step)
        return (
            f"[{self.variant}] steps={self.n_steps}\n"
            f"  CE  forward={self.ce_forward:.6f}  generate={self.ce_generate:.6f}  "
            f"delta={self.ce_delta:+.6f}\n"
            f"  logprob delta  max|.|={self.max_abs_logprob_delta:.3e}\n"
            f"  KL(fwd||gen)   mean={self.mean_kl:.3e}  max={self.max_kl:.3e}\n"
            f"  top-1 agreement={self.top1_agreement:.4f}  first argmax divergence at step={div}"
        )


@contextmanager
def _force_tokens(gold: Sequence[int]) -> Iterator[Dict[str, int]]:
    """
    Patch the generation module's token selection so the decode loop emits ``gold`` in order.

    Everything else in :meth:`TransformerGenerationModule.generate_batch` -- prefill, KV cache setup,
    landmark prompt construction and decode-mode configuration, top-k retrieval, chunked prefill --
    runs untouched, so the logits under test come from the real generation path rather than a
    reimplementation of it. Only the choice of which token to append is overridden.

    :param gold: Token ids to emit, one per decode step.

    :returns: A dict whose ``"steps"`` key holds the number of selections made, so the caller can
        confirm the loop consumed exactly the tokens it supplied.
    """
    state = {"steps": 0}

    def _select(logits: torch.Tensor, **kwargs) -> torch.Tensor:
        i = state["steps"]
        state["steps"] = i + 1
        if i >= len(gold):
            # The loop is running past the forced continuation; hold on the last token. The caller
            # sizes max_new_tokens to len(gold), so this is a guard, not a normal path.
            i = len(gold) - 1
        return torch.full((logits.shape[0],), int(gold[i]), dtype=torch.long, device=logits.device)

    with mock.patch.object(gm_mod, "select_next_token", _select):
        yield state


def forced_generate_batch(
    gm: TransformerGenerationModule,
    prompt_ids: torch.Tensor,
    gold: Sequence[int],
    *,
    model_space_prompt: Optional[torch.Tensor] = None,
    **generation_kwargs,
) -> DecodeTrace:
    """
    Run :meth:`TransformerGenerationModule.generate_batch` over ``prompt_ids``, forcing ``gold`` as
    the generated tokens, and record the logits produced at each step.

    :param gm: The generation module under test.
    :param prompt_ids: Content-space prompt of shape ``(1, P)`` -- what an eval harness would pass.
    :param gold: The gold continuation to force, in content space.
    :param model_space_prompt: The token stream the model actually sees for this prompt, of shape
        ``(1, P')``, when generation rewrites it. Landmark generation inserts memory tokens, so pass
        :func:`~olmo_core.generate.generation_module.transformer.generation_module._build_landmark_prompt`
        applied to ``prompt_ids``; leave ``None`` when the prompt reaches the model unchanged.
    :param generation_kwargs: Overrides forwarded to ``generate_batch``.

    :returns: The decode trace.

    :raises ValueError: If the decode loop did not take exactly ``len(gold)`` steps, which means the
        run stopped early (an EOS or stop-string hit) and the trace would be misaligned.
    """
    if prompt_ids.shape[0] != 1:
        raise ValueError("forced decoding is defined for batch size 1 only.")

    with _force_tokens(gold) as state:
        _, logits, _ = gm.generate_batch(
            prompt_ids,
            return_logits=True,
            completions_only=True,
            log_timing=False,
            max_new_tokens=len(gold),
            **generation_kwargs,
        )

    if logits is None:
        raise ValueError("generate_batch returned no logits; pass return_logits=True.")
    if state["steps"] != len(gold):
        raise ValueError(
            f"decode loop took {state['steps']} steps for {len(gold)} gold tokens -- it stopped "
            "early (EOS or a stop string), so the trace cannot be aligned. Choose a gold "
            "continuation that avoids the EOS id, or disable stop conditions."
        )

    prompt_in_model_space = prompt_ids if model_space_prompt is None else model_space_prompt
    gold_t = torch.tensor([list(gold)], dtype=prompt_ids.dtype, device=prompt_ids.device)
    fed_ids = torch.cat([prompt_in_model_space.to(prompt_ids.device), gold_t], dim=1)

    # Step s is produced by the query at the token just before the gold token it predicts. This holds
    # for both decode shapes in ``generate_batch``: the ordinary path prefills the whole prompt with
    # ``logits_to_keep=1`` (so step 0's query is the last prompt token), and the landmark path
    # prefills ``prompt[:-1]`` and re-queries the final prompt token as its first decode step.
    p = prompt_in_model_space.shape[1]
    step_query_pos = [p - 1 + s for s in range(len(gold))]

    return DecodeTrace(
        fed_ids=fed_ids,
        step_logits=logits[0].float(),
        step_query_pos=step_query_pos,
        gold=list(gold),
    )


def forced_manual_decode(
    gm: TransformerGenerationModule,
    prefill_ids: Sequence[int],
    gold: Sequence[int],
    *,
    device: torch.device,
    max_length: int,
    inject_every: Optional[int] = None,
    inject_token: Optional[int] = None,
) -> DecodeTrace:
    """
    Forced decode through the bespoke prefill-then-single-token loop that
    ``src/corpus_reasoning/eval/eval_lc_native_docchunk.py`` uses, rather than through
    ``generate_batch``.

    That eval script does not call ``generate_batch`` at all -- it drives ``gm.model`` directly -- so
    a consistency check against ``generate_batch`` would not cover the code path that produces its
    reported numbers. This mirrors its loop, including the landmark token it injects after every
    ``mem_freq`` generated tokens to keep the tail block-aligned.

    :param gm: The generation module under test.
    :param prefill_ids: The prompt token ids, already in model space.
    :param gold: The gold continuation to force.
    :param device: Device to run on.
    :param max_length: Cache size to allocate, at least ``len(prefill_ids) + len(gold)`` plus any
        injected tokens.
    :param inject_every: Inject ``inject_token`` after every this many generated tokens (the
        landmark variant's behaviour); ``None`` disables injection.
    :param inject_token: The token id to inject.

    :returns: The decode trace.
    """
    if inject_every is not None and inject_token is None:
        raise ValueError("inject_every requires inject_token.")

    gm.prepare_inference_cache(1, max_length)
    leftpad = torch.zeros(1, dtype=torch.int32, device=device)

    fed: List[int] = list(prefill_ids)
    step_logits: List[torch.Tensor] = []
    step_query_pos: List[int] = []

    logits = gm.model(
        torch.tensor([list(prefill_ids)], device=device), logits_to_keep=1, cache_leftpad=leftpad
    )
    since_inject = 0
    for s, tok in enumerate(gold):
        # ``logits`` currently holds the prediction for this step, produced by the query at the last
        # token fed so far.
        step_logits.append(logits[0, -1].float().detach())
        step_query_pos.append(len(fed) - 1)
        fed.append(int(tok))

        if s == len(gold) - 1:
            break

        logits = gm.model(torch.tensor([[int(tok)]], device=device), logits_to_keep=1)
        since_inject += 1
        if inject_every is not None and since_inject == inject_every:
            assert inject_token is not None
            logits = gm.model(torch.tensor([[inject_token]], device=device), logits_to_keep=1)
            fed.append(int(inject_token))
            since_inject = 0

    return DecodeTrace(
        fed_ids=torch.tensor([fed], device=device),
        step_logits=torch.stack(step_logits),
        step_query_pos=step_query_pos,
        gold=list(gold),
    )


def reference_forward(
    gm: TransformerGenerationModule,
    fed_ids: torch.Tensor,
    *,
    train_mode: bool = False,
    pad_to_multiple: Optional[int] = None,
    pad_id: int = 0,
) -> torch.Tensor:
    """
    Run one teacher-forced forward pass over ``fed_ids`` with **no KV cache**, returning full logits.

    Freeing the inference cache is load-bearing rather than tidiness: every variant's ``sdpa``
    branches on ``self.kv_cache_manager is None``, so leaving a populated cache in place would send
    this "reference" down the same cached path it is meant to be checked against, and the comparison
    would pass by construction.

    :param gm: The generation module whose model to run.
    :param fed_ids: The exact token stream to score, shape ``(1, L)``.
    :param train_mode: Run under ``model.train()`` instead of ``model.eval()``. The summary-token
        mask consults ``self.training`` to decide between the training mask and the serving mask
        (see :meth:`~olmo_core.nn.transformer.Transformer.set_summary_eval_mask_mode`), so this
        selects which of the two semantics the reference represents. Leave ``False`` for a
        correctness check; set ``True`` to measure the train/serve gap.
    :param pad_to_multiple: Right-pad the sequence to a multiple of this before the forward. The
        eager landmark forward rejects a sequence that is not a whole number of blocks, and a forced
        continuation almost never lands on a block boundary. The padding is appended, so it is
        causally in the future of every position being scored and cannot change their logits.
    :param pad_id: Token id to pad with.

    :returns: Logits of shape ``(1, L, vocab_size)`` for the *unpadded* length ``L``, in float32.
    """
    gm.free_inference_cache()

    real_len = fed_ids.shape[1]
    ids = fed_ids
    if pad_to_multiple:
        n_pad = (-real_len) % pad_to_multiple
        if n_pad:
            tail = torch.full((1, n_pad), pad_id, dtype=ids.dtype, device=ids.device)
            ids = torch.cat([ids, tail], dim=1)

    if train_mode:
        # ``model_forward`` forces eval mode, so drive the model directly for the training-semantics
        # reference. Dropout is off in these configs; the mode flag is read for mask selection.
        gm.model.train()
        with torch.inference_mode():
            logits = gm.model(ids.to(gm.device))
        gm.model.eval()
    else:
        logits = gm.model_forward(ids)
    return logits[:, :real_len].float()


def reference_step_logits(
    gm: TransformerGenerationModule,
    trace: DecodeTrace,
    *,
    train_mode: bool = False,
    pad_to_multiple: Optional[int] = None,
    pad_id: int = 0,
) -> torch.Tensor:
    """
    Teacher-forced forward over ``trace.fed_ids`` returning logits **only** at the positions the
    trace compares, shape ``(S, vocab_size)``.

    This is the form to use on real checkpoints. Materializing full logits is fine for the tiny test
    models but not for a production one: at a 250k vocab and a 16k-token prompt the full tensor is
    several terabytes. Passing the query positions as ``logits_to_keep`` restricts the LM head to the
    handful of rows the comparison actually reads.

    :param gm: The generation module.
    :param trace: The forced-decode trace whose positions to score.
    :param train_mode: Run under ``model.train()``; see :func:`reference_forward`.
    :param pad_to_multiple: Block alignment for variants whose eager forward rejects a partial block.
    :param pad_id: Token id to pad with.

    :returns: Logits at each compared position, shape ``(S, vocab_size)``, in float32.
    """
    gm.free_inference_cache()

    ids = trace.fed_ids
    if pad_to_multiple:
        n_pad = (-ids.shape[1]) % pad_to_multiple
        if n_pad:
            tail = torch.full((1, n_pad), pad_id, dtype=ids.dtype, device=ids.device)
            ids = torch.cat([ids, tail], dim=1)
    ids = ids.to(gm.device)
    pos = torch.tensor([trace.step_query_pos], dtype=torch.long, device=gm.device)

    if train_mode:
        gm.model.train()
        with torch.inference_mode():
            logits = gm.model(ids, logits_to_keep=pos)
        gm.model.eval()
    else:
        logits = gm.model_forward(ids, logits_to_keep=pos)
    return logits[0].float()


def compare_paths(
    variant: str,
    trace: DecodeTrace,
    ref_logits: torch.Tensor,
) -> ConsistencyReport:
    """
    Compare the generation path's per-step distributions against the teacher-forced forward pass.

    Both the gold-token cross-entropy and the full-distribution KL are reported. The CE alone is not
    enough: two paths can assign the gold token the same logprob while disagreeing about everything
    else, which is exactly the shape a mask bug takes when the gold token happens to sit in a region
    both masks admit.

    :param variant: Name of the variant under test, for the report.
    :param trace: The forced-decode trace.
    :param ref_logits: Either full logits over ``trace.fed_ids``, shape ``(1, L, vocab_size)``, as
        returned by :func:`reference_forward`; or logits already restricted to the compared
        positions, shape ``(S, vocab_size)``, as returned by :func:`reference_step_logits`.

    :returns: The comparison report.
    """
    if ref_logits.dim() == 3:
        pos = torch.tensor(trace.step_query_pos, device=ref_logits.device)
        ref_step = ref_logits[0].index_select(0, pos).float()  # (S, V)
    elif ref_logits.dim() == 2:
        ref_step = ref_logits.float()
    else:
        raise ValueError(
            f"ref_logits must be (1, L, V) full logits or (S, V) per-step logits, got "
            f"{tuple(ref_logits.shape)}."
        )
    if ref_step.shape[0] != len(trace.gold):
        raise ValueError(
            f"reference has {ref_step.shape[0]} rows for {len(trace.gold)} compared steps."
        )
    gen_step = trace.step_logits.to(ref_step.device).float()  # (S, V)

    ref_lp = F.log_softmax(ref_step, dim=-1)
    gen_lp = F.log_softmax(gen_step, dim=-1)

    gold = torch.tensor(trace.gold, device=ref_lp.device).unsqueeze(-1)
    ref_gold_lp = ref_lp.gather(-1, gold).squeeze(-1)
    gen_gold_lp = gen_lp.gather(-1, gold).squeeze(-1)

    kl = (ref_lp.exp() * (ref_lp - gen_lp)).sum(-1)

    ref_top1 = ref_lp.argmax(-1)
    gen_top1 = gen_lp.argmax(-1)
    agree = ref_top1 == gen_top1
    first_div = None if bool(agree.all()) else int((~agree).nonzero()[0].item())

    delta = (ref_gold_lp - gen_gold_lp).abs()

    return ConsistencyReport(
        variant=variant,
        n_steps=len(trace.gold),
        ce_forward=float(-ref_gold_lp.mean()),
        ce_generate=float(-gen_gold_lp.mean()),
        max_abs_logprob_delta=float(delta.max()),
        mean_kl=float(kl.mean()),
        max_kl=float(kl.max()),
        top1_agreement=float(agree.float().mean()),
        first_divergent_step=first_div,
        per_step_kl=[float(x) for x in kl],
        per_step_logprob_delta=[float(x) for x in (ref_gold_lp - gen_gold_lp)],
    )


def run_variant(
    build: Callable[[], TransformerGenerationModule],
    prompt_ids: torch.Tensor,
    gold: Sequence[int],
    *,
    variant: str,
    train_mode: bool = False,
    **kwargs,
) -> ConsistencyReport:
    """
    Convenience wrapper: build the module, force-decode, score the same stream teacher-forced, and
    compare.

    :param build: Callable returning the generation module.
    :param prompt_ids: Content-space prompt, shape ``(1, P)``.
    :param gold: Gold continuation to force.
    :param variant: Name for the report.
    :param train_mode: Passed to :func:`reference_forward`.
    :param kwargs: Forwarded to :func:`forced_generate_batch`.

    :returns: The comparison report.
    """
    gm = build()
    trace = forced_generate_batch(gm, prompt_ids, gold, **kwargs)
    ref = reference_forward(gm, trace.fed_ids, train_mode=train_mode)
    return compare_paths(variant, trace, ref)
