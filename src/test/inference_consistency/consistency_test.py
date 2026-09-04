"""
Inference-consistency tests: for each attention variant, does the distribution the model assigns to
a continuation under a single teacher-forced forward pass match the distribution it assigns to the
same continuation while generating it?

Read :mod:`corpus_reasoning.eval.inference_consistency` for why the decode loop is forced rather than left to
free-run. The short version: without forcing, the two paths stop conditioning on the same history at
the first token where the model's argmax differs from the gold continuation, and a mismatch after
that point says nothing about whether the code paths agree.

What each variant owes is not the same. ``dense_nocache`` and ``dense`` compute one function and must
reproduce it exactly. The landmark variants deliberately serve different semantics at decode time
than a batched forward computes, so their tests record the divergence and bound it rather than
asserting it away -- see the ``why`` field on each
:class:`~test.inference_consistency.variants.VariantSpec`.
"""

import pytest
import torch

from corpus_reasoning.eval.inference_consistency import (
    compare_paths,
    forced_generate_batch,
    reference_forward,
    reference_step_logits,
)
from olmo_core.testing.utils import has_flash_attn_2

from .variants import (
    IDS,
    LANDMARK_NO_TOPK,
    VARIANTS,
    VariantSpec,
    gold_continuation,
    landmark_drift_gold_budget,
    model_space_prompt,
    sequences_for,
)


def _run(spec: VariantSpec, *, train_mode: bool = False):
    """
    Force-decode a gold continuation, score the same stream teacher-forced, and compare.

    :param spec: The variant under test.
    :param train_mode: Passed through to :func:`~corpus_reasoning.eval.inference_consistency.reference_forward`.

    :returns: The :class:`~corpus_reasoning.eval.inference_consistency.ConsistencyReport`.
    """
    gm = spec.build()
    prompt, gold = sequences_for(spec)
    msp = model_space_prompt(spec, gm, prompt)

    if spec.is_landmark and spec.expect == "identical":
        # Exact agreement is only claimable until the continuation reaches the next landmark slot;
        # past that the eager forward reads a generated content token as a landmark and the decode
        # path does not. Fail loudly with the reason rather than letting a geometry change surface as
        # a mysterious numerical divergence.
        assert msp is not None and spec.block_multiple is not None
        budget = landmark_drift_gold_budget(msp.shape[1], spec.block_multiple)
        assert spec.n_gold <= budget, (
            f"{spec.name} asks for exact agreement over {spec.n_gold} gold tokens, but with a "
            f"{msp.shape[1]}-token landmark prompt and block_size={spec.block_multiple} only "
            f"{budget} land before the next landmark slot. Beyond that the two paths disagree by "
            f"construction, not because of a bug."
        )

    trace = forced_generate_batch(
        gm,
        prompt,
        gold,
        model_space_prompt=msp,
        **spec.generation_kwargs,
    )
    ref = reference_forward(
        gm,
        trace.fed_ids,
        train_mode=train_mode,
        pad_to_multiple=spec.block_multiple,
        pad_id=IDS.pad,
    )
    return compare_paths(spec.name, trace, ref)


def _skip_if_unsupported(spec: VariantSpec):
    """Skip when this variant's cached decode path cannot run on the current machine."""
    if spec.requires_gpu and not torch.cuda.is_available():
        pytest.skip("requires a GPU (the KV cache lives behind the flash-attention backend)")
    if spec.requires_flash and not has_flash_attn_2:
        pytest.skip("requires flash-attn 2")


def _param(name: str):
    spec = VARIANTS[name]
    marks = []
    if spec.requires_gpu:
        marks.append(pytest.mark.gpu)
    return pytest.param(name, marks=marks, id=name)


IDENTICAL = [n for n, s in VARIANTS.items() if s.expect == "identical"]
GAP = [n for n, s in VARIANTS.items() if s.expect == "gap"]


@pytest.mark.parametrize("name", [_param(n) for n in IDENTICAL])
def test_generation_matches_teacher_forced_forward(name: str):
    """
    The generation path must reproduce the teacher-forced forward pass for every variant whose two
    code paths are supposed to compute the same function.

    A failure here means the cross-entropy a loss or perplexity script reports for a continuation is
    not the cross-entropy implied by the distributions the model actually produced while generating
    it -- so one of the two numbers is describing a model that does not exist.
    """
    spec = VARIANTS[name]
    _skip_if_unsupported(spec)

    report = _run(spec)

    assert report.max_abs_logprob_delta < spec.atol, report.summary()
    assert report.max_kl < spec.kl_atol, report.summary()
    # The gold-token logprob agreeing is not enough on its own: a mask bug that only moves
    # probability among tokens the gold answer never uses leaves the CE untouched. Requiring the
    # argmax to agree at every step is the cheap check that the rest of the distribution moved too.
    assert report.top1_agreement == 1.0, report.summary()


@pytest.mark.parametrize("name", [_param(n) for n in GAP])
def test_designed_decode_gap_is_bounded_and_reported(name: str, capsys):
    """
    For variants whose decode path deliberately serves different semantics than a batched forward,
    record the divergence and hold it under a budget.

    This is not a correctness assertion, and it must not be read as one: the gap is expected. What
    the budget catches is the version of "different" that is a bug -- a decode path that has drifted
    far enough to no longer be conditioning on the prompt still shows up as a gap, and with no
    ceiling this test would pass on it. The printed report is the actual deliverable; it is the
    number to quote when asking whether a landmark model's eval-time behaviour resembles what its
    loss curve was measuring.
    """
    spec = VARIANTS[name]
    _skip_if_unsupported(spec)

    report = _run(spec)

    with capsys.disabled():
        print("\n" + report.summary())
        print(f"  designed gap: {spec.why}")

    assert spec.gap_max_kl_budget is not None, f"{name} is expect='gap' with no budget set"
    assert report.max_kl < spec.gap_max_kl_budget, (
        f"{name} diverges further than its recorded budget "
        f"({report.max_kl:.4f} > {spec.gap_max_kl_budget}); the decode path may have stopped "
        f"attending to the prompt rather than merely serving different semantics.\n"
        + report.summary()
    )
    assert report.ce_forward == report.ce_forward, "forward CE is NaN"
    assert report.ce_generate == report.ce_generate, "generate CE is NaN"


def test_landmark_gap_decomposes_into_topk_and_block_drift(capsys):
    """
    Pin the two independent causes of the landmark decode gap, so neither can be mistaken for the
    other or for a cache bug.

    The landmark decode diverges from the batched forward for exactly two reasons, and this
    separates them by varying one at a time:

    1. **Hard top-k retrieval**, applied on single-query decode steps only. Turning it off closes
       the gap completely -- the cached decode reproduces the eager forward to float32 noise.
    2. **Landmark drift over generated tokens.** Landmark slots are fixed by absolute position, and
       ``generate_batch`` never inserts landmark tokens among generated ones, so once the
       continuation reaches the next landmark slot the eager forward reads that generated *content*
       token as a landmark and decode does not. This reappears with top-k off and is unrelated to it.

    Without this decomposition the headline "landmark decode disagrees with the forward pass" is
    unactionable, and a genuine KV-cache regression would hide inside a gap everyone had learned to
    expect.
    """
    spec = VARIANTS["sparse_landmark"]
    _skip_if_unsupported(spec)

    prompt = spec.make_prompt()
    gm0 = spec.build()
    msp = model_space_prompt(spec, gm0, prompt)
    assert msp is not None and spec.block_multiple is not None
    budget = landmark_drift_gold_budget(msp.shape[1], spec.block_multiple)

    def measure(n_gold: int, **gen_kwargs):
        gm = spec.build()
        gold = gold_continuation(n_gold)
        trace = forced_generate_batch(gm, prompt, gold, model_space_prompt=msp, **gen_kwargs)
        ref = reference_step_logits(gm, trace, pad_to_multiple=spec.block_multiple, pad_id=IDS.pad)
        return compare_paths(spec.name, trace, ref)

    in_block = budget - 2
    past_slot = budget + 4

    topk_on = measure(in_block)
    topk_off = measure(in_block, **LANDMARK_NO_TOPK)
    topk_off_past_slot = measure(past_slot, **LANDMARK_NO_TOPK)

    with capsys.disabled():
        print(f"\nlandmark gap decomposition (block_size={spec.block_multiple}, budget={budget})")
        for label, r in [
            (f"top-k ON  (default 0.1), gold={in_block} in-block", topk_on),
            (f"top-k OFF,               gold={in_block} in-block", topk_off),
            (f"top-k OFF,               gold={past_slot} past slot", topk_off_past_slot),
        ]:
            print(f"  {label:<48} max_kl={r.max_kl:.3e}  top1={r.top1_agreement:.2f}")

    # 1. Top-k off, inside the block: the paths must actually agree.
    assert topk_off.max_kl < 1e-5, topk_off.summary()
    assert topk_off.top1_agreement == 1.0, topk_off.summary()

    # 2. Top-k is what opens the gap in the default configuration.
    assert topk_on.max_kl > 100 * max(topk_off.max_kl, 1e-12), (
        "turning top-k retrieval off did not measurably change the comparison, so this test is not "
        f"exercising it:\n  on:  {topk_on.summary()}\n  off: {topk_off.summary()}"
    )

    # 3. Landmark drift is a separate cause that survives with top-k off.
    assert topk_off_past_slot.max_kl > 1e-4, (
        "running the continuation past the next landmark slot did not reintroduce a gap, so the "
        "landmark-drift mechanism this test documents is not what it claims:\n"
        + topk_off_past_slot.summary()
    )


@pytest.mark.parametrize("mode", ["extend_last_block", "generation_only"])
def test_landmark_drift_budget_predicts_break_in_both_decode_modes(mode: str):
    """
    :func:`landmark_drift_gold_budget` must predict the exact continuation length at which the
    generation and forward paths start to disagree, for **both** landmark decode modes.

    Both modes are needed to pin the formula. They produce differently-aligned prompts --
    ``generation_only`` pads the prompt to end exactly on a landmark slot, ``extend_last_block``
    leaves a partial trailing block -- so a formula anchored on the wrong reference point can match
    one and miss the other. An earlier version measured from the last prompt position rather than
    the first landmark slot a *generated* token occupies, which was right for ``extend_last_block``
    and off by a whole block for ``generation_only``: it treated the prompt's own real landmark as
    the drift point.
    """
    spec = VARIANTS["sparse_landmark"]
    _skip_if_unsupported(spec)

    from olmo_core.generate.generation_module.transformer.generation_module import (
        _build_landmark_prompt,
    )

    prompt = spec.make_prompt()
    gm0 = spec.build()
    layers = gm0._landmark_attention_layers()
    mem_freq = int(getattr(layers[0], "mem_freq"))
    block_size = mem_freq + 1
    msp = _build_landmark_prompt(
        prompt, mem_freq, IDS.landmark, mode=mode, pad_id=IDS.pad, num_landmarks=1
    )
    budget = landmark_drift_gold_budget(msp.shape[1], block_size)

    def measure(n_gold: int):
        gm = spec.build()
        trace = forced_generate_batch(
            gm,
            prompt,
            gold_continuation(n_gold),
            model_space_prompt=msp,
            landmark_decode_mode=mode,
            **LANDMARK_NO_TOPK,
        )
        ref = reference_step_logits(gm, trace, pad_to_multiple=block_size, pad_id=IDS.pad)
        return compare_paths(spec.name, trace, ref)

    at_budget = measure(budget)
    past_budget = measure(budget + 1)

    assert at_budget.max_kl < 1e-5 and at_budget.top1_agreement == 1.0, (
        f"{mode}: budget={budget} was predicted safe but the paths already disagree there.\n"
        + at_budget.summary()
    )
    assert past_budget.max_kl > 1e-4, (
        f"{mode}: budget={budget} predicts divergence at {budget + 1} gold tokens, but the paths "
        f"still agree there -- the budget is too conservative and is costing real coverage.\n"
        + past_budget.summary()
    )


def test_harness_control_has_teeth():
    """
    Deliberately misalign the trace and confirm the comparison notices.

    Without this, a bug that made every variant's report come back empty or trivially equal would
    read as five passing consistency tests. The control asserts the machinery can actually fail:
    shifting the query positions by one must break the agreement that
    :func:`test_generation_matches_teacher_forced_forward` relies on.
    """
    spec = VARIANTS["dense_nocache"]
    gm = spec.build()
    prompt, gold = sequences_for(spec)
    trace = forced_generate_batch(gm, prompt, gold, **spec.generation_kwargs)
    ref = reference_forward(gm, trace.fed_ids)

    aligned = compare_paths("aligned", trace, ref)
    assert aligned.top1_agreement == 1.0, aligned.summary()

    trace.step_query_pos = [p - 1 for p in trace.step_query_pos]
    shifted = compare_paths("shifted", trace, ref)
    assert shifted.max_kl > 1e-3, (
        "shifting the reference by one position did not change the comparison, so the test is not "
        "reading the positions it thinks it is:\n" + shifted.summary()
    )


def test_trace_rejects_misaligned_gold():
    """
    The trace must refuse to be built if the recorded query positions do not predict the tokens that
    were actually forced -- the failure mode that would silently compare the wrong pairs of
    distributions.
    """
    import torch

    from corpus_reasoning.eval.inference_consistency import DecodeTrace

    with pytest.raises(ValueError, match="alignment is wrong"):
        DecodeTrace(
            fed_ids=torch.tensor([[1, 2, 3, 4]]),
            step_logits=torch.zeros(2, 8),
            step_query_pos=[0, 1],
            gold=[99, 100],  # fed_ids says 2, 3
        )
