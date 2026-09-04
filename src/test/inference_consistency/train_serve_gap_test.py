"""
Train/serve mask-gap tests: is the function a variant is *served* at inference the same function its
training loss was measuring?

This is the second half of the inference-consistency question and a different failure mode from the
one in :mod:`test.inference_consistency.consistency_test`. There, both paths were asked to compute
the serving semantics and had to agree. Here both paths are teacher-forced forwards over an identical
token stream, differing only in ``model.training`` -- so any disagreement is the model computing a
*different function* at eval time than at train time.

That gap is not always a bug. For the summary-token variant it is a deliberate serving decision
(:meth:`~olmo_core.nn.transformer.Transformer.set_summary_eval_mask_mode`), and both arms are
legitimate questions to ask a checkpoint. What is a bug is serving an arm the model never trained on
and reading the result as a capability finding -- which is why these tests pin the default and assert
the two arms are actually distinguishable, rather than checking any particular number.

None of this needs a KV cache, so unlike the generation-path tests these all run on CPU.
"""

import torch

from corpus_reasoning.eval.inference_consistency import (
    reference_forward,
)

from .variants import (
    IDS,
    N_SUMMARY_TOKENS,
    build_dense,
    build_document_chunked,
    build_summary_token,
    document_prompt,
    plain_prompt,
)


def _forward_gap(gm, fed_ids, *, pad_to_multiple=None) -> float:
    """
    Max per-step KL between the training-mode and eval-mode teacher-forced forwards over the same
    stream.

    :param gm: The generation module.
    :param fed_ids: The token stream to score.
    :param pad_to_multiple: Block alignment for the reference forward, if the variant needs it.

    :returns: Max KL over all positions.
    """
    train = reference_forward(
        gm, fed_ids, train_mode=True, pad_to_multiple=pad_to_multiple, pad_id=IDS.pad
    )
    serve = reference_forward(
        gm, fed_ids, train_mode=False, pad_to_multiple=pad_to_multiple, pad_id=IDS.pad
    )
    train_lp = torch.log_softmax(train[0], dim=-1)
    serve_lp = torch.log_softmax(serve[0], dim=-1)
    kl = (train_lp.exp() * (train_lp - serve_lp)).sum(-1)
    return float(kl.max())


def test_dense_has_no_train_serve_gap():
    """
    Plain causal attention must compute the same function in both modes -- the control that gives
    the summary-token result below its meaning.

    If this drifts, the gap measured for any other variant cannot be attributed to its mask, because
    something mode-dependent (dropout left on, a norm in the wrong state) is moving the logits for
    every model.
    """
    gm = build_dense()
    ids = plain_prompt(48)
    assert _forward_gap(gm, ids) < 1e-6


def test_summary_token_serving_default_is_causal():
    """
    A summary-token model must default to serving the **causal** arm.

    Serving the restricted mask to a model that trained entirely on the causal arm is a train/test
    mismatch that presents as a capability result -- the model looks unable to answer from summaries
    when it was simply never asked to. The default is the safeguard, so pin it here rather than
    leaving it to be rediscovered from a suspicious eval.
    """
    gm = build_summary_token()
    assert gm.model._summary_eval_mask_mode == "causal"


def test_summary_token_arms_are_distinguishable():
    """
    The two summary-token serving arms must actually produce different distributions.

    This is the negative control for the test above. If ``"restricted"`` and ``"causal"`` happened to
    compute the same thing -- a roles tensor that came back empty, a mask that silently degraded to
    causal, a layout the builder did not recognize -- then pinning the default would be protecting
    nothing, and an eval that reported "no difference between arms" would be reporting a broken mask
    rather than a finding about summaries.
    """
    gm = build_summary_token()
    ids = document_prompt(n_docs=3, doc_len=9, n_summary_tokens=N_SUMMARY_TOKENS)

    gm.model.set_summary_eval_mask_mode("causal")
    causal = reference_forward(gm, ids)
    gm.model.set_summary_eval_mask_mode("restricted")
    restricted = reference_forward(gm, ids)

    causal_lp = torch.log_softmax(causal[0], dim=-1)
    restricted_lp = torch.log_softmax(restricted[0], dim=-1)
    kl = (causal_lp.exp() * (causal_lp - restricted_lp)).sum(-1)
    assert float(kl.max()) > 1e-3, (
        "the two summary-token serving arms are numerically identical, so the mask is not being "
        "applied; any eval comparing them would be measuring nothing."
    )


def test_summary_token_restricted_arm_matches_training_mask():
    """
    Serving the ``"restricted"`` arm must reproduce what the training forward computes.

    ``restricted`` is defined as "no example is on the causal arm", which is exactly the state a
    training forward is in when the mixture probability is zero. If these two diverge, the arm named
    after the training mask is not the training mask.
    """
    gm = build_summary_token()
    ids = document_prompt(n_docs=3, doc_len=9, n_summary_tokens=N_SUMMARY_TOKENS)
    gm.model.set_summary_eval_mask_mode("restricted")
    assert _forward_gap(gm, ids) < 1e-6


def test_summary_token_causal_arm_differs_from_training_mask():
    """
    Serving the ``"causal"`` arm must differ from the training forward -- that is the whole point of
    the arm, and the measurement a summary-token eval is implicitly making.
    """
    gm = build_summary_token()
    ids = document_prompt(n_docs=3, doc_len=9, n_summary_tokens=N_SUMMARY_TOKENS)
    gm.model.set_summary_eval_mask_mode("causal")
    assert _forward_gap(gm, ids) > 1e-3


def test_document_chunked_has_no_train_serve_gap():
    """
    The chunked-document mask is reconstructed from the boundary tokens on every forward and does not
    consult ``self.training``, so both modes must agree.

    A gap here would mean the role reconstruction has become mode-dependent, which would make every
    document-chunked eval number incomparable to the loss it was trained against.
    """
    gm = build_document_chunked()
    ids = document_prompt(n_docs=3, doc_len=9)
    assert _forward_gap(gm, ids) < 1e-6


def test_document_chunked_mask_is_actually_applied():
    """
    Confirm the chunked mask changes the model's output at all.

    :func:`test_document_chunked_has_no_train_serve_gap` passes trivially if ``chunk_ids`` never
    reach the attention layers -- the variant falls back to ordinary causal attention when they are
    absent, by design. This distinguishes "no gap because the mask is stable" from "no gap because
    there is no mask", by checking the structured prompt is scored differently from the same tokens
    with the boundary markers stripped out.
    """
    gm = build_document_chunked()
    structured = document_prompt(n_docs=3, doc_len=9)

    # Same length and content, but with the boundary ids replaced by ordinary content tokens, so no
    # chunk roles can be derived and the layer falls through to causal attention.
    flat = structured.clone()
    flat[flat == IDS.doc_start] = 42
    flat[flat == IDS.doc_end] = 43

    a = torch.log_softmax(reference_forward(gm, structured)[0], dim=-1)
    b = torch.log_softmax(reference_forward(gm, flat)[0], dim=-1)
    # Compare only positions whose tokens are identical in both streams, so the difference cannot be
    # attributed to the swapped tokens themselves.
    same = (structured[0] == flat[0]).nonzero().squeeze(-1)
    kl = (a[same].exp() * (a[same] - b[same])).sum(-1)
    assert float(kl.max()) > 1e-3, (
        "stripping the document boundaries did not change the output, so chunk_ids are not reaching "
        "the attention layers and the document-chunked variant is running as plain causal attention."
    )
