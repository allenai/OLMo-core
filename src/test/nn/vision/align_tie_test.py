"""Why an all-text microbatch still keeps the vision FSDP collectives symmetric.

Background. ``MultimodalCollator`` always emits an images tensor: a microbatch with no
image rows still gets one dummy zero crop whose ``pooled_patches_idx`` are all ``-1``, so
``valid_rows`` is all-False, ``image_features`` is ``(0, d)`` and nothing is spliced. The
concern raised in review was that such a rank would then run no connector/ViT backward
while its peers did, issuing fewer FSDP collectives and deadlocking the DP group — which
is a real FSDP2 behaviour: ``FSDPParamGroup.post_backward`` early-returns without a
reduce-scatter when no parameter in the group has a gradient
(``if len(fsdp_params_with_grad) == 0: return``), and the root post-backward callback that
covers "backward did not run" groups calls that same method. A 2-process gloo repro
confirms the general mechanism: give one rank a genuinely disconnected vision output and
``backward()`` hangs.

**But this model does not hit it.** The splice
``flat[is_image_patch] = flat[is_image_patch] + image_features.reshape(-1, d)`` runs
whenever ``images is not None`` — which the collator guarantees — and a masked
``index_put`` with an all-False mask still builds an autograd node. Backward therefore
reaches the connector and ViT with an *empty* gradient, and their weight gradients come
out present-but-zero. FSDP sees a non-empty ``fsdp_params_with_grad`` and issues the
reduce-scatter on every rank.

So ``image_align_tie`` is redundant for the current forward, and these tests pin the
property that makes it redundant. If the splice is ever changed such that an all-text
microbatch leaves the vision parameters with ``None`` gradients, these tests fail — and at
that point the tie becomes load-bearing again and must not be disabled.

Caveat this cannot cover: production runs use ``torch.compile``. Whether inductor
preserves the edge through an empty ``index_put`` is not exercised here (no GPU), so a
compiled 2-rank check is still the outstanding validation before relying on the tie being
optional.
"""

from __future__ import annotations

from test.nn.vision.multimodal_test import (
    _IMAGE_PATCH_TOKEN,
    _LM_VOCAB,
    _make_inputs,
    _tiny_multimodal_cfg,
)

import pytest
import torch


def _all_text_batch(batch: int = 2, seq_len: int = 16, n_patches_per_crop: int = 4):
    """A microbatch with no image rows, shaped the way the collator shapes one."""
    input_ids = torch.randint(2, _LM_VOCAB, (batch, seq_len))
    assert (input_ids != _IMAGE_PATCH_TOKEN).all(), "this batch must contain no image tokens"
    images = torch.zeros(batch, 1, n_patches_per_crop, 14 * 14 * 3)
    pooled_patches_idx = torch.full((batch, 1, n_patches_per_crop), -1, dtype=torch.long)
    return input_ids, images, pooled_patches_idx


def _backward_and_collect(cfg, inputs):
    model = cfg.build(init_device="cpu")
    input_ids, images, pooled_patches_idx = inputs
    out = model(input_ids, images=images, pooled_patches_idx=pooled_patches_idx)
    logits = out.logits if hasattr(out, "logits") else out
    logits.float().sum().backward()
    return model


def _grad_summary(module):
    grads = [p.grad for p in module.parameters()]
    n_present = sum(g is not None for g in grads)
    magnitude = sum(0.0 if g is None else float(g.abs().sum()) for g in grads)
    return n_present, len(grads), magnitude


@pytest.mark.parametrize("image_align_tie", [True, False])
def test_all_text_batch_still_gives_vision_params_gradients(image_align_tie):
    """The invariant that keeps FSDP symmetric — and it holds with the tie off too.

    ``None`` gradients are what makes FSDP skip a group's reduce-scatter. Present-but-zero
    gradients still issue it, so every DP rank makes the same number of collectives
    regardless of how text-only vs image examples landed across ranks.
    """
    cfg = _tiny_multimodal_cfg()
    cfg.image_align_tie = image_align_tie
    model = _backward_and_collect(cfg, _all_text_batch())

    for name, module in (("connector", model.connector), ("vision", model.vision)):
        n_present, n_total, magnitude = _grad_summary(module)
        assert n_present == n_total, (
            f"{name}: {n_total - n_present} of {n_total} params have a None gradient on an "
            "all-text microbatch. FSDP skips the reduce-scatter for a group with no "
            "gradients, so this rank would issue fewer collectives than a peer whose "
            "microbatch had image rows, and the DP group would deadlock. If this fails, "
            "image_align_tie is load-bearing again and must stay on."
        )
        # Nothing was spliced, so the gradient is exactly zero either way.
        assert magnitude == pytest.approx(0.0), name


def test_tie_does_not_change_the_gradient_on_an_all_text_batch():
    """The tie contributes exactly 0, so turning it on cannot perturb training."""
    inputs = _all_text_batch()
    torch.manual_seed(0)
    on = _backward_and_collect(_with_tie(True), inputs)
    torch.manual_seed(0)
    off = _backward_and_collect(_with_tie(False), inputs)

    for p_on, p_off in zip(on.connector.parameters(), off.connector.parameters()):
        assert torch.equal(p_on.grad, p_off.grad)


def test_image_batch_gives_nonzero_vision_grads():
    """Control: the all-text zeros above are the batch's doing, not a broken hookup."""
    model = _backward_and_collect(_with_tie(True), _make_inputs(2, 16))
    for name, module in (("connector", model.connector), ("vision", model.vision)):
        n_present, n_total, magnitude = _grad_summary(module)
        assert n_present == n_total, name
        assert magnitude > 0.0, name


def test_align_tie_defaults_on():
    """Default remains on. It is redundant for the current forward (see module docstring)
    but costs nothing correctness-wise, and the compiled-mode check is outstanding."""
    assert _tiny_multimodal_cfg().image_align_tie is True


def _with_tie(value: bool):
    cfg = _tiny_multimodal_cfg()
    cfg.image_align_tie = value
    return cfg
