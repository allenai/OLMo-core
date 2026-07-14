"""Self-contained GPU smoke for the olmo-core FlexAttention monkeypatch.

Builds a real (random-init) olmo-core Qwen3-0.6B model on CUDA, synthesizes a tiny
batch of marker-wrapped input_ids, installs chunked / modified_swa FlexAttention, and
runs forward + backward. Verifies:
  - install_flex_attention patches every Attention module (n_patched > 0),
  - the pre-hook builds a BlockMask each forward (holder.block_mask is set),
  - forward produces finite logits and backward produces finite grads,
  - the masked forward differs from the stock (standard causal) forward — i.e. the
    custom mask is actually being applied, not silently ignored.

Needs no dataset, no checkpoint, and no network (config is code; tokenizer not used).
Run via jobs/smoke_olmo_flex.sh.
"""

import sys

import torch

from corpus_reasoning.lib.olmo_models import build_transformer_config, resolve_olmo_model
from corpus_reasoning.lib.olmo_flex_attention import install_flex_attention

BASE = "Qwen/Qwen3-0.6B-Base"
DS, DE, EOS = 1001, 1002, 151643  # synthetic marker ids + real Qwen3 eos
B, S = 2, 64


def _build_model():
    spec = resolve_olmo_model(BASE)
    cfg = build_transformer_config(spec)
    # Build directly on CUDA and randomly initialize (no meta/FSDP path needed for a
    # single-GPU forward smoke).
    model = cfg.build(init_device="cuda")
    model.init_weights(device=torch.device("cuda"))
    return model.eval()


def _synth_batch():
    """A tiny batch: [prefix] <DS> doc <DE> <DS> doc <DE> [query] <EOS> [pad...]."""
    torch.manual_seed(0)
    ids = torch.randint(10, 900, (B, S), dtype=torch.long)
    for b in range(B):
        ids[b, 0:3] = torch.tensor([20, 21, 22])      # instruction prefix
        ids[b, 3] = DS; ids[b, 8] = DE                # doc 0
        ids[b, 9] = DS; ids[b, 14] = DE               # doc 1
        ids[b, 50] = EOS                              # real terminator
        ids[b, 51:] = EOS                             # pad region (Qwen pad == eos)
    return ids.cuda()


def _forward_logits(model, ids):
    out = model(input_ids=ids)
    return out if isinstance(out, torch.Tensor) else out.logits


def main():
    model = _build_model()
    ids = _synth_batch()

    with torch.no_grad():
        base_logits = _forward_logits(model, ids).float()
    assert torch.isfinite(base_logits).all(), "stock forward produced non-finite logits"
    print(f"ok   stock forward: logits {tuple(base_logits.shape)} finite")

    for pattern, window in [("chunked", 0), ("modified_swa", 8)]:
        holder = install_flex_attention(
            model, pattern=pattern, doc_start_id=DS, doc_end_id=DE, eos_id=EOS,
            window=window, compile_flex=False,
        )
        assert holder.n_patched > 0, f"{pattern}: patched 0 attention modules"

        # forward + backward
        logits = _forward_logits(model, ids)
        assert holder.block_mask is not None, f"{pattern}: pre-hook did not set BlockMask"
        assert torch.isfinite(logits).all(), f"{pattern}: non-finite logits"
        loss = logits.float().log_softmax(-1).mean()
        loss.backward()
        g = next(p.grad for p in model.parameters() if p.grad is not None)
        assert torch.isfinite(g).all(), f"{pattern}: non-finite grad"

        # the mask must actually change the output vs stock causal
        delta = (logits.float() - base_logits).abs().max().item()
        assert delta > 1e-3, f"{pattern}: output identical to stock (mask not applied?)"
        print(f"ok   {pattern}: patched={holder.n_patched} blockmask=set "
              f"loss={loss.item():.4f} max|Δ vs stock|={delta:.3f}")
        model.zero_grad(set_to_none=True)

    print("\nSMOKE PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # surface the real error clearly in the slurm log
        import traceback
        traceback.print_exc()
        print(f"\nSMOKE FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
