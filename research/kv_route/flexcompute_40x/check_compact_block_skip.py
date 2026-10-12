"""Does the COMPACTED block skip compute the same thing as the masked one?

Compaction (``enable_block_skip(compact=True)``) gathers the kept tokens instead of masking them,
which is what turns block skipping into a wall-clock saving. It is only worth having if it agrees
with the masked path where it claims to, so this checks both claims in the module docstring:

* **plain-attention blocks: exact.** Same (query, key) pairs at the same RoPE positions, so the
  only difference should be kernel rounding.
* **GatedDeltaNet blocks: NOT exact.** The causal short convolution sees the previous KEPT tokens
  instead of zero-filled skipped slots. Reported, not asserted -- the point is to size it.

Runs in fp32 on the GPU when one is present (the hybrid's fla kernels are CUDA-only); the
dense case also runs on CPU.

    python research/kv_route/flexcompute_40x/check_compact_block_skip.py
"""

import torch

from olmo_core.nn.transformer import TransformerConfig

VOCAB = 512


def build(kind: str, n_layers: int = 4):
    if kind == "dense":
        cfg = TransformerConfig.llama2_271M(vocab_size=VOCAB, n_layers=n_layers)
    else:
        # tiny hybrid: 4 blocks, so the stack still contains both a GDN mixer and a full-attention
        # one (qwen3_5 puts attention every 4th block)
        cfg = TransformerConfig.qwen3_5_0_8B(vocab_size=VOCAB, n_layers=n_layers)
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cfg.build(init_device="cpu").to(dev)
    model.init_weights(max_seq_len=64, device=dev)
    model.train()
    return model


def run(model, ids, keep_frac: float, compact: bool):
    model.enable_block_skip(start_layer=0, target=0.5, compact=compact)
    holder = model._block_skip["holder"]
    B, T = ids.shape
    dev = next(model.parameters()).device
    keep = torch.zeros(B, T, dtype=torch.bool, device=dev)
    k = int(round(keep_frac * T))
    keep[:, torch.linspace(0, T - 1, k, device=dev).round().long().unique()] = True

    orig = holder.begin_forward

    def wrapped(*a, **kw):
        orig(*a, **kw)
        for li in model._block_skip["routed"]:
            holder._choice_cache[(int(li), B, T)] = keep

    holder.begin_forward = wrapped
    with torch.no_grad():
        out = model(ids.to(next(model.parameters()).device))
    return out, keep


def main():
    torch.manual_seed(0)
    T = 64
    ids = torch.randint(0, VOCAB, (1, T))

    for kind in ("dense", "hybrid"):
        try:
            m_mask = build(kind)
            m_comp = build(kind)
        except Exception as exc:  # a scale that will not build tiny on CPU
            print(f"{kind}: skipped ({type(exc).__name__}: {exc})")
            continue
        m_comp.load_state_dict(m_mask.state_dict())
        for frac in (1.0, 0.5, 0.25):
            a, keep = run(m_mask, ids, frac, compact=False)
            b, _ = run(m_comp, ids, frac, compact=True)
            sel = keep[0]
            da = a[0][sel].float()
            db = b[0][sel].float()
            rel = (da - db).abs().max() / da.abs().max().clamp(min=1e-9)
            # skipped rows must be untouched by both paths
            skipped = (a[0][~sel] - b[0][~sel]).abs().max() if (~sel).any() else torch.tensor(0.0)
            print(
                f"{kind:>7} keep={frac:<5} kept-rows rel max-err {rel:.3e}  "
                f"skipped-rows max-diff {float(skipped):.3e}"
            )


if __name__ == "__main__":
    main()
