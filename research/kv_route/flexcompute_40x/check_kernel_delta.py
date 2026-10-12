"""Attribute the gated-vs-dense loss difference: gather, or CE kernel swap?

After the unfused fallback, a gated step runs UNFUSED CE while the dense baseline runs FUSED, so
the two losses stop being bit-identical. This separates the two possible causes by measuring each
in isolation on one model:

  A. gather effect   -- dense vs gated, both forced to the SAME implementation
  B. kernel effect   -- dense fused vs dense unfused, no gathering at all

If A is ~0 and B matches the difference seen end-to-end, the gathering is still exact and the
residual is the kernel swap we chose deliberately.
"""

import torch

from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 248320


def main():
    device = torch.device("cuda")
    T, density = 8192, 0.0064
    n_keep = max(1, int(round(T * density)))
    torch.manual_seed(0)
    labels = torch.full((1, T), -100, dtype=torch.long, device=device)
    labels[:, -n_keep:] = torch.randint(0, VOCAB, (1, n_keep), device=device)
    input_ids = torch.randint(0, VOCAB, (1, T), device=device)

    cfg = TransformerConfig.qwen3_5_0_8B(vocab_size=VOCAB)
    torch.manual_seed(0)
    model = cfg.build(init_device="cpu").to(device)
    model.init_weights(max_seq_len=T, device=device)
    model = model.to(torch.bfloat16)
    model.eval()

    def loss_of(sparse, impl, max_rows):
        head = model.lm_head
        head._sparse_loss_positions = sparse
        head._loss_implementation = impl
        head._sparse_loss_unfused_max_rows = max_rows
        with torch.no_grad():
            out = model(input_ids, labels=labels, loss_reduction="sum", return_logits=False)
        return float(out.loss)

    # A. gather effect, same implementation on both sides (max_rows=0 disables the fallback)
    for impl in (LMLossImplementation.default, LMLossImplementation.fused_linear):
        d = loss_of(False, impl, 0)
        g = loss_of(True, impl, 0)
        print(f"A gather   {str(impl):>14}: dense={d:.6f} gated={g:.6f} rel={abs(g-d)/abs(d):.2e}")

    # B. kernel effect, no gathering on either side
    d_fused = loss_of(False, LMLossImplementation.fused_linear, 0)
    d_unfused = loss_of(False, LMLossImplementation.default, 0)
    print(f"B kernel   fused vs unfused (dense): {d_fused:.6f} vs {d_unfused:.6f} "
          f"rel={abs(d_unfused - d_fused)/abs(d_fused):.2e}")

    # End-to-end: what training actually does now.
    d = loss_of(False, LMLossImplementation.fused_linear, 4096)
    g = loss_of(True, LMLossImplementation.fused_linear, 4096)
    print(f"E2E        fused dense vs gated(auto-unfused): rel={abs(g-d)/abs(d):.2e}")


if __name__ == "__main__":
    main()
