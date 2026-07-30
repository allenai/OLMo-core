"""Side-by-side check: our s2_attn vs LongLoRA paper's reference S^2-Attn.

The paper's recipe (Section 3.1, "Shift Short Attention" / S^2-Attn):
  - Split heads in half along the head dim.
  - Roll the back half of heads along the sequence by -G/2.
  - Reshape the sequence into n_g = T/G groups along a new batch axis.
  - Run causal SDPA inside each group.
  - Reshape back and un-roll the back half by +G/2.

We re-implement this from the formula and check that our `s2_attn` (which is
the HF-wrapper version registered into ALL_ATTENTION_FUNCTIONS) produces
matching output on a fixed random input.
"""

import torch
import torch.nn.functional as F


def reference_s2_attn(q, k, v, G):
    """Pure-tensor reference: q,k,v are [B, H, T, D], assumes H even, T%G==0.

    Returns [B, H, T, D].
    """
    B, H, T, D = q.shape
    assert H % 2 == 0
    assert T % G == 0
    assert T >= 2 * G
    half = H // 2
    shift = G // 2
    n_g = T // G

    # Roll back-half heads by -G/2 along seq.
    q2 = q.clone(); q2[:, half:] = torch.roll(q2[:, half:], shifts=-shift, dims=2)
    k2 = k.clone(); k2[:, half:] = torch.roll(k2[:, half:], shifts=-shift, dims=2)
    v2 = v.clone(); v2[:, half:] = torch.roll(v2[:, half:], shifts=-shift, dims=2)

    # Reshape to groups: [B,H,T,D] -> [B*n_g, H, G, D]
    qg = q2.view(B, H, n_g, G, D).transpose(1, 2).reshape(B * n_g, H, G, D)
    kg = k2.view(B, H, n_g, G, D).transpose(1, 2).reshape(B * n_g, H, G, D)
    vg = v2.view(B, H, n_g, G, D).transpose(1, 2).reshape(B * n_g, H, G, D)

    # Causal SDPA per group.
    og = F.scaled_dot_product_attention(qg, kg, vg, is_causal=True)

    # Reshape back: [B*n_g, H, G, D] -> [B, H, T, D]
    out = og.view(B, n_g, H, G, D).transpose(1, 2).reshape(B, H, T, D)

    # Un-roll back-half heads by +G/2.
    out_un = out.clone()
    out_un[:, half:] = torch.roll(out[:, half:], shifts=+shift, dims=2)

    return out_un


def test_s2_attn_matches_reference():
    """Drive our HF-wrapped s2_attn the same as the trainer does; compare to ref."""
    import sys
    # sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
    from corpus_reasoning.lib.longlora_attn import install_s2_attn
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    torch.manual_seed(0)
    B, H, T, D, G = 2, 8, 32, 4, 8        # 32/8 = 4 groups, 2G=16 <= T
    q = torch.randn(B, H, T, D, dtype=torch.float32)
    k = torch.randn(B, H, T, D, dtype=torch.float32)
    v = torch.randn(B, H, T, D, dtype=torch.float32)

    # Reference output.
    ref = reference_s2_attn(q, k, v, G)

    # Our s2_attn: requires a stub Module with training=True (or force_eval=True).
    class StubConfig:
        _attn_implementation = "s2_attn"

    class StubModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = StubConfig()
            self._attn_implementation = "s2_attn"
        def modules(self):
            return [self]
    model = StubModule()
    model.train()
    install_s2_attn(model, group_size=G)
    s2 = ALL_ATTENTION_FUNCTIONS["s2_attn"]
    out, _ = s2(model, q, k, v, attention_mask=None)
    # The HF wrapper convention returns [B, T, H, D]; transpose to compare.
    if out.dim() == 4 and out.shape[1] == T and out.shape[2] == H:
        out_bhtd = out.transpose(1, 2).contiguous()
    else:
        out_bhtd = out

    max_diff = (out_bhtd - ref).abs().max().item()
    print(f"reference vs s2_attn: max abs diff = {max_diff:.3e}")
    assert max_diff < 1e-5, f"S^2-Attn outputs disagree (max diff {max_diff})"
    print("PASS: s2_attn matches the paper-recipe reference.")


if __name__ == "__main__":
    test_s2_attn_matches_reference()
