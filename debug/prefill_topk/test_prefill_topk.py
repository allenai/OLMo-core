"""
Validation for ``olmo_core.nn.attention.landmark_prefill_topk`` (prefill-wide top-k landmark
retrieval). Run on a GPU node with the corpus-reasoning-olmo env:

    python debug/prefill_topk/test_prefill_topk.py

Checks (all on random tensors, no checkpoint needed):

1. **Dense equivalence** -- with ``top_k=None`` the eager implementation reproduces the fused Triton
   prefill kernel (plain landmark and compressive landmark), i.e. the new code path is the same
   attention the model was trained/evaluated with when top-k is off.
2. **Decode agreement** -- for the LAST query position, the eager per-query top-k output matches what
   the existing decode path (``_decode_one`` with ``set_landmark_eval_decode(top_k=...)``) produces
   for that same position, across several ``top_k`` values and with/without the compressive
   ``nonselected_landmark_mass``. This pins the new prefill semantics to the already-shipped decode
   semantics.
3. **Monotone limit** -- ``top_k >= n_blocks`` collapses back to the dense result.
"""

import sys

import torch

from olmo_core.nn.attention.landmark_compressive import fused_compressive_landmark_attention
from olmo_core.nn.attention.landmark_fast import fused_landmark_attention_fast
from olmo_core.nn.attention.landmark_prefill_topk import landmark_topk_prefill_attention

DEV = "cuda"
DTYPE = torch.bfloat16


def _mk(B, H, T, D, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(B, H, T, D, generator=g, device=DEV, dtype=DTYPE)
    k = torch.randn(B, H, T, D, generator=g, device=DEV, dtype=DTYPE)
    v = torch.randn(B, H, T, D, generator=g, device=DEV, dtype=DTYPE)
    return q, k, v


def _report(name, a, b, tol):
    err = (a.float() - b.float()).abs().max().item()
    rel = err / max(b.float().abs().max().item(), 1e-6)
    ok = err <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: max_abs_err={err:.3e} (rel {rel:.2e}, tol {tol})")
    return ok


def test_dense_equivalence():
    print("[1] dense equivalence vs the fused Triton prefill kernel (top_k=None)")
    ok = True
    B, H, T, D, Lb = 1, 4, 512, 64, 64
    q, k, v = _mk(B, H, T, D, seed=1)
    scale = D**-0.5
    is_mem = (torch.arange(T, device=DEV) % Lb) == (Lb - 1)

    ref = fused_landmark_attention_fast(q, k, v, is_mem, sm_scale=scale, block_size=Lb)
    got = landmark_topk_prefill_attention(
        q, k, v, block_size=Lb, softmax_scale=scale, top_k=None, compressive=False
    )
    ok &= _report("plain landmark", got, ref, 3e-2)

    ref_c = fused_compressive_landmark_attention(q, k, v, is_mem, sm_scale=scale, block_size=Lb)
    got_c = landmark_topk_prefill_attention(
        q, k, v, block_size=Lb, softmax_scale=scale, top_k=None, compressive=True
    )
    ok &= _report("compressive landmark", got_c, ref_c, 3e-2)

    # top_k >= n_blocks must collapse to dense.
    got_k = landmark_topk_prefill_attention(
        q, k, v, block_size=Lb, softmax_scale=scale, top_k=T // Lb, compressive=True
    )
    ok &= _report("compressive, top_k = n_blocks == dense", got_k, ref_c, 3e-2)
    return ok


def _decode_reference(q, k, v, *, Lb, scale, top_k, compressive, alpha, qpos=None):
    """Run the SHIPPED decode path for query position ``qpos`` and return its output."""
    from olmo_core.nn.attention.landmark_compressive import FastCompressiveLandmarkAttention
    from olmo_core.nn.attention.landmark_fast import FastLandmarkAttention

    cls = FastCompressiveLandmarkAttention if compressive else FastLandmarkAttention
    attn = cls.__new__(cls)  # bypass __init__: _decode_one only needs these attributes
    attn.block_size = Lb
    attn.mem_freq = Lb - 1
    attn.softmax_scale = scale
    attn._eval_prompt_len = None
    attn._eval_decode_mode = "extend_last_block"
    attn._eval_top_k = top_k
    attn._ragged_qpos = None
    if compressive:
        attn.nonselected_landmark_mass = alpha
    if qpos is None:
        qpos = q.shape[2] - 1
    # The decode sees only the cache up to and including its own position.
    return attn._decode_one(
        q[:, :, qpos : qpos + 1], k[:, :, : qpos + 1], v[:, :, : qpos + 1], qpos
    )


def test_decode_agreement():
    print("[2] per-query top-k == the shipped decode top-k, at several query positions")
    ok = True
    B, H, T, D, Lb = 1, 8, 1024, 64, 64
    n_blocks = T // Lb
    q, k, v = _mk(B, H, T, D, seed=2)
    scale = D**-0.5
    # last position (a landmark row), a mid-sequence landmark row, a mid-sequence content row, and
    # an early row whose past-block count is below top_k (must degrade to "keep all past blocks").
    positions = [T - 1, 5 * Lb - 1, 7 * Lb + 13, Lb + 3]

    for compressive, alpha in [(False, 0.0), (True, 0.0), (True, 0.1)]:
        for top_k in [1, 2, 4, n_blocks]:
            full = landmark_topk_prefill_attention(
                q,
                k,
                v,
                block_size=Lb,
                softmax_scale=scale,
                top_k=top_k,
                compressive=compressive,
                nonselected_mass=alpha,
            )
            worst, worst_pos = 0.0, -1
            for qpos in positions:
                ref = _decode_reference(
                    q,
                    k,
                    v,
                    Lb=Lb,
                    scale=scale,
                    top_k=top_k,
                    compressive=compressive,
                    alpha=alpha,
                    qpos=qpos,
                )
                err = (full[:, :, qpos : qpos + 1].float() - ref.float()).abs().max().item()
                if err > worst:
                    worst, worst_pos = err, qpos
            tag = f"compressive={compressive} alpha={alpha} top_k={top_k} (worst @ pos {worst_pos})"
            passed = worst <= 3e-2
            ok &= passed
            print(f"  {'PASS' if passed else 'FAIL'}  {tag}: max_abs_err={worst:.3e}")
    return ok


def test_tiling_invariance():
    print("[3] query_tile does not change the result")
    B, H, T, D, Lb = 1, 4, 512, 64, 64
    q, k, v = _mk(B, H, T, D, seed=3)
    scale = D**-0.5
    a = landmark_topk_prefill_attention(
        q, k, v, block_size=Lb, softmax_scale=scale, top_k=2, compressive=True, query_tile=64
    )
    b = landmark_topk_prefill_attention(
        q, k, v, block_size=Lb, softmax_scale=scale, top_k=2, compressive=True, query_tile=512
    )
    return _report("tile 64 vs 512", a, b, 0.0)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA device (the reference kernels are Triton)")
    all_ok = True
    all_ok &= test_dense_equivalence()
    all_ok &= test_decode_agreement()
    all_ok &= test_tiling_invariance()
    print("ALL PASS" if all_ok else "FAILURES ABOVE")
    sys.exit(0 if all_ok else 1)
