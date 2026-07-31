"""
Validation for ``olmo_core.nn.attention.landmark_sparse_decode``.

The sparse decode must be a pure optimization: same selection, same grouped softmax, same alpha
reserve as the shipped dense-masked decode -- only cheaper. So the test is equality against
``FastLandmarkAttention._decode_one`` / ``FastCompressiveLandmarkAttention._decode_one`` fed the
GQA-expanded cache, across:

  * plain landmark and compressive landmark,
  * ``nonselected_landmark_mass`` 0 and 0.1,
  * several ``top_k`` (including ``None`` = dense gating and ``k >= n_blocks``),
  * eval mode ("one long local block" for generated tokens) and per-block prompt decode,
  * landmark-position and content-position queries,
  * GQA (32 q-heads / 8 KV heads) and MHA (n_rep = 1).

    python debug/sparse_landmark_inference/test_sparse_decode.py
"""

import sys

import torch

from olmo_core.nn.attention.landmark import repeat_kv
from olmo_core.nn.attention.landmark_compressive import FastCompressiveLandmarkAttention
from olmo_core.nn.attention.landmark_fast import FastLandmarkAttention
from olmo_core.nn.attention.landmark_sparse_decode import sparse_landmark_decode

DEV = "cuda"
DTYPE = torch.bfloat16
TOL = 3e-2


def _mk_attn(compressive, Lb, scale, top_k, alpha, prompt_len):
    cls = FastCompressiveLandmarkAttention if compressive else FastLandmarkAttention
    attn = cls.__new__(cls)
    attn.block_size = Lb
    attn.mem_freq = Lb - 1
    attn.softmax_scale = scale
    attn._eval_prompt_len = prompt_len
    attn._eval_decode_mode = "extend_last_block"
    attn._eval_top_k = top_k
    attn._ragged_qpos = None
    if compressive:
        attn.nonselected_landmark_mass = alpha
    return attn


def run_case(*, B, H, H_kv, D, Lb, total, qpos, prompt_len, compressive, alpha, top_k, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(B, H, 1, D, generator=g, device=DEV, dtype=DTYPE)
    kc = torch.randn(B, H_kv, total, D, generator=g, device=DEV, dtype=DTYPE)
    vc = torch.randn(B, H_kv, total, D, generator=g, device=DEV, dtype=DTYPE)
    scale = D**-0.5
    n_rep = H // H_kv

    attn = _mk_attn(compressive, Lb, scale, top_k, alpha, prompt_len)
    ref = attn._decode_one(q, repeat_kv(kc, n_rep), repeat_kv(vc, n_rep), qpos)

    # Mirror the section/total rules the shipped decode applies for this qpos.
    if prompt_len is not None and qpos >= prompt_len:
        section_start = (prompt_len // Lb) * Lb
        eff_total = total
    else:
        eff_total = qpos if qpos % Lb == Lb - 1 else total
        section_start = (qpos // Lb) * Lb

    got = sparse_landmark_decode(
        q,
        kc,
        vc,
        block_size=Lb,
        softmax_scale=scale,
        section_start=section_start,
        total=eff_total,
        top_k=top_k,
        compressive=compressive,
        nonselected_mass=alpha if compressive else 0.0,
    )
    err = (got.float() - ref.float()).abs().max().item()
    return err


def main():
    ok = True
    Lb, D = 64, 128
    cases = []
    # --- eval mode (generated token, one long local block) ---
    for compressive, alpha in [(False, 0.0), (True, 0.0), (True, 0.1)]:
        for top_k in [1, 4, 12, None, 999]:
            cases.append(
                dict(
                    B=1,
                    H=32,
                    H_kv=8,
                    D=D,
                    Lb=Lb,
                    total=1088,
                    qpos=1087,
                    prompt_len=1024,
                    compressive=compressive,
                    alpha=alpha,
                    top_k=top_k,
                    label=f"eval GQA compressive={compressive} a={alpha} k={top_k}",
                )
            )
    # --- per-block prompt decode: content-position and landmark-position queries ---
    for qpos, tag in [(700, "content-pos"), (Lb * 11 - 1, "landmark-pos")]:
        for compressive in (False, True):
            cases.append(
                dict(
                    B=1,
                    H=32,
                    H_kv=8,
                    D=D,
                    Lb=Lb,
                    total=qpos + 1,
                    qpos=qpos,
                    prompt_len=None,
                    compressive=compressive,
                    alpha=0.1,
                    top_k=3,
                    label=f"prompt {tag} compressive={compressive}",
                )
            )
    # --- MHA (n_rep = 1) and batch > 1 ---
    cases.append(
        dict(
            B=1,
            H=8,
            H_kv=8,
            D=D,
            Lb=Lb,
            total=1088,
            qpos=1087,
            prompt_len=1024,
            compressive=True,
            alpha=0.1,
            top_k=4,
            label="eval MHA (n_rep=1)",
        )
    )
    cases.append(
        dict(
            B=3,
            H=32,
            H_kv=8,
            D=D,
            Lb=Lb,
            total=1088,
            qpos=1087,
            prompt_len=1024,
            compressive=True,
            alpha=0.1,
            top_k=4,
            label="eval batch=3",
        )
    )

    for i, c in enumerate(cases):
        label = c.pop("label")
        err = run_case(seed=i, **c)
        passed = err <= TOL
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {label}: max_abs_err={err:.3e}")
    print("ALL PASS" if ok else "FAILURES ABOVE")
    return ok


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA device")
    sys.exit(0 if main() else 1)
