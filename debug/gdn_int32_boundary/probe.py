"""
Does the GDN chunk kernel break past 262,144 tokens in ONE call, and does a chunked call survive?

Hypothesis (2026-09-30): the chunk kernel's per-chunk state tensor holds ceil(T/64) * H * K * V
values; with Qwen3.5-4B's H=32 value heads and K=V=128 that is 2^31 exactly at T = 262,144, so a
single call on a longer sequence overflows int32 indexing ("illegal memory access"). Every YaRN
r256k eval job (prompts to ~272k) died that way on 7 different nodes.

Each case runs in its own process (an illegal access poisons the CUDA context):

    python debug/gdn_int32_boundary/probe.py            # all cases
    python debug/gdn_int32_boundary/probe.py one 262144 # one case
"""

import subprocess
import sys

H, K, V = 32, 128, 128


def run(mode: str, T: int) -> None:
    import torch

    from olmo_core.nn.attention.flash_linear_attn_api import (
        dispatch_chunk_gated_delta_rule,
    )

    torch.manual_seed(0)
    dev, bf = "cuda", torch.bfloat16

    def inputs(t):
        q = torch.randn(1, t, H, K, device=dev, dtype=bf)
        k = torch.randn(1, t, H, K, device=dev, dtype=bf)
        v = torch.randn(1, t, H, V, device=dev, dtype=bf)
        g = -torch.rand(1, t, H, device=dev, dtype=torch.float32) * 0.1
        beta = torch.rand(1, t, H, device=dev, dtype=bf).sigmoid()
        return q, k, v, g, beta

    if mode == "one":
        o, s = dispatch_chunk_gated_delta_rule(
            *inputs(T), output_final_state=True, use_qk_l2norm_in_kernel=True
        )
    else:  # two chunks, the second continuing from the first's final state
        half = T // 2
        _, s = dispatch_chunk_gated_delta_rule(
            *inputs(half), output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        o, s = dispatch_chunk_gated_delta_rule(
            *inputs(T - half),
            initial_state=s,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    torch.cuda.synchronize()
    print(f"OK {mode} T={T:,}  finite={bool(torch.isfinite(o).all())}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run(sys.argv[1], int(sys.argv[2]))
        sys.exit(0)
    for mode, T in [
        ("one", 262144 - 64),
        ("one", 262144),
        ("one", 262144 + 64),
        ("one", 272000),
        ("two", 272000),
    ]:
        r = subprocess.run([sys.executable, __file__, mode, str(T)], capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()
        msg = next((ln for ln in tail if ln.startswith("OK")), None) or next(
            (ln for ln in reversed(tail) if "Error" in ln or "illegal" in ln),
            tail[-1] if tail else "?",
        )
        print(f"{mode:>3} T={T:>7,}  rc={r.returncode}  {msg[:160]}", flush=True)
