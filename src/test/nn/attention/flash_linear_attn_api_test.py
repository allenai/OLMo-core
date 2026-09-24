import pytest
import torch

from olmo_core.nn.attention import KimiDeltaAttentionConfig
from olmo_core.nn.attention import flash_linear_attn_api as api
from olmo_core.nn.attention import kda


@pytest.mark.parametrize("entrypoint", ["build", "dispatch"])
@pytest.mark.parametrize(
    "missing,message",
    [
        ("package", "requires the kernel-fun package"),
        ("cuda13", "requires the CUDA 13 CuTe DSL"),
    ],
)
def test_experimental_kda_rejects_incompatible_installation(
    monkeypatch, entrypoint, missing, message
):
    # These errors must surface before importing or launching GPU kernels, even
    # on a machine that cannot run FLA. Cover layer construction and direct dispatch.
    monkeypatch.setattr(kda, "has_fla", lambda: True)
    monkeypatch.setattr(api, "has_fla", lambda: True)
    monkeypatch.setattr(api, "kernel_fun", None if missing == "package" else object())
    monkeypatch.setattr(api, "_has_cuda13_cute", False)
    with pytest.raises(RuntimeError, match=message):
        if entrypoint == "build":
            KimiDeltaAttentionConfig(n_heads=2, use_experimental_kernels=True).build(
                256, layer_idx=0, n_layers=1, init_device="meta"
            )
        else:
            x = torch.empty(0)
            api.dispatch_chunk_kda(
                q=x, k=x, v=x, g=x, beta=x, A_log=x, dt_bias=x, use_experimental_kernels=True
            )
