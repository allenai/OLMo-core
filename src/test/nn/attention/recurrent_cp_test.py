"""
Correctness tests for Ulysses context parallelism (CP) in the recurrent sequence mixers,
:class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet` and
:class:`~olmo_core.nn.attention.kda.KimiDeltaAttention`.

Both layers implement CP the same way: each rank holds a contiguous slice of the sequence for
the per-token projections, an all-to-all switches to head parallelism so the causal convolutions
and the recurrent kernel see the full sequence for ``heads / cp_degree`` heads, and a second
all-to-all switches back. Per-channel and per-head parameters that are consumed after the
exchange (the conv filters, and for KDA the in-kernel gate parameters) are kept whole on every
rank and sliced to the rank's heads in ``forward``.

The tests here check that contract at three levels:

1. **Slice consistency (CPU, no kernels):** the head slice, gate slice, and conv channel slices a
   rank uses must describe exactly the heads that the all-to-all delivers to that rank.
2. **Forward parity (multi-GPU):** each rank's local output must match the corresponding slice of
   the single-process, full-sequence output, with and without packed-document boundaries that
   straddle the CP split.
3. **Backward parity (multi-GPU):** with a loss that is a sum over tokens, gradients summed over
   the CP group must match the full-sequence gradients for every parameter and for the input.
   Parameters that are sliced per rank must additionally carry zero gradient outside the rank's
   slice, which is what makes the data-parallel gradient reduction correct in training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.nn.attention.kda import KimiDeltaAttention
from olmo_core.nn.attention.recurrent import GatedDeltaNet
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.testing import run_distributed_test
from olmo_core.testing.utils import requires_fla, requires_multi_gpu
from olmo_core.utils import get_default_device, seed_all

D_MODEL = 128
N_HEADS = 8


@dataclass
class Tolerance:
    """
    ``rel_fro`` bounds the relative Frobenius error; ``rtol``/``atol`` bound the max abs error
    relative to the largest magnitude in the expected tensor.
    """

    rel_fro: float
    rtol: float
    atol: float


# Forward activations are bf16 under autocast; gradients are accumulated in fp32 but flow through
# bf16 activations, and the CP path sums per-rank contributions in a different order.
FWD_TOL = Tolerance(rel_fro=2e-2, rtol=5e-2, atol=1e-2)
GRAD_TOL = Tolerance(rel_fro=3e-2, rtol=5e-2, atol=1e-4)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, tol: Tolerance):
    actual = actual.detach().float()
    expected = expected.detach().float()
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    diff = actual - expected
    rel_fro = (diff.norm() / expected.norm().clamp_min(1e-6)).item()
    max_abs = diff.abs().max().item()
    scale = expected.abs().max().clamp_min(1e-6).item()
    assert rel_fro <= tol.rel_fro and max_abs <= tol.atol + tol.rtol * scale, (
        f"{name}: rel_fro={rel_fro:.3e} (tol {tol.rel_fro:.1e}), "
        f"max_abs={max_abs:.3e} vs allowed {tol.atol + tol.rtol * scale:.3e} (scale {scale:.3e})"
    )


@dataclass
class MixerSpec:
    name: str
    kwargs: Dict[str, Any]

    def build(self, device: str, **overrides: Any):
        kwargs = {**self.kwargs, **overrides}
        if self.name == "gdn":
            return GatedDeltaNet(init_device=device, **kwargs)
        elif self.name == "kda":
            return KimiDeltaAttention(init_device=device, **kwargs)
        raise NotImplementedError(self.name)


MIXERS: Dict[str, MixerSpec] = {
    "gdn": MixerSpec("gdn", {"d_model": D_MODEL, "n_heads": N_HEADS}),
    "kda": MixerSpec("kda", {"d_model": D_MODEL, "n_heads": N_HEADS, "allow_neg_eigval": True}),
}


@dataclass
class Case:
    batch_size: int
    seq_len: int
    doc_lens: Optional[List[int]] = None
    """Document lengths that sum to ``seq_len``; packed inputs require ``batch_size == 1``."""
    overrides: Dict[str, Any] = field(default_factory=dict)
    mixers: tuple = ("gdn", "kda")

    def cu_doc_lens(self, device) -> Optional[torch.Tensor]:
        if self.doc_lens is None:
            return None
        assert self.batch_size == 1 and sum(self.doc_lens) == self.seq_len
        return torch.tensor(
            [0, *torch.tensor(self.doc_lens).cumsum(0).tolist()], dtype=torch.int32, device=device
        )


CASES: Dict[str, Case] = {
    # Two instances, no document boundaries.
    "batch2": Case(batch_size=2, seq_len=128),
    # Packed documents whose boundaries (100, 156) do not line up with the CP split at
    # 128 (degree 2) or 64/128/192 (degree 4), so a document straddles ranks.
    "docs": Case(batch_size=1, seq_len=256, doc_lens=[100, 56, 100]),
    # Several kernel chunks per document, boundaries again off the CP split.
    "long_docs": Case(batch_size=1, seq_len=1024, doc_lens=[300, 400, 324]),
    # Grouped value heads: q/k heads are repeated onto v heads after the exchange.
    "gva": Case(batch_size=1, seq_len=128, overrides={"n_v_heads": 2 * N_HEADS}, mixers=("gdn",)),
    # The kernel-fun path only engages without packed documents; elsewhere it falls back to FLA.
    "experimental": Case(
        batch_size=1, seq_len=512, overrides={"use_experimental_kernels": True}, mixers=("kda",)
    ),
}


def _init_real_weights(module, seed: int = 777):
    """Real init so decays are in the strong regime; ``torch.empty`` leaves A_log as garbage."""
    generator = torch.Generator(device=module.A_log.device).manual_seed(seed)
    with torch.no_grad():
        module.init_weights(
            init_method=InitMethod.normal,
            d_model=module.d_model,
            block_idx=0,
            num_blocks=2,
            generator=generator,
        )


def _expected_param_slices(module, cp_rank: int, cp_world_size: int) -> Dict[str, slice]:
    """
    The dim-0 slice of every parameter that a rank should touch under Ulysses CP. Anything
    outside these slices must receive zero gradient on that rank.
    """
    slices: Dict[str, slice] = {}
    for conv_name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        conv = getattr(module, conv_name)
        local = conv.hidden_size // cp_world_size
        conv_slice = slice(cp_rank * local, (cp_rank + 1) * local)
        slices[f"{conv_name}.weight"] = conv_slice
        if conv.bias is not None:
            slices[f"{conv_name}.bias"] = conv_slice
    if isinstance(module, KimiDeltaAttention):
        local_heads = module.n_heads // cp_world_size
        slices["A_log"] = slice(cp_rank * local_heads, (cp_rank + 1) * local_heads)
        local_gate = module.gate_dim // cp_world_size
        slices["dt_bias"] = slice(cp_rank * local_gate, (cp_rank + 1) * local_gate)
    return slices


def _fake_mesh(cp_world_size: int, cp_rank: int) -> MagicMock:
    mesh = MagicMock()
    mesh.size.return_value = cp_world_size
    mesh.get_local_rank.return_value = cp_rank
    return mesh


# ---------------------------------------------------------------------------------------------
# Level 1: slice consistency, CPU only.
# ---------------------------------------------------------------------------------------------


@requires_fla
@pytest.mark.parametrize("mixer_name", list(MIXERS))
@pytest.mark.parametrize("cp_world_size", [2, 4])
@pytest.mark.parametrize("gva", [False, True], ids=["", "gva"])
def test_ulysses_cp_slices_match_head_partition(mixer_name: str, cp_world_size: int, gva: bool):
    """
    ``all_to_all_cp2hp`` hands rank ``r`` the contiguous channel block ``[r*C/CP, (r+1)*C/CP)``
    of every exchanged tensor. Every slice the layer takes after the exchange must describe
    exactly the heads inside that block.
    """
    if gva and mixer_name != "gdn":
        pytest.skip("only GatedDeltaNet supports grouped value heads")
    overrides = {"n_v_heads": 2 * N_HEADS} if gva else {}
    for cp_rank in range(cp_world_size):
        module = MIXERS[mixer_name].build("meta", **overrides)
        module.apply_cp(_fake_mesh(cp_world_size, cp_rank), uly=UlyssesContextParallelStyle())
        assert module.cp_enabled

        expected = _expected_param_slices(module, cp_rank, cp_world_size)
        for conv_name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            conv = getattr(module, conv_name)
            assert conv.cp_enabled
            assert conv._cp_channel_slice == expected[f"{conv_name}.weight"], conv_name

        # Heads covered by the q/k conv channel block on this rank.
        qk_channels = range(*expected["q_conv1d.weight"].indices(module.key_dim))
        qk_heads = sorted({c // module.head_k_dim for c in qk_channels})
        v_channels = range(*expected["v_conv1d.weight"].indices(module.value_dim))
        v_heads = sorted({c // module.head_v_dim for c in v_channels})

        n_local = module.n_heads // cp_world_size
        assert qk_heads == list(range(cp_rank * n_local, (cp_rank + 1) * n_local))
        if isinstance(module, KimiDeltaAttention):
            assert module._cp_head_slice == expected["A_log"]
            assert module._cp_gate_slice == expected["dt_bias"]
            assert list(range(*module._cp_head_slice.indices(module.n_heads))) == qk_heads
            # dt_bias is per key channel, so its slice must equal the q/k conv channel block.
            assert module._cp_gate_slice == expected["q_conv1d.weight"]
            assert v_heads == qk_heads
        else:
            # GDN repeats q/k head ``h`` onto v heads ``[h*rep, (h+1)*rep)``; the v channel block
            # this rank receives must be exactly those heads.
            rep = module.n_v_heads // module.n_heads
            assert v_heads == [h * rep + j for h in qk_heads for j in range(rep)]


@requires_fla
@pytest.mark.parametrize("mixer_name", list(MIXERS))
def test_ulysses_cp_degree_one_is_noop(mixer_name: str):
    module = MIXERS[mixer_name].build("meta")
    module.apply_cp(_fake_mesh(1, 0), uly=UlyssesContextParallelStyle())
    assert not module.cp_enabled
    for conv_name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        assert not getattr(module, conv_name).cp_enabled


@requires_fla
@pytest.mark.parametrize("mixer_name", list(MIXERS))
def test_ulysses_cp_rejects_ring(mixer_name: str):
    """Ring attention has no meaning for a recurrent scan, so ``apply_cp`` must reject it."""
    module = MIXERS[mixer_name].build("meta")
    with pytest.raises(NotImplementedError, match="Ring"):
        module.apply_cp(_fake_mesh(2, 0), ring=RingContextParallelStyle())


@requires_fla
@pytest.mark.parametrize("mixer_name", list(MIXERS))
@pytest.mark.parametrize("n_heads,cp_world_size", [(8, 3), (6, 4)])
def test_ulysses_cp_requires_divisible_heads(mixer_name: str, n_heads: int, cp_world_size: int):
    module = MIXERS[mixer_name].build("meta", d_model=n_heads * 16, n_heads=n_heads)
    with pytest.raises((ValueError, AssertionError)):
        module.apply_cp(_fake_mesh(cp_world_size, 0), uly=UlyssesContextParallelStyle())


# ---------------------------------------------------------------------------------------------
# Levels 2 and 3: forward and backward parity against a single-process reference, multi-GPU.
# ---------------------------------------------------------------------------------------------


def _build_reference(
    mixer_name: str, case_name: str, ref_path: Path, checkpoint_dir: Path, device: torch.device
):
    """Run the full sequence in one process and save inputs, outputs, and gradients."""
    spec, case = MIXERS[mixer_name], CASES[case_name]
    seed_all(0)
    module = spec.build(device.type, **case.overrides)
    _init_real_weights(module)

    x = torch.randn(
        case.batch_size, case.seq_len, D_MODEL, device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    # A fixed random weight makes the loss a plain sum over tokens, so it splits exactly across
    # the sequence shards.
    loss_weight = torch.randn(case.batch_size, case.seq_len, D_MODEL, device=device)
    cu_doc_lens = case.cu_doc_lens(device)

    with torch.autocast(device.type, dtype=torch.bfloat16):
        y = module(x, cu_doc_lens=cu_doc_lens)
    (y.float() * loss_weight).sum().backward()
    assert x.grad is not None

    grads = {}
    for name, param in module.named_parameters():
        assert param.grad is not None, f"reference produced no gradient for {name}"
        grads[name] = param.grad.detach().float().cpu()
    torch.save(
        {
            "x": x.detach().cpu(),
            "loss_weight": loss_weight.cpu(),
            "cu_doc_lens": None if cu_doc_lens is None else cu_doc_lens.cpu(),
            "y": y.detach().cpu(),
            "x_grad": x.grad.detach().cpu(),
            "grads": grads,
        },
        ref_path,
    )
    save_model_and_optim_state(checkpoint_dir, module)


def _run_ulysses_cp_parity(checkpoint_dir: str, ref_path: str, mixer_name: str, case_name: str):
    spec, case = MIXERS[mixer_name], CASES[case_name]
    device = get_default_device()
    rank, world_size = get_rank(), get_world_size()
    mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("cp",))
    cp_group = mesh["cp"].get_group()

    module = spec.build(device.type, **case.overrides)
    module.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())
    load_model_and_optim_state(checkpoint_dir, module)

    ref = torch.load(ref_path, map_location=device)
    seq_len = ref["x"].size(1)
    assert seq_len % world_size == 0
    t_local = seq_len // world_size
    local = slice(rank * t_local, (rank + 1) * t_local)

    # Contiguous sequence shard, as the Ulysses load balancer produces. Document boundaries
    # stay global under Ulysses because the layer sees the full sequence after the exchange.
    x_local = ref["x"][:, local].clone().requires_grad_(True)
    cu_doc_lens = ref["cu_doc_lens"]

    with torch.autocast(device.type, dtype=torch.bfloat16):
        y_local = module(x_local, cu_doc_lens=cu_doc_lens)
    assert y_local.shape == x_local.shape, y_local.shape
    _assert_close(f"rank{rank} output", y_local, ref["y"][:, local], FWD_TOL)

    (y_local.float() * ref["loss_weight"][:, local]).sum().backward()

    # Sliced parameters must only receive gradient inside this rank's slice.
    for name, param_slice in _expected_param_slices(module, rank, world_size).items():
        grad = module.get_parameter(name).grad
        assert grad is not None, name
        outside = torch.ones(grad.shape[0], dtype=torch.bool, device=grad.device)
        outside[param_slice] = False
        assert (grad[outside] == 0).all(), f"rank{rank} {name}: gradient leaked outside CP slice"
        assert (grad[param_slice] != 0).any(), f"rank{rank} {name}: no gradient inside CP slice"

    # Summed over the CP group, per-rank gradients must reproduce the full-sequence gradients.
    for name, param in module.named_parameters():
        assert param.grad is not None, f"rank{rank} produced no gradient for {name}"
        grad = param.grad.detach().float().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=cp_group)
        _assert_close(f"grad {name}", grad, ref["grads"][name], GRAD_TOL)

    assert x_local.grad is not None
    _assert_close(f"rank{rank} x.grad", x_local.grad, ref["x_grad"][:, local], GRAD_TOL)


def _parity_params():
    params = []
    for mixer_name in MIXERS:
        for case_name, case in CASES.items():
            if mixer_name not in case.mixers:
                continue
            marks = []
            if case.overrides.get("use_experimental_kernels"):
                marks.append(
                    pytest.mark.skipif(
                        not _has_kernel_fun(), reason="requires the kernel-fun package"
                    )
                )
            params.append(
                pytest.param(mixer_name, case_name, id=f"{mixer_name}-{case_name}", marks=marks)
            )
    return params


def _has_kernel_fun() -> bool:
    try:
        import kernel_fun  # noqa: F401
    except ImportError:
        return False
    return True


@requires_multi_gpu
@requires_fla
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("mixer_name,case_name", _parity_params())
def test_ulysses_cp_forward_backward_parity(
    tmp_path, mixer_name: str, case_name: str, world_size: int
):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")

    ref_path = tmp_path / "reference.pt"
    checkpoint_dir = tmp_path / "checkpoint"
    _build_reference(mixer_name, case_name, ref_path, checkpoint_dir, torch.device("cuda"))

    run_distributed_test(
        _run_ulysses_cp_parity,
        world_size=world_size,
        backend="nccl",
        start_method="spawn",
        func_args=(str(checkpoint_dir), str(ref_path), mixer_name, case_name),
    )
