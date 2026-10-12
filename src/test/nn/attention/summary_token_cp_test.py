"""
Ulysses context-parallel parity for :class:`SummaryTokenAttention`.

These run on CPU/gloo with the ``torch`` backend, so they gate the CP path without GPUs.

The specific hazard being tested: :class:`SummaryTokenAttention` overrides ``sdpa`` and therefore
**bypasses the backend**, which is where Ulysses normally performs its all-to-all. If that gather is
missing, q/k/v stay sequence-sharded while ``summary_roles`` is full-length, and the mask gets built
for the wrong sequence. The shapes disagree only by the CP degree, so this is the kind of bug that
produces a plausible forward pass and trains silently wrong -- which is why the gradient half matters
as much as the forward half.

Also covers ``cp == n_heads`` (one head per rank), where a ``reshape`` in the collectives can skip the
copy it looks like it is making.
"""

import copy
from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.attention.ring import UlyssesContextParallelStyle
from olmo_core.nn.attention.summary_mask import build_summary_roles
from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.utils import get_default_device, seed_all

DOC_START, DOC_END, SUMM, EOS, PAD = 900, 901, 902, 903, 904
N_SUMMARY = 2


def _ids() -> torch.Tensor:
    """A 20-token example: instruction, two documents each with a summary run, query, eos, padding."""
    ids = [10, 11]
    for d in range(2):
        ids += [DOC_START, 20 + d, 21 + d, DOC_END] + [SUMM] * N_SUMMARY
    ids += [DOC_START, 50, DOC_END, EOS, PAD, PAD]
    assert len(ids) == 20, len(ids)
    return torch.tensor([ids])


def _build(d_model: int, n_heads: int, n_kv_heads: int, device: torch.device):
    seed_all(0)
    cfg = AttentionConfig(
        name=AttentionType.summary_token,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        bias=False,
        backend=AttentionBackendName.torch,
        n_summary_tokens=N_SUMMARY,
    )
    return cfg.build(d_model=d_model, layer_idx=0, n_layers=1, init_device=device.type)


def _test_ulysses_parity(n_heads: int, n_kv_heads: int, causal_arm: bool):
    device = get_default_device()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("cp",))

    head_dim = 8
    d_model = n_heads * head_dim
    ids = _ids().to(device)
    B, T = ids.shape

    roles = build_summary_roles(
        ids,
        doc_start_id=DOC_START,
        doc_end_id=DOC_END,
        summary_token_id=SUMM,
        eos_id=EOS,
        pad_id=PAD,
    )
    causal_example = torch.tensor([causal_arm], device=device)

    att_ref = _build(d_model, n_heads, n_kv_heads, device)
    att_cp = copy.deepcopy(att_ref)
    att_cp.apply_cp(mesh["cp"], uly=UlyssesContextParallelStyle())

    seed_all(1234)
    x = torch.randn(B, T, d_model, device=device)

    # Reference: the full sequence with no CP, identical on every rank.
    out_ref = att_ref(x, summary_roles=roles, causal_example=causal_example)
    out_ref.sum().backward()

    # CP: this rank owns a contiguous slice of the sequence, but the SAME full-length roles --
    # after the all-to-all inside sdpa the mask is applied to the whole sequence.
    t_local = T // world_size
    x_local = x[:, rank * t_local : (rank + 1) * t_local, :]
    out_cp = att_cp(x_local, summary_roles=roles, causal_example=causal_example)
    out_cp.sum().backward()

    torch.testing.assert_close(
        out_cp,
        out_ref[:, rank * t_local : (rank + 1) * t_local, :],
        msg=lambda m: f"rank {rank}: forward mismatch under CP\n{m}",
    )

    # Each rank holds a partial gradient; the training step sums them across the CP group. Reduce
    # everything first and only then assert -- a rank that fails early would leave its peers blocked
    # in all_reduce, hanging the test instead of failing it.
    reduced = []
    for (name, p_ref), (_, p_cp) in zip(
        att_ref.named_parameters(), att_cp.named_parameters(), strict=True
    ):
        assert p_cp.grad is not None and p_ref.grad is not None
        grad = p_cp.grad.clone()
        dist.all_reduce(grad)
        reduced.append((name, p_ref.grad, grad))
    for name, want, got in reduced:
        torch.testing.assert_close(
            got, want, msg=lambda m, name=name: f"rank {rank}: grad mismatch for {name}\n{m}"
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "n_heads,n_kv_heads",
    [
        pytest.param(4, 2, id="cp_divides_kv_heads"),
        pytest.param(4, 1, id="kv_replication"),
        pytest.param(2, 1, id="one_head_per_rank"),
    ],
)
@pytest.mark.parametrize("causal_arm", [False, True])
def test_ulysses_parity(backend, n_heads, n_kv_heads, causal_arm):
    run_distributed_test(
        partial(_test_ulysses_parity, n_heads, n_kv_heads, causal_arm),
        backend=backend,
        world_size=2,
    )


def test_ring_cp_is_rejected():
    """
    Ring CP cannot express an arbitrary mask -- each rank's rows are a non-contiguous permutation and
    the kernel understands only ``causal + cu_seqlens`` -- so it must fail loudly rather than mask
    the wrong positions.

    Driven by setting the backend's CP state directly rather than through ``apply_cp``: the flash
    backends accept ring CP (which is what makes this reachable in a real run), but they cannot be
    constructed on CPU, and ``TorchAttentionBackend`` refuses ring before this guard is consulted.
    """
    from olmo_core.exceptions import OLMoConfigurationError

    att = _build(32, 4, 2, torch.device("cpu"))
    att.backend.cp_enabled = True
    att.backend.ring = object()
    att.backend.uly = None

    q = k = v = torch.randn(1, 20, 4, 8)
    with pytest.raises(OLMoConfigurationError, match="Ulysses"):
        att.sdpa(q, k, v)
