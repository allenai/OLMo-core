import copy

import torch
import torch.nn.functional as F

from olmo_core.distributed.utils import get_local_tensor
from olmo_core.nn.moe.v2.router import MoERouterV2


def test_router_uses_bf16_saved_input_and_matches_reference_grads():
    torch.manual_seed(123)

    router = MoERouterV2(
        d_model=512,
        num_experts=8,
        top_k=2,
        init_device="cpu",
        dtype=torch.float32,
        use_recompute_fp32_cast=True,
    )
    router_ref = copy.deepcopy(router)

    x = torch.randn(1, 8, 512, dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    saved_meta: list[tuple[tuple[int, ...], torch.dtype]] = []

    def pack(t: torch.Tensor):
        saved_meta.append((tuple(t.shape), t.dtype))
        return t

    def unpack(t):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        scores, _, _, _ = router(x, scores_only=True)
        loss = scores.square().mean()
    loss.backward()

    # Reference old behavior: explicit fp32 cast before router linear.
    w_ref = get_local_tensor(router_ref.weight).view(router_ref.num_experts, router_ref.d_model).float()
    logits_ref = F.linear(x_ref.float(), w_ref).float()
    scores_ref = logits_ref.softmax(dim=-1)
    loss_ref = scores_ref.square().mean()
    loss_ref.backward()

    assert x.grad is not None
    assert x_ref.grad is not None
    torch.testing.assert_close(x.grad, x_ref.grad, atol=2e-3, rtol=2e-3)

    assert router.weight.grad is not None
    assert router_ref.weight.grad is not None
    torch.testing.assert_close(router.weight.grad, router_ref.weight.grad, atol=2e-3, rtol=2e-3)

    x_shape_saved_dtypes = [dtype for shape, dtype in saved_meta if shape == tuple(x.shape)]
    assert x_shape_saved_dtypes, "Expected router input-shaped tensor to be saved for backward."
    assert torch.bfloat16 in x_shape_saved_dtypes
    assert torch.float32 not in x_shape_saved_dtypes
