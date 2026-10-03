"""Native identity, reference-weight gradients, actual dispatch and schedule checks."""

import argparse
import json

import olmoe3_adaptive_decay_plan as p
import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.v2.router import MoERouterConfigV2, MoERouterV2


def check(device="cpu"):
    """Compare the actual router against a native-width oracle, including backward."""
    p.self_test()
    torch.manual_seed(20261003)
    native = MoERouterV2(
        d_model=32,
        num_experts=512,
        top_k=16,
        normalize_expert_weights=1.0,
        restore_weight_scale=True,
        init_device=device,
    )
    with torch.no_grad():
        native.weight.normal_(0, 0.2)
    reference = MoERouterV2(
        d_model=32,
        num_experts=512,
        top_k=16,
        reference_top_k=16,
        normalize_expert_weights=1.0,
        restore_weight_scale=True,
        init_device=device,
    )
    reference.load_state_dict(native.state_dict())
    x = torch.randn(2, 9, 32, device=device)
    full, ids, _, _ = native(x, False)
    same, same_ids, _, _ = reference(x, False)
    assert torch.equal(full, same) and torch.equal(ids, same_ids)
    for k in (16, 14, 12, 10, 8):
        reference.top_k = k
        reference.zero_grad(set_to_none=True)
        weights, selected, counts, aux = reference(x, False)
        assert selected.shape == (2, 9, k) and int(counts.sum()) == 2 * 9 * k
        assert torch.equal(selected, ids[..., :k])
        assert torch.equal(weights, full[..., :k])
        assert int(aux[2].sum()) == 2 * 9 * k
        # The native-width oracle retains gradients through all sixteen weights
        # in the denominator while dropping eight contributions from the output.
        independent_weight = reference.weight.detach().clone().requires_grad_(True)
        logits = F.linear(x.float(), independent_weight.view(512, 32).float())
        raw, _ = logits.softmax(-1).topk(16, dim=-1)
        oracle = 16 * raw / raw.sum(-1, keepdim=True)
        coefficients = torch.randn_like(weights)
        (weights * coefficients).sum().backward()
        (oracle[..., :k] * coefficients).sum().backward()
        torch.testing.assert_close(
            reference.weight.grad, independent_weight.grad, rtol=2e-5, atol=2e-6
        )
    with torch.no_grad():
        reference.weight.zero_()
        reference.top_k = 16
        tied, tied_ids, _, _ = reference(x, False)
        reference.top_k = 8
        low, low_ids, _, _ = reference(x, False)
        assert torch.equal(low_ids, tied_ids[..., :8]) and torch.equal(low, tied[..., :8])
    for invalid in (
        dict(top_k=17),
        dict(original_top_k=16),
        dict(normalize_expert_weights=None),
    ):
        cfg = dict(
            d_model=32,
            num_experts=512,
            top_k=8,
            reference_top_k=16,
            normalize_expert_weights=1.0,
            restore_weight_scale=True,
        )
        cfg.update(invalid)
        try:
            MoERouterV2(**cfg)
        except OLMoConfigurationError:
            pass
        else:
            raise AssertionError(invalid)
    config = MoERouterConfigV2(
        d_model=32,
        num_experts=512,
        top_k=8,
        reference_top_k=16,
        normalize_expert_weights=1.0,
        restore_weight_scale=True,
    )
    assert config.build().reference_top_k == 16
    result = dict(
        passed=True,
        device=device,
        native_identity=True,
        retained_weights_exact=True,
        narrow_dispatch_counts=True,
        denominator_gradients=True,
        native_tie_order=True,
        unsupported_scaling_rejected=True,
        schedule_boundaries=True,
    )
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    check(parser.parse_args().device)
