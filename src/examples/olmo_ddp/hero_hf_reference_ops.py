"""Slow, transparent expert reference math for offline conversion checks only.

This module never runs in a trainer or serving process. It replaces only permutation,
unpermutation and grouped GEMM implementations inside the current qualification process;
routing, parameters, architecture and attention semantics remain unchanged. The standalone
HF loop and native-vLLM checks do not use these adapters.
"""

import torch


def permute(inp, routing_map, num_out_tokens, map_type):
    """Stable expert-major gather; return the inverse gather as an opaque map."""
    if map_type != "index" or num_out_tokens != routing_map.numel():
        raise ValueError("Reference supports only dropless index-map evaluation")
    num_tokens, topk = routing_map.shape
    order = routing_map.flatten().argsort(stable=True)
    inverse = order.argsort().reshape(num_tokens, topk)
    return inp.index_select(0, order // topk), inverse


def unpermute(inp, row_id_map, restore_shape, map_type, merging_probs):
    """Restore token-major order and perform the weighted expert sum in FP32."""
    if map_type != "index" or tuple(row_id_map.shape) != tuple(merging_probs.shape):
        raise ValueError("Unsupported unpermutation inputs")
    values = inp.index_select(0, row_id_map.flatten()).reshape(*row_id_map.shape, inp.shape[-1])
    result = (values.float() * merging_probs.float().unsqueeze(-1)).sum(1).to(inp.dtype)
    return result.reshape(restore_shape)


def gmm(a, b, batch_sizes, trans_b=False):
    """One ordinary GEMM per expert, retaining the core's up/gate weight layout."""
    sizes = batch_sizes.tolist()
    if sum(sizes) != a.shape[0] or len(sizes) != b.shape[0]:
        raise ValueError("Expert splits do not exactly cover the input")
    output = []
    offset = 0
    for expert, size in enumerate(sizes):
        weight = b[expert].T if trans_b else b[expert]
        output.append(a[offset : offset + size] @ weight)
        offset += size
    return torch.cat(output, dim=0)


def install() -> None:
    """Install process-local validation adapters without editing training modules."""
    import olmo_core.nn.moe.utils as utils
    import olmo_core.nn.moe.v2.routed_experts as experts

    utils.moe_permute = permute
    utils.moe_unpermute = unpermute
    experts.USE_TORCH_GROUPED_MM = False
    experts.REQUIRES_HOST_SIDE_SPLIT_SIZES = True
    experts.gmm_no_compile = gmm


def self_test() -> None:
    """Check gather/expert/scatter composition against direct token-wise math on CPU."""
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(7, 16, generator=generator)
    route = torch.tensor([[2, 0], [1, 2], [0, 3], [3, 2], [1, 0], [2, 3], [0, 1]])
    probs = torch.rand(7, 2, generator=generator)
    weights = torch.randn(4, 16, 16, generator=generator)
    gathered, inverse = permute(x, route, route.numel(), "index")
    splits = route.flatten().bincount(minlength=4)
    for transpose in (False, True):
        expert_out = gmm(gathered, weights, splits, trans_b=transpose)
        actual = unpermute(expert_out, inverse, x.shape, "index", probs)
        expected = torch.stack(
            [
                sum(
                    (x[t] @ (weights[e].T if transpose else weights[e])) * probs[t, k]
                    for k, e in enumerate(route[t])
                )
                for t in range(x.shape[0])
            ]
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    self_test()
    print("Portable expert reference composition passed")
