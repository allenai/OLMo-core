import pytest
import torch

import olmo_core.ops.moe as ops
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.emo import EmoRouterConfig
from olmo_core.nn.moe.v2.emo_router import EmoRouterV2
from olmo_core.nn.moe.v2.router import MoERouterConfigV2


def test_segment_ids_from_eos() -> None:
    input_ids = torch.tensor([[1, 2, 0, 3, 0, 4], [0, 1, 0, 0, 2, 3]])
    assert torch.equal(
        ops.segment_ids_from_eos(input_ids, 0),
        torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 1, 2, 3, 3]]),
    )


@pytest.mark.parametrize("tokens", [[1, 2, 0, 3, 0, 4], [0, 1, 0, 0, 2, 0], [1, 2, 3], [0]])
def test_routing_segments_match_attention_documents(tokens):
    from olmo_core.data.utils import get_document_lengths

    ids = torch.tensor(tokens)
    lengths = get_document_lengths(ids, eos_token_id=0)
    expected = torch.arange(len(lengths)).repeat_interleave(lengths)
    torch.testing.assert_close(ops.segment_ids_from_eos(ids[None], 0)[0], expected)


def test_emo_config_validates_against_routed_experts() -> None:
    with pytest.raises(OLMoConfigurationError, match="number of routed experts"):
        EmoRouterConfig(
            eos_token_id=0,
            min_document_expert_pool=2,
            max_document_expert_pool=5,
        ).validate_for_router(num_experts=4, top_k=2)


def test_emo_router_uses_document_pool_across_all_routed_experts() -> None:
    router = MoERouterConfigV2(
        d_model=4,
        num_experts=4,
        top_k=1,
        emo=EmoRouterConfig(
            eos_token_id=0,
            min_document_expert_pool=1,
            max_document_expert_pool=1,
            eval_document_expert_pool=1,
        ),
    ).build()
    assert isinstance(router, EmoRouterV2)
    router.eval()
    with torch.no_grad():
        router.weight.copy_(torch.eye(4).reshape(-1))

    x = torch.tensor([[[4.0, 0, 0, 0], [3.0, 0, 0, 0], [0, 0, 4.0, 0], [0, 0, 3.0, 0]]])
    segment_ids = torch.tensor([[0, 0, 1, 1]])
    _, indices, counts, _ = router(x, False, segment_ids=segment_ids)

    assert indices is not None
    assert torch.equal(indices.squeeze(-1), torch.tensor([[0, 0, 2, 2]]))
    assert counts is not None and counts.shape == (4,)


def test_emo_router_requires_segment_ids() -> None:
    router = MoERouterConfigV2(
        d_model=4,
        num_experts=4,
        top_k=1,
        emo=EmoRouterConfig(
            eos_token_id=0,
            min_document_expert_pool=1,
            max_document_expert_pool=2,
        ),
    ).build()
    with pytest.raises(OLMoConfigurationError, match="segment_ids"):
        router(torch.randn(1, 2, 4), False)


def test_emo_recompute_preserves_stochastic_routing_and_gradients():
    from copy import deepcopy

    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(118)
    original = (
        MoERouterConfigV2(
            d_model=8,
            num_experts=8,
            top_k=2,
            emo=EmoRouterConfig(
                eos_token_id=0, min_document_expert_pool=2, max_document_expert_pool=8
            ),
        )
        .build()
        .train()
    )
    reference, recomputed = deepcopy(original), deepcopy(original)
    tokens = torch.tensor([[1, 2, 0, 3, 4, 0, 5, 6]])
    segments = ops.segment_ids_from_eos(tokens, 0)
    source = torch.randn(1, 8, 8)
    results = []
    for router, enabled in ((reference, False), (recomputed, True)):
        x = source.clone().requires_grad_()
        selections = []

        def forward(x):
            weights, indices, _, _ = router(x, False, segment_ids=segments)
            selections.append(indices.detach().clone())
            return weights

        torch.manual_seed(209)
        output = (
            checkpoint(forward, x, use_reentrant=False, preserve_rng_state=True)
            if enabled
            else forward(x)
        )
        output.square().sum().backward()
        for selection in selections:
            torch.testing.assert_close(selection, selections[0], rtol=0, atol=0)
        results.append((output.detach(), x.grad, router.weight.grad, torch.get_rng_state()))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
