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
        torch.tensor([[0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 2, 2]]),
    )


def test_emo_config_validates_against_routed_experts() -> None:
    with pytest.raises(OLMoConfigurationError, match="number of routed experts"):
        EmoRouterConfig(
            eos_token_id=0,
            min_document_expert_pool=2,
            max_document_expert_pool=5,
        ).validate(num_experts=4, top_k=2)


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

    x = torch.tensor(
        [[[4.0, 0, 0, 0], [3.0, 0, 0, 0], [0, 0, 4.0, 0], [0, 0, 3.0, 0]]]
    )
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
