"""
Coverage for how the transformer discovers document-routed (EMo) blocks and routes per-token
segment IDs to them, including the pipeline-parallel path where segment IDs are supplied by the
caller instead of derived from token IDs.
"""

from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.train.train_module.transformer.common import get_emo_segment_ids
from olmo_core.train.train_module.transformer.config import (
    TransformerPipelineParallelConfig,
)

EOS_TOKEN_ID = 0


def _build_model(eos_token_ids: Optional[List[Optional[int]]] = None) -> Transformer:
    """
    Build a tiny transformer, attaching a stand-in document router to the blocks named by
    ``eos_token_ids``. Only the two attributes the model reads are needed, which keeps this
    independent of the MoE kernels the real router depends on.
    """
    n_layers = 2 if eos_token_ids is None else len(eos_token_ids)
    model = TransformerConfig(
        d_model=32,
        vocab_size=64,
        n_layers=n_layers,
        block=TransformerBlockConfig(
            sequence_mixer=AttentionConfig(n_heads=2),
            feed_forward=FeedForwardConfig(hidden_size=64),
            layer_norm=LayerNormConfig(),
        ),
        lm_head=LMHeadConfig(),
    ).build()

    if eos_token_ids is not None:
        for block, eos_token_id in zip(model.blocks.values(), eos_token_ids):
            if eos_token_id is not None:
                block.routed_experts_router = SimpleNamespace(
                    requires_segment_ids=True, eos_token_id=eos_token_id
                )
    return model


def test_emo_blocks_and_eos_token_id_discovered_from_blocks() -> None:
    model = _build_model([None, EOS_TOKEN_ID])
    assert model.emo_block_indices == [1]
    assert model.emo_eos_token_id == EOS_TOKEN_ID


def test_emo_eos_token_id_is_none_without_document_routed_blocks() -> None:
    model = _build_model()
    assert model.emo_block_indices == []
    assert model.emo_eos_token_id is None


def test_emo_eos_token_id_rejects_disagreeing_routers() -> None:
    model = _build_model([0, 1])
    with pytest.raises(OLMoConfigurationError, match="same eos_token_id"):
        model.emo_eos_token_id


def test_segment_ids_derived_from_token_ids_without_pipeline_parallelism() -> None:
    model = _build_model([EOS_TOKEN_ID, EOS_TOKEN_ID])
    input_ids = torch.tensor([[1, 2, EOS_TOKEN_ID, 3]])

    _, _, _, per_block_kwargs, _ = model._prepare_inputs(input_ids)

    expected = torch.tensor([[0, 0, 1, 1]])
    assert set(per_block_kwargs) == {0, 1}
    for block_idx in (0, 1):
        assert torch.equal(per_block_kwargs[block_idx]["segment_ids"], expected)


def test_caller_supplied_segment_ids_take_precedence() -> None:
    model = _build_model([EOS_TOKEN_ID])
    input_ids = torch.tensor([[1, 2, EOS_TOKEN_ID, 3]])
    # Deliberately unlike anything derivable from 'input_ids' so the source is unambiguous.
    segment_ids = torch.tensor([[0, 1, 1, 2]])

    _, _, _, per_block_kwargs, _ = model._prepare_inputs(input_ids, segment_ids=segment_ids)

    assert torch.equal(per_block_kwargs[0]["segment_ids"], segment_ids)


def test_pipeline_parallelism_requires_caller_supplied_segment_ids() -> None:
    model = _build_model([EOS_TOKEN_ID])
    model._pp_enabled = True

    with pytest.raises(OLMoConfigurationError, match="requires 'segment_ids'"):
        model._prepare_inputs(torch.tensor([[1, 2, EOS_TOKEN_ID, 3]]))


def test_segment_ids_route_to_blocks_given_hidden_state_input() -> None:
    # A non-first pipeline stage receives hidden states in place of token IDs, so segment IDs
    # cannot be derived locally and must survive being passed in alongside them.
    model = _build_model([EOS_TOKEN_ID])
    model._pp_enabled = True
    hidden_states = torch.randn(2, 4, 32)
    segment_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])

    _, _, _, per_block_kwargs, _ = model._prepare_inputs(hidden_states, segment_ids=segment_ids)

    assert torch.equal(per_block_kwargs[0]["segment_ids"], segment_ids)


def test_segment_ids_shape_must_match_the_input() -> None:
    model = _build_model([EOS_TOKEN_ID])

    with pytest.raises(ValueError, match="must match the input"):
        model._prepare_inputs(
            torch.tensor([[1, 2, EOS_TOKEN_ID, 3]]), segment_ids=torch.tensor([[0, 0, 1]])
        )


def test_segment_ids_ignored_by_a_stage_without_document_routed_blocks() -> None:
    # Every rank derives segment IDs from its own batch, so a stage holding only dense blocks
    # receives them too and must simply drop them.
    model = _build_model()
    model._pp_enabled = True

    _, _, all_block_kwargs, per_block_kwargs, _ = model._prepare_inputs(
        torch.randn(1, 4, 32), segment_ids=torch.tensor([[0, 0, 1, 1]])
    )

    assert per_block_kwargs == {}
    assert "segment_ids" not in all_block_kwargs


def test_segment_ids_helper_rejects_parts_that_disagree_on_eos_token_id() -> None:
    # A pipeline split leaves each part validating only its own blocks, so the disagreement has to
    # be caught again across the parts a rank holds.
    parts = [_build_model([0]), _build_model([1])]
    with pytest.raises(OLMoConfigurationError, match="same eos_token_id"):
        get_emo_segment_ids(parts, torch.tensor([[1, 2, 0, 3]]))


def test_segment_ids_helper_returns_none_without_document_routed_parts() -> None:
    assert get_emo_segment_ids([_build_model()], torch.tensor([[1, 2, 0, 3]])) is None


def test_block_topology_caches_are_invalidated_after_pruning() -> None:
    # Splitting a model into pipeline stages prunes blocks from a deepcopy, so a chunk whose caches
    # were populated beforehand would otherwise keep describing the unsplit block list.
    model = _build_model([EOS_TOKEN_ID, EOS_TOKEN_ID, EOS_TOKEN_ID])
    assert model.emo_block_indices == [0, 1, 2]

    del model.blocks["2"]
    assert model.emo_block_indices == [0, 1, 2], "expected the cache to be stale until invalidated"

    model.invalidate_block_topology_caches()
    assert model.emo_block_indices == [0, 1]


def test_pipeline_split_rejects_blocks_that_disagree_on_eos_token_id() -> None:
    # Each stage only sees its own blocks, so a disagreement has to be caught while the whole block
    # list is still visible; otherwise routing silently depends on where the split landed. The
    # check runs before the mesh is touched, so none is needed here.
    model = _build_model([0, 0, 1, 1])
    config = TransformerPipelineParallelConfig(degree=2)

    with pytest.raises(OLMoConfigurationError, match="same eos_token_id"):
        config.split_model(
            model,
            pp_mesh=None,  # type: ignore[arg-type]
            device=torch.device("cpu"),
        )
