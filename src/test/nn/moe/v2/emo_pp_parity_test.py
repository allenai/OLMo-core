"""
Numerical parity between an unsplit EMo model and the same model split into pipeline stages.

The unsplit model derives per-token segment IDs from token IDs internally. A split model cannot:
only the first stage sees token IDs, so later stages take the segment IDs as an input. These tests
pin that the two produce identical routing decisions and identical outputs.

Transport and microbatch alignment are covered separately by the multi-GPU pipeline execution
tests; here both stages run in one process so any difference is attributable to the routing itself.
"""

import copy
from typing import Dict, List

import torch

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.ddp.model import OLMoDDPModel
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.emo import EmoRouterConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.ops.moe import segment_ids_from_eos
from olmo_core.testing import requires_gpu

EOS_TOKEN_ID = 0
N_LAYERS = 4
SPLIT_AFTER = 2


def _emo_model_config(d_model: int = 64) -> OLMoDDPModelConfig:
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=DType.float32)
    return OLMoDDPModelConfig(
        init_seed=0,
        d_model=d_model,
        recompute_each_block=False,
        vocab_size=128,
        n_layers=N_LAYERS,
        name=TransformerType.moe_fused_v2,
        block=OLMoDDPTransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            attention=AttentionConfig(
                name=AttentionType.default,
                n_heads=4,
                bias=False,
                use_flash=False,
                dtype=DType.float32,
            ),
            routed_experts=RoutedExpertsConfig(
                d_model=d_model, hidden_size=128, num_experts=8, bias=False, dtype=DType.float32
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=d_model,
                num_experts=8,
                top_k=2,
                dtype=DType.float32,
                emo=EmoRouterConfig(
                    eos_token_id=EOS_TOKEN_ID,
                    min_document_expert_pool=2,
                    max_document_expert_pool=8,
                    # Pin the pool so the comparison is well defined: while training, EMo samples a
                    # pool size per document from the global RNG, which the two runs would not share.
                    eval_document_expert_pool=4,
                ),
            ),
            shared_experts=None,
            layer_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=DType.float32),
    )


def _split_into_stages(model: OLMoDDPModel) -> List[OLMoDDPModel]:
    """
    Prune ``model`` into two pipeline stages the same way
    :meth:`TransformerPipelineParallelConfig.split_model` does, without needing a process group.

    CUDA events cannot be deepcopied, so they are purged before the split and reinstalled on the
    original and every chunk afterwards, matching what the train module does around its split.
    """
    model.purge_cuda_events()

    stages = []
    for stage_idx, (start, stop) in enumerate([(0, SPLIT_AFTER), (SPLIT_AFTER, N_LAYERS)]):
        chunk = copy.deepcopy(model)
        if stage_idx != 0:
            chunk.embeddings = None  # type: ignore[assignment]
            chunk.embedding_norm = None  # type: ignore[assignment]
        for block_idx in range(N_LAYERS):
            if not start <= block_idx < stop:
                del chunk.blocks[str(block_idx)]
        if stage_idx != 1:
            chunk.lm_head = None  # type: ignore[assignment]
        chunk.invalidate_block_topology_caches()
        chunk._pp_enabled = True
        chunk.eval()
        stages.append(chunk)

    model.install_cuda_events()
    for chunk in stages:
        chunk.install_cuda_events()
    return stages


def _build_reference_and_stages():
    """
    Build the model and split it immediately, before any forward.

    The split has to happen on a freshly built model, as it does in the train module: a forward
    leaves per-block runtime state behind that the deepcopy would either trip over or carry along.
    """
    model = _emo_model_config().build(init_device="cuda")
    model.eval()
    first, last = _split_into_stages(model)
    return model, first, last


def _capture_expert_indices(model: OLMoDDPModel, into: Dict[int, torch.Tensor]) -> List:
    """Record each document router's selected experts, keyed by the block index it belongs to."""
    handles = []
    for block_key, block in model.blocks.items():
        router = getattr(block, "routed_experts_router", None)
        if router is None:
            continue

        def hook(_module, _args, output, block_idx=int(block_key)):
            # The router returns (expert_weights, expert_indices, counts, aux_loss_info).
            into[block_idx] = output[1].detach().clone()

        handles.append(router.register_forward_hook(hook))
    return handles


def _batch(device: torch.device):
    # Several packed documents of differing lengths, including one that ends exactly at the
    # sequence boundary, so document pooling has something non-trivial to do.
    input_ids = torch.tensor(
        [
            [7, 3, EOS_TOKEN_ID, 5, 9, 1, EOS_TOKEN_ID, 4],
            [2, EOS_TOKEN_ID, 6, 8, EOS_TOKEN_ID, 3, 5, EOS_TOKEN_ID],
        ],
        device=device,
    )
    labels = input_ids.roll(-1, dims=1)
    return input_ids, labels


@requires_gpu
def test_split_stages_match_unsplit_model():
    device = torch.device("cuda")
    model, first, last = _build_reference_and_stages()

    input_ids, labels = _batch(device)
    segment_ids = segment_ids_from_eos(input_ids, EOS_TOKEN_ID)

    reference_indices: Dict[int, torch.Tensor] = {}
    handles = _capture_expert_indices(model, reference_indices)
    with torch.no_grad():
        reference = model(input_ids, labels=labels, loss_reduction="sum")
    for handle in handles:
        handle.remove()

    staged_indices: Dict[int, torch.Tensor] = {}
    handles = _capture_expert_indices(first, staged_indices)
    handles += _capture_expert_indices(last, staged_indices)
    with torch.no_grad():
        hidden = first(input_ids, segment_ids=segment_ids)
        staged = last(hidden, labels=labels, loss_reduction="sum", segment_ids=segment_ids)
    for handle in handles:
        handle.remove()

    # Every block must route to the same experts, including the ones on the later stage that could
    # not have derived the segment IDs themselves.
    assert set(staged_indices) == set(reference_indices) == set(range(N_LAYERS))
    for block_idx in range(N_LAYERS):
        assert torch.equal(
            staged_indices[block_idx], reference_indices[block_idx]
        ), f"block {block_idx} routed differently"

    torch.testing.assert_close(staged.loss, reference.loss, rtol=1e-6, atol=1e-6)


@requires_gpu
def test_split_stages_diverge_without_segment_ids_on_later_stage():
    # Guards the assertion above against being vacuous: if the later stage's segment IDs were
    # wrong, the parity check has to be able to notice.
    device = torch.device("cuda")
    model, first, last = _build_reference_and_stages()

    input_ids, labels = _batch(device)
    segment_ids = segment_ids_from_eos(input_ids, EOS_TOKEN_ID)
    # Every token in one document, which is not what the batch actually contains.
    wrong_segment_ids = torch.zeros_like(segment_ids)

    with torch.no_grad():
        reference = model(input_ids, labels=labels, loss_reduction="sum")

    with torch.no_grad():
        hidden = first(input_ids, segment_ids=segment_ids)
        staged = last(hidden, labels=labels, loss_reduction="sum", segment_ids=wrong_segment_ids)

    # Well clear of the tolerance the parity test above passes at, so that test is not vacuous.
    assert (staged.loss - reference.loss).abs() > 1e-4


@requires_gpu
def test_split_stages_match_unsplit_model_gradients():
    device = torch.device("cuda")
    model, first, last = _build_reference_and_stages()

    input_ids, labels = _batch(device)
    segment_ids = segment_ids_from_eos(input_ids, EOS_TOKEN_ID)

    model(input_ids, labels=labels, loss_reduction="sum").loss.backward()
    reference_grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    hidden = first(input_ids, segment_ids=segment_ids)
    last(hidden, labels=labels, loss_reduction="sum", segment_ids=segment_ids).loss.backward()

    staged_grads: Dict[str, torch.Tensor] = {}
    for stage in (first, last):
        for name, param in stage.named_parameters():
            if param.grad is not None:
                staged_grads[name] = param.grad.detach()

    assert reference_grads, "reference run produced no gradients"
    missing = set(reference_grads) - set(staged_grads)
    assert not missing, f"stages produced no gradient for {sorted(missing)[:5]}"
    for name, expected in reference_grads.items():
        torch.testing.assert_close(
            staged_grads[name], expected, rtol=1e-5, atol=1e-6, msg=f"gradient mismatch for {name}"
        )
