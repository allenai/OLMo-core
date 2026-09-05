import pytest
import torch
from torch.distributed.checkpoint.metadata import (
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)

from olmo_core.train.train_module.transformer.headwise_qk_checkpoint import (
    finish_qk_expansion,
    prepare_qk_expansion,
)


@pytest.mark.parametrize("suffix", ["main", "exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("heads", [4, 8])
def test_expand_qk_state(suffix, heads):
    name = "blocks.7.sequence_mixer.q_norm.weight"
    key = f"{name}.{suffix}"
    target = torch.empty(heads * 128)
    state = {key: target}
    metadata = Metadata({key: TensorStorageMetadata(TensorProperties(), torch.Size([128]), [])})
    expansions = prepare_qk_expansion(state, metadata, {name: (heads, 128)})
    source = torch.linspace(0.5, 1.5, 128)
    state[key].copy_(source)
    finish_qk_expansion(state, expansions)
    assert state[key] is target
    torch.testing.assert_close(target.view(heads, 128), source.expand(heads, 128))
    x = torch.randn(2, 3, heads, 128)
    normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(normalized * source, normalized * target.view(heads, 128))


def test_reject_unrelated_shape_change():
    key = "blocks.7.w_q.weight.main"
    state = {key: torch.empty(1024)}
    metadata = Metadata({key: TensorStorageMetadata(TensorProperties(), torch.Size([128]), [])})
    with pytest.raises(ValueError, match="Unsupported checkpoint shape change"):
        prepare_qk_expansion(state, metadata, {})


def test_noop_matching_shape():
    key = "blocks.7.q_norm.weight.main"
    state = {key: torch.empty(1024)}
    metadata = Metadata({key: TensorStorageMetadata(TensorProperties(), torch.Size([1024]), [])})
    assert not prepare_qk_expansion(state, metadata, {key.rsplit(".", 1)[0]: (8, 128)})
