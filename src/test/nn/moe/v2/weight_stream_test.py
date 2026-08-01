from types import SimpleNamespace

import torch

from olmo_core.nn.moe.v2 import weight_stream


class _FakeEPMesh:
    def __init__(self, group):
        self.group = group

    def __getitem__(self, _name):
        return self

    def get_group(self):
        return self.group


class _FakeEPModel(torch.nn.Module):
    def __init__(self, local_weight, group):
        super().__init__()
        self.local_weight = local_weight
        self.owner = SimpleNamespace(_ep_sharded=True, ep_mesh=_FakeEPMesh(group))

    def state_dict(self, *args, **kwargs):
        del args, kwargs
        return {"blocks.0.routed_experts.w_down": self.local_weight}

    def get_submodule(self, _target):
        return self.owner


def test_fused_qkv_streams_hf_attention_weights_for_dense_layer():
    config = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=3,
        dense_layers_indices=[0],
    )
    fused = torch.arange(24 * 5, dtype=torch.float32).reshape(24, 5)

    weights = list(
        weight_stream.iter_olmo3moe_tensor_to_hf(config, "blocks.0.attention.w_qkv.weight", fused)
    )

    assert [name for name, _ in weights] == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    torch.testing.assert_close(weights[0][1], fused[:12])
    torch.testing.assert_close(weights[1][1], fused[12:18])
    torch.testing.assert_close(weights[2][1], fused[18:])


def test_ep_weights_use_single_output_all_gather_and_stream_hf_experts(monkeypatch):
    group = object()
    local = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    model = _FakeEPModel(local, group)
    config = SimpleNamespace(
        model_type="olmo3moe",
        n_routed_experts=4,
        hidden_size=4,
        moe_intermediate_size=3,
        dense_layers_indices=[],
        shared_expert_intermediate_size=None,
    )
    gathers = []

    monkeypatch.setattr(weight_stream.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(weight_stream.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(weight_stream.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(weight_stream.dist, "get_process_group_ranks", lambda _group: [0, 1])

    def all_gather_into_tensor(output, input_, *, group):
        gathers.append((output.shape, input_.shape, group))
        output[:2].copy_(input_)
        output[2:].copy_(input_ + 100)

    monkeypatch.setattr(weight_stream.dist, "all_gather_into_tensor", all_gather_into_tensor)

    metadata = weight_stream.get_olmo_ddp_hf_weight_metadata(model, config)
    weights = list(weight_stream.iter_olmo_ddp_hf_weights(model, config))

    assert gathers == [((4, 3, 4), (2, 3, 4), group)]
    assert [name for name, _, _ in metadata] == [name for name, _ in weights]
    assert [shape for _, _, shape in metadata] == [(4, 3)] * 4
    torch.testing.assert_close(weights[0][1], local[0].T)
    torch.testing.assert_close(weights[2][1], (local[0] + 100).T)
