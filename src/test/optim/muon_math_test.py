import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer
from olmo_core.optim.muon_math import muon_direction, newton_schulz, uses_muon
from olmo_core.testing.distributed import run_distributed_test


def test_parameter_routing():
    for name, shape in [
        ("blocks.0.attention.w_q.weight", (8, 16)),
        ("blocks.0.feed_forward.router.weight", (16, 8)),
        ("blocks.0.feed_forward.routed_experts.w_down", (4, 8, 16)),
    ]:
        assert uses_muon(name, shape)
    for name, shape in [
        ("embeddings.weight", (16, 8)),
        ("lm_head.w_out.weight", (16, 8)),
        ("blocks.0.attention.q_norm.weight", (4, 8)),
        ("blocks.0.attention.q_conv1d.weight", (8, 4)),
        ("blocks.0.attention.dt_bias", (8,)),
    ]:
        assert not uses_muon(name, shape)


def test_five_steps_match_singular_value_polynomial():
    torch.manual_seed(17)
    matrix = torch.randn(7, 11)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    s = s / matrix.norm()
    for a, b, c in [(3.4445, -4.7750, 2.0315)] * 5:
        s = a * s + b * s**3 + c * s**5
    expected = (u * s) @ vh
    torch.testing.assert_close(newton_schulz(matrix), expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(newton_schulz(matrix.T), expected.T, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(newton_schulz(torch.zeros(3, 5))) == 0


def test_experts_and_packed_swiglu_are_independent():
    torch.manual_seed(5)
    packed = torch.randn(3, 8, 6)
    expected = torch.stack(
        [
            torch.cat([newton_schulz(part) * 6**0.5 * 0.2 for part in expert.chunk(2, dim=0)])
            for expert in packed
        ]
    )
    torch.testing.assert_close(muon_direction("experts.w_up_gate", packed), expected)


def _run_sharded_muon():
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
    torch.manual_seed(13)
    name = "experts.w_up_gate"
    shape = (3, 8, 6)  # DP shards cut through the second expert.
    initial = torch.randn(shape)
    grad = torch.randn(shape)
    for placement in (Replicate(), Shard(0)):
        optim = object.__new__(OLMoDDPOptimizer)
        optim.muon_momentum = 0.95
        optim.muon_rms = 0.2
        optim.states = {
            f"{name}.main": distribute_tensor(initial.flatten().clone(), mesh, [placement]),
            f"{name}.exp_avg": distribute_tensor(torch.zeros(initial.numel()), mesh, [placement]),
            f"{name}.step": distribute_tensor(torch.tensor(0.0), mesh, [Replicate()]),
        }
        optim.main_grad = {name: distribute_tensor(grad.flatten(), mesh, [placement]).to_local()}
        group = {"lr": 0.003, "weight_decay": 0.1}
        expected = initial.clone()
        momentum = torch.zeros_like(grad)
        for factor in (1.0, 0.0, 1.0):
            before = {key: value.full_tensor().clone() for key, value in optim.states.items()}
            optim._step_muon(name, shape, group, torch.tensor(factor))
            if factor:
                momentum = 0.95 * momentum + grad
                expected = expected * (1 - 0.003 * 0.1) - 0.003 * muon_direction(
                    name, 0.95 * momentum + grad
                )
            else:
                for key, value in before.items():
                    torch.testing.assert_close(
                        optim.states[key].full_tensor(), value, atol=0, rtol=0
                    )
            torch.testing.assert_close(
                optim.states[f"{name}.main"].full_tensor().reshape(shape), expected
            )
            torch.testing.assert_close(
                optim.states[f"{name}.exp_avg"].full_tensor().reshape(shape), momentum
            )

    # Exercise dispatch through the real optimizer, its AdamW fallback, and resume.
    def build():
        params = {
            "hidden.weight": torch.nn.Parameter(initial[0].clone()),
            "embeddings.weight": torch.nn.Parameter(initial[0].clone()),
        }
        return OLMoDDPOptimizer(
            [{"named_params": params, "pg": "dp"}],
            world_mesh={"dense": mesh, "moe": None},
            algorithm="muon_adamw",
            use_distributed=False,
            lr=0.003,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        )

    optim = build()
    adam_param = torch.nn.Parameter(initial[0].clone())
    adam = torch.optim.AdamW([adam_param], lr=0.003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    optim.main_grad = {name: grad[0].flatten() for name in ("hidden.weight", "embeddings.weight")}
    adam_param.grad = grad[0].clone()
    adam.step()
    with torch.no_grad():
        optim._step_foreach()
    torch.testing.assert_close(
        optim.states["embeddings.weight.main"].full_tensor().reshape(8, 6), adam_param
    )
    checkpoint = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in optim.state_dict().items()
    }
    resumed = build()
    resumed.load_state_dict(checkpoint)
    resumed.main_grad = optim.main_grad
    with torch.no_grad():
        optim._step_foreach()
        resumed._step_foreach()
    for key in optim.states:
        torch.testing.assert_close(
            optim.states[key].full_tensor(), resumed.states[key].full_tensor()
        )


def test_shards_preserve_logical_matrices_and_skip_state():
    run_distributed_test(_run_sharded_muon, world_size=2, backend="gloo", start_method="spawn")
