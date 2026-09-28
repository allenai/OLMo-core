import pytest
import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    lora_param_names,
    merge_lora_,
)


class _Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attention = nn.Module()
        self.attention.w_q = nn.Linear(d, d, bias=False)  # type: ignore[assignment]
        self.attention.w_out = nn.Linear(d, d, bias=True)  # type: ignore[assignment]
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = self.attention.w_out(self.attention.w_q(x))
        return self.norm(x + h)


class _Toy(nn.Module):
    """Mirrors the shape that matters: `lm.blocks.<i>.attention.*` under a ModuleDict."""

    def __init__(self, d: int = 16, n_blocks: int = 2):
        super().__init__()
        self.lm = nn.Module()
        self.lm.blocks = nn.ModuleDict(  # type: ignore[assignment]
            {str(i): _Block(d) for i in range(n_blocks)}
        )
        self.lm.embeddings = nn.Embedding(8, d)  # type: ignore[assignment]
        self.connector = nn.Linear(d, d)

    def forward(self, x):
        for block in self.lm.blocks.values():
            x = block(x)
        return self.connector(x)


def _cfg(**kwargs) -> LoRAConfig:
    base = dict(
        rank=4,
        alpha=8.0,
        target_modules=["lm.blocks.*.attention.w_q", "lm.blocks.*.attention.w_out"],
    )
    base.update(kwargs)
    return LoRAConfig(**base)  # type: ignore[arg-type]


def test_zero_init_is_bitwise_identical_to_base():
    torch.manual_seed(0)
    model = _Toy()
    x = torch.randn(3, 5, 16)
    with torch.no_grad():
        before = model(x).clone()

    apply_lora(model, _cfg())
    assert isinstance(model.lm.blocks["0"].attention.w_q, LoRALinear)
    with torch.no_grad():
        after = model(x)

    # `lora_B` is zeros, so the added term is exactly zero -- not merely close.
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_key_space_is_base_names_plus_adapters_only():
    torch.manual_seed(0)
    model = _Toy()
    before = {n for n, _ in model.named_parameters()}

    new_names = apply_lora(model, _cfg())
    after = {n for n, _ in model.named_parameters()}

    assert before <= after, "apply_lora renamed or dropped an existing parameter"
    assert after - before == set(new_names)
    assert set(new_names) == set(lora_param_names(model))
    # Two adapters per targeted layer, two targeted layers per block, two blocks.
    assert len(new_names) == 2 * 2 * 2


def test_merge_round_trip_matches_adapted_forward():
    torch.manual_seed(0)
    model = _Toy()
    apply_lora(model, _cfg())
    # Give the adapters something to say -- with B=0 the merge is trivially correct.
    for name, p in model.named_parameters():
        if name.endswith(".lora_B"):
            with torch.no_grad():
                p.normal_(std=0.1)

    x = torch.randn(3, 5, 16)
    model.eval()
    with torch.no_grad():
        adapted = model(x).clone()

    merged_names = merge_lora_(model)
    assert merged_names == [
        "lm.blocks.0.attention.w_out",
        "lm.blocks.0.attention.w_q",
        "lm.blocks.1.attention.w_out",
        "lm.blocks.1.attention.w_q",
    ]
    assert lora_param_names(model) == []
    assert not any(isinstance(m, LoRALinear) for m in model.modules())

    with torch.no_grad():
        merged = model(x)
    torch.testing.assert_close(merged, adapted, rtol=1e-5, atol=1e-6)


def test_merged_model_has_the_unadapted_key_space():
    torch.manual_seed(0)
    reference = {n for n, _ in _Toy().named_parameters()}
    model = _Toy()
    apply_lora(model, _cfg())
    merge_lora_(model)
    assert {n for n, _ in model.named_parameters()} == reference


def test_trainable_set_is_adapters_plus_whatever_was_left_unfrozen():
    torch.manual_seed(0)
    model = _Toy()
    # Stand in for the train module's freeze loop: freeze the whole LM, keep the connector.
    for name, p in model.named_parameters():
        if name.startswith("lm."):
            p.requires_grad_(False)

    new_names = apply_lora(model, _cfg())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == set(new_names) | {"connector.weight", "connector.bias"}


def test_base_weights_are_frozen_even_if_freeze_loop_did_not_run():
    torch.manual_seed(0)
    model = _Toy()
    apply_lora(model, _cfg())
    assert not model.lm.blocks["0"].attention.w_q.weight.requires_grad
    assert not model.lm.blocks["0"].attention.w_out.bias.requires_grad


def test_unmatched_pattern_raises():
    model = _Toy()
    with pytest.raises(OLMoConfigurationError, match="matched no nn.Linear"):
        apply_lora(model, _cfg(target_modules=["lm.blocks.*.attention.w_q", "lm.mlp.*"]))


def test_double_apply_raises():
    model = _Toy()
    apply_lora(model, _cfg())
    with pytest.raises(OLMoConfigurationError, match="already a LoRALinear"):
        apply_lora(model, _cfg())


def test_adapter_init_is_rank_deterministic():
    torch.manual_seed(0)
    a = _Toy()
    torch.manual_seed(1234)  # a different global seed must not change the adapters
    b = _Toy()
    apply_lora(a, _cfg())
    apply_lora(b, _cfg())
    torch.testing.assert_close(
        a.lm.blocks["0"].attention.w_q.lora_A,
        b.lm.blocks["0"].attention.w_q.lora_A,
        rtol=0,
        atol=0,
    )


def test_gradients_flow_only_to_adapters():
    torch.manual_seed(0)
    model = _Toy()
    for name, p in model.named_parameters():
        if name.startswith("lm."):
            p.requires_grad_(False)
    apply_lora(model, _cfg())
    model(torch.randn(2, 4, 16)).sum().backward()

    lora_grads = [
        model.get_parameter(n).grad for n in lora_param_names(model) if n.endswith("lora_A")
    ]
    assert all(g is not None and torch.isfinite(g).all() for g in lora_grads)
    assert model.lm.blocks["0"].attention.w_q.weight.grad is None


def test_bad_rank_rejected():
    with pytest.raises(OLMoConfigurationError):
        LoRAConfig(rank=0)
    with pytest.raises(OLMoConfigurationError):
        LoRAConfig(target_modules=[])


def test_broken_optional_fla_does_not_break_package_import(monkeypatch):
    """A *broken* optional dependency must degrade like a missing one.

    `import fla` transitively imports torchaudio, whose compiled extension can raise
    `OSError` (not `ImportError`) when it does not match the installed torch. The guard
    used to catch only `ImportError`, so a bad image took down `import olmo_core` itself.
    """
    import builtins
    import importlib

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "fla" or name.startswith("fla."):
            raise OSError("undefined symbol: torch_dtype_float4_e2m1fn_x2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    module = importlib.reload(
        importlib.import_module("olmo_core.nn.attention.flash_linear_attn_api")
    )
    try:
        assert module.has_fla() is False
    finally:
        monkeypatch.undo()
        importlib.reload(module)


def test_adapter_init_draws_on_the_generators_device_not_the_parameters():
    """Regression: `Tensor.uniform_(generator=...)` requires the generator's device to match
    the tensor's, so seeding a CUDA parameter from the CPU generator raised
    "Expected a 'cuda' device type for generator but found 'cpu'" on the first real run.
    The generator is CPU on purpose -- that is what makes every rank draw identical
    adapters without a collective -- so the draw happens on CPU and is copied in."""
    layer = LoRALinear(8, 4, rank=2, alpha=4.0, bias=False)
    gen = torch.Generator(device="cpu").manual_seed(7919)
    layer.reset_lora_parameters(generator=gen)

    expected = torch.empty(2, 8).uniform_(
        -((3.0 / 8) ** 0.5),
        (3.0 / 8) ** 0.5,
        generator=torch.Generator(device="cpu").manual_seed(7919),
    )
    torch.testing.assert_close(layer.lora_A, expected, rtol=0, atol=0)
    assert torch.count_nonzero(layer.lora_B) == 0
