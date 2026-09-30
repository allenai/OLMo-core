"""CPU proofs of group objectives under weighted labels, accumulation, and DP."""

import pytest
import torch
import torch.distributed as dist

from olmo_core.data.utils import split_batch
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.functional import weighted_cross_entropy_loss
from olmo_core.optim import AdamWConfig, OLMoDDPOptimizerConfig
from olmo_core.testing.distributed import run_distributed_test
from olmo_core.train.train_module.transformer.ddp_train_module import OLMoDDPTrainModule
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModule,
    MultimodalOLMoDDPTrainModuleConfig,
    MultimodalTransformerTrainModuleConfig,
    _normalize_loss_groups,
    _validate_loss_group_weights,
)

_COEFFICIENTS = {"text": 0.3, "vision": 0.7}


def _batch():
    return {
        "input_ids": torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [4, 0, 1]]),
        "labels": torch.tensor([[1, 2, -100], [2, 3, 4], [3, -100, 1], [0, 1, 2]]),
        "loss_masks": torch.tensor(
            [[0.25, 0.5, 99.0], [1.0, 0.1, 0.0], [2.0, 7.0, 0.5], [0.0, 0.25, 1.0]]
        ),
        "loss_group_names": ["text", "text", "vision", "vision"],
        "router_token_mask": torch.ones((4, 3), dtype=torch.bool),
    }


def _normalize(batch):
    return _normalize_loss_groups(
        batch,
        _COEFFICIENTS,
        label_ignore_index=-100,
        device=torch.device("cpu"),
        dp_process_group=None,
    )


def _ce(parameter, batch):
    logits = parameter[batch["input_ids"]].reshape(-1, 5)
    return weighted_cross_entropy_loss(
        logits, batch["labels"].reshape(-1), batch["loss_masks"].reshape(-1)
    )[0]


def _reference(parameter, batch):
    losses = []
    for group, coefficient in _COEFFICIENTS.items():
        rows = torch.tensor([name == group for name in batch["loss_group_names"]])
        group_batch = {
            key: value[rows] for key, value in batch.items() if isinstance(value, torch.Tensor)
        }
        denominator = (group_batch["loss_masks"] * (group_batch["labels"] != -100)).sum()
        losses.append(coefficient * _ce(parameter, group_batch) / denominator)
    return sum(losses)


@pytest.mark.parametrize("microbatch_rows", [1, 2, 4])
@pytest.mark.parametrize("mass_scale", [1.0, 0.01])
def test_separate_objective_matches_weighted_global_reference(microbatch_rows, mass_scale):
    original = _batch()
    original["loss_masks"] *= mass_scale
    saved = original["loss_masks"].clone()
    normalized, mass = _normalize(original)
    torch.testing.assert_close(mass, torch.tensor([1.85, 3.75]) * mass_scale)
    torch.testing.assert_close(original["loss_masks"], saved)
    assert (normalized["loss_masks"][original["labels"] == -100] == 0).all()
    torch.testing.assert_close(normalized["router_token_mask"], original["router_token_mask"])
    parameter = torch.randn(
        (5, 5), generator=torch.Generator().manual_seed(123), requires_grad=True
    )
    reference = parameter.detach().clone().requires_grad_()
    denominator = normalized["loss_masks"].sum().clamp_min(1)
    total = 0
    for microbatch in split_batch(normalized, microbatch_rows):
        loss = _ce(parameter, microbatch) / denominator
        total += loss.detach()
        loss.backward()
    expected = _reference(reference, original)
    expected.backward()
    torch.testing.assert_close(total, expected.detach())
    torch.testing.assert_close(parameter.grad, reference.grad)


@pytest.mark.parametrize("change", ["metadata", "unknown", "zero", "ignored", "nan", "negative"])
def test_group_objective_rejects_invalid_or_unsupervised_groups(change):
    batch = _batch()
    if change == "metadata":
        batch.pop("loss_group_names")
    elif change == "unknown":
        batch["loss_group_names"][0] = "unconfigured"
    elif change == "zero":
        batch["loss_masks"][:2] = 0
    elif change == "ignored":
        batch["labels"][:2] = -100
    elif change == "nan":
        batch["loss_masks"][0, 0] = float("nan")
    else:
        batch["loss_masks"][0, 0] = -1
    with pytest.raises(OLMoConfigurationError):
        _normalize(batch)


def _distributed_objective_and_failure_checks():
    torch.set_num_threads(1)
    rank = dist.get_rank()
    for mass_scale in (1.0, 0.01):
        original = _batch()
        original["loss_masks"] *= mass_scale
        # A group may be absent on an individual rank; the global update must cover it.
        local = split_batch(original, 2)[rank]
        normalized, mass = _normalize(local)
        torch.testing.assert_close(mass, torch.tensor([1.85, 3.75]) * mass_scale)
        global_divisor = normalized["loss_masks"].sum()
        dist.all_reduce(global_divisor)
        # Cover both inherited train modules' divisor clamp placements.
        for denominator in (
            global_divisor.clamp_min(1) / dist.get_world_size(),
            (global_divisor / dist.get_world_size()).clamp_min(1),
        ):
            parameter = torch.randn(
                (5, 5), generator=torch.Generator().manual_seed(123), requires_grad=True
            )
            reference = parameter.detach().clone().requires_grad_()
            for microbatch in split_batch(normalized, 1):
                (_ce(parameter, microbatch) / denominator).backward()
            dist.all_reduce(parameter.grad)
            parameter.grad /= dist.get_world_size()
            _reference(reference, original).backward()
            torch.testing.assert_close(parameter.grad, reference.grad)
    # Invalid metadata on just one rank must cause a coordinated error, not a hang.
    malformed = _batch()
    if rank == 0:
        malformed.pop("loss_group_names")
    with pytest.raises(OLMoConfigurationError):
        _normalize(malformed)
    # Missing supervised mass must also fail on every rank after the shared reduction.
    missing = _batch()
    missing["loss_masks"][:2] = 0
    with pytest.raises(OLMoConfigurationError, match="positive finite supervised mass"):
        _normalize(missing)
    dist.barrier()


def test_group_objective_distributed_accumulation_and_collective_failures():
    run_distributed_test(
        _distributed_objective_and_failure_checks,
        world_size=2,
        backend="gloo",
        start_method="spawn",
    )


@pytest.mark.parametrize(
    "weights", [{}, {"a": -1.0, "b": 2.0}, {"a": 0.5}, {"a": float("nan")}, {"": 1.0}]
)
def test_loss_group_config_validation(weights):
    with pytest.raises(OLMoConfigurationError):
        _validate_loss_group_weights(weights)


def test_loss_group_metadata_does_not_reach_model():
    module = object.__new__(MultimodalOLMoDDPTrainModule)
    module._pp_config = None
    module.response_logits_only = False
    _, _, kwargs = module._prepare_batch(_batch())
    assert "loss_group_names" not in kwargs
    assert _validate_loss_group_weights(None) == {}
    assert list(_validate_loss_group_weights({"vision": 0.7, "text": 0.3})) == ["text", "vision"]


@pytest.mark.parametrize(
    "config_type,optim",
    [
        (MultimodalOLMoDDPTrainModuleConfig, OLMoDDPOptimizerConfig(lr=1e-3)),
        (MultimodalTransformerTrainModuleConfig, AdamWConfig(lr=1e-3)),
    ],
)
def test_group_objective_config_round_trip(config_type, optim):
    config = config_type(
        rank_microbatch_size=16,
        max_sequence_length=16,
        optim=optim,
        loss_group_weights=_COEFFICIENTS,
    )
    assert config_type.from_dict(config.as_config_dict()) == config


def test_group_normalization_precedes_inherited_accumulation(monkeypatch):
    module = object.__new__(MultimodalOLMoDDPTrainModule)
    module.loss_group_weights = _COEFFICIENTS
    module.label_ignore_index = -100
    module.device = torch.device("cpu")
    module.dp_group = None
    original = _batch()
    expected, _ = _normalize(original)
    received = []

    def train_batch(self, batch, dry_run=False):
        assert self is module and dry_run
        received.append(batch)

    monkeypatch.setattr(OLMoDDPTrainModule, "train_batch", train_batch)
    module.train_batch(original, dry_run=True)
    assert len(received) == 1
    torch.testing.assert_close(received[0]["loss_masks"], expected["loss_masks"])
    torch.testing.assert_close(received[0]["labels"], original["labels"])


def test_group_weights_preserve_compiled_loss_interface():
    normalized, _ = _normalize(_batch())
    parameter = torch.randn((5, 5), requires_grad=True)
    compiled_ce = torch.compile(_ce, backend="eager", fullgraph=True)
    actual = compiled_ce(parameter, normalized)
    expected = _ce(parameter, normalized)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, parameter)[0], torch.autograd.grad(expected, parameter)[0]
    )


def test_grouped_source_telemetry_retains_exposure_without_global_target_errors():
    module = object.__new__(MultimodalOLMoDDPTrainModule)
    module.source_loss_mass_targets = {"replay": 0.35, "caption": 0.65}
    module.label_ignore_index = -100
    batch = _batch()
    batch["pack_source_names"] = [["replay"], ["replay"], ["caption"], ["caption"]]
    batch["example_ids"] = torch.zeros_like(batch["input_ids"])
    results = []
    for grouped in (False, True):
        source_batch = dict(batch)
        if not grouped:
            source_batch.pop("loss_group_names")
        metrics = {}
        module.record_metric = lambda name, value, *args, metrics=metrics, **kwargs: (
            metrics.__setitem__(name, value)
        )
        module._record_source_data_metrics(source_batch, batch["router_token_mask"])
        results.append(metrics)
    ordinary, grouped = results
    target_errors = {name for name in ordinary if name.endswith("loss_mass_target_abs_error")}
    assert len(target_errors) == 2
    assert grouped.keys() == ordinary.keys() - target_errors
    for name, value in grouped.items():
        torch.testing.assert_close(value, ordinary[name])
    assert "source/replay/loss_mass_share" in grouped
    assert "source/caption/active_loss_weight" in grouped


def test_group_telemetry_separates_active_mass_and_objective_coefficients(monkeypatch):
    module = object.__new__(MultimodalOLMoDDPTrainModule)
    module.loss_group_weights = _COEFFICIENTS
    module.label_ignore_index = -100
    module.device = torch.device("cpu")
    module.dp_group = None
    original = _batch()
    metrics = {}
    module.record_metric = lambda name, value, *args, **kwargs: metrics.__setitem__(name, value)
    module._diagnostics_enabled_for_step = lambda: False
    observed_batches = []
    module._record_data_metrics = observed_batches.append
    monkeypatch.setattr(OLMoDDPTrainModule, "train_batch", lambda *args, **kwargs: None)
    module.train_batch(original)
    assert observed_batches[0] is original
    for group, coefficient in _COEFFICIENTS.items():
        assert metrics[f"group/{group}/objective_weight"] == coefficient
    torch.testing.assert_close(metrics["group/text/active_loss_weight"], torch.tensor(1.85))
    torch.testing.assert_close(metrics["group/vision/active_loss_weight"], torch.tensor(3.75))
