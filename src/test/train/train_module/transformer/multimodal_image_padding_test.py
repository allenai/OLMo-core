import importlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from olmo_core.data.utils import split_batch
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.vision import MultimodalOLMoDDPModel, VisionEncoderConfig
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizerConfig
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModule,
    MultimodalOLMoDDPTrainModuleConfig,
    _trim_microbatch_image_padding,
)


def _batch(crop_counts=(0, 1, 3)):
    size = len(crop_counts)
    images = torch.zeros(size, 9, 4, 14 * 14 * 3)
    pooled = torch.full((size, 5, 4), -1, dtype=torch.long)
    inputs = torch.full((size, 8), 2, dtype=torch.long)
    for row, count in enumerate(crop_counts):
        images[row, :count] = torch.randn(count, 4, 14 * 14 * 3)
        pooled[row, :count] = torch.arange(count * 4).reshape(count, 4)
        inputs[row, :count] = 1
    return {
        "input_ids": inputs,
        "labels": torch.full_like(inputs, 2),
        "loss_masks": torch.ones_like(inputs, dtype=torch.float32),
        "position_ids": torch.arange(8).expand(size, -1),
        "router_token_mask": torch.ones_like(inputs, dtype=torch.bool),
        "images": images,
        "pooled_patches_idx": pooled,
        "image_crop_counts": torch.tensor(crop_counts),
        "pooled_token_counts": torch.tensor(crop_counts),
    }


@pytest.mark.parametrize("size", [1, 2, 3])
def test_trims_after_microbatch_split_without_changing_token_fields(size):
    batch = _batch()
    for microbatch in split_batch(batch, size):
        result = _trim_microbatch_image_padding(microbatch)
        kept = max(int(microbatch["image_crop_counts"].max()), 1)
        assert result["images"].shape[1] == kept
        assert result["pooled_patches_idx"].shape[1] == kept
        for name in ["input_ids", "labels", "loss_masks", "position_ids", "router_token_mask"]:
            assert result[name] is microbatch[name]
        assert microbatch["images"].shape[1] == 9
        assert microbatch["pooled_patches_idx"].shape[1] == 5
        assert (
            result["images"].untyped_storage().data_ptr()
            == microbatch["images"].untyped_storage().data_ptr()
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_prepare_batch_keeps_dummy_vision_path_and_removes_diagnostic_metadata(enabled):
    module = object.__new__(MultimodalOLMoDDPTrainModule)
    module._pp_config = None
    module.response_logits_only = True
    module.trim_microbatch_image_padding = enabled
    batch = _batch((0,))
    inputs, labels, kwargs = module._prepare_batch(dict(batch))
    assert inputs is batch["input_ids"] and labels is batch["labels"]
    assert kwargs["images"].shape == (1, 1 if enabled else 9, 4, 14 * 14 * 3)
    assert kwargs["pooled_patches_idx"].shape == (1, 1 if enabled else 5, 4)
    assert "image_crop_counts" not in kwargs and "pooled_token_counts" not in kwargs
    assert kwargs["response_logits_only"] is True
    assert inputs.shape == (1, 8)


def test_native_text_eval_without_image_tensors_is_unchanged():
    module = object.__new__(MultimodalOLMoDDPTrainModule)
    module._pp_config = None
    module.response_logits_only = False
    module.trim_microbatch_image_padding = True
    inputs = torch.tensor([[2, 3, 4]])
    result, _, kwargs = module._prepare_batch({"input_ids": inputs})
    assert result is inputs and kwargs == {}


@pytest.mark.parametrize("field", ["image_crop_counts", "pooled_token_counts"])
@pytest.mark.parametrize("bad", ["missing", "shape", "negative", "too_large", "float"])
def test_rejects_missing_or_invalid_count_metadata(field, bad):
    batch = _batch()
    if bad == "missing":
        batch.pop(field)
    elif bad == "shape":
        batch[field] = torch.zeros(3, 1, dtype=torch.long)
    elif bad == "negative":
        batch[field][0] = -1
    elif bad == "too_large":
        batch[field][0] = 10
    else:
        batch[field] = batch[field].float()
    with pytest.raises(OLMoConfigurationError, match=field):
        _trim_microbatch_image_padding(batch)


@pytest.mark.parametrize(
    "failure", ["crop_reference", "discarded_pool", "negative_index", "image_rank", "pool_rank"]
)
def test_rejects_trimming_real_features_or_malformed_tensors(failure):
    batch = _batch()
    if failure == "crop_reference":
        batch["pooled_patches_idx"][1, 0, 0] = 4
    elif failure == "discarded_pool":
        batch["pooled_token_counts"][1] = 0
    elif failure == "negative_index":
        batch["pooled_patches_idx"][0, 0, 0] = -2
    elif failure == "image_rank":
        batch["images"] = batch["images"][0]
    else:
        batch["pooled_patches_idx"] = batch["pooled_patches_idx"][0]
    with pytest.raises(OLMoConfigurationError):
        _trim_microbatch_image_padding(batch)


@pytest.mark.parametrize("field", ["attention_dropout", "residual_dropout"])
def test_nonzero_vision_dropout_is_rejected_before_materializing_optimizer(field):
    model = object.__new__(MultimodalOLMoDDPModel)
    torch.nn.Module.__init__(model)
    model.cfg = SimpleNamespace(vision=VisionEncoderConfig(**{field: 0.1}))
    model.lm = SimpleNamespace(tbo=False)
    with pytest.raises(OLMoConfigurationError, match="zero vision"):
        MultimodalOLMoDDPTrainModule(model, trim_microbatch_image_padding=True)


def test_config_is_opt_in_and_old_saved_configs_keep_it_disabled():
    config = MultimodalOLMoDDPTrainModuleConfig(
        rank_microbatch_size=8,
        max_sequence_length=8,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
    )
    assert config.trim_microbatch_image_padding is False
    old = config.as_config_dict()
    old.pop("trim_microbatch_image_padding")
    assert type(config).from_dict(old).trim_microbatch_image_padding is False
    changed = config.merge(["--trim_microbatch_image_padding=true"])
    assert type(config).from_dict(changed.as_config_dict()).trim_microbatch_image_padding is True


@pytest.mark.parametrize("remote_crops", [1, 9])
def test_preserves_collective_order_and_remote_crop_padding(monkeypatch, remote_crops):
    helpers = importlib.import_module("test.nn.vision.multimodal_test")
    multimodal = importlib.import_module("olmo_core.nn.vision.multimodal")
    model = helpers._tiny_multimodal_cfg().build(init_device="cpu").float()
    events = []

    def all_reduce(value, *, op):
        assert op == torch.distributed.ReduceOp.MAX
        events.append("crop_max")
        value.fill_(max(int(value.item()), remote_crops))

    monkeypatch.setattr(multimodal, "is_distributed", lambda: True)
    monkeypatch.setattr(multimodal.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(multimodal, "barrier", lambda: events.append("barrier"))
    handle = model.vision.register_forward_pre_hook(
        lambda _, args: events.append(("vision", args[0].shape[0]))
    )
    batch = _batch((0,))
    for inputs, expected_crops in [
        (batch, 9),
        (_trim_microbatch_image_padding(batch), remote_crops),
    ]:
        events.clear()
        output = model.encode_images(inputs["images"], inputs["pooled_patches_idx"])
        assert output.shape[0] == 0
        assert events == ["crop_max", ("vision", expected_crops), "barrier"]
    handle.remove()


@pytest.mark.parametrize("crop_counts", [(0,), (1,), (1, 3)])
def test_real_vision_connector_forward_and_gradient_oracle(crop_counts):
    helpers = importlib.import_module("test.nn.vision.multimodal_test")
    torch.manual_seed(42)
    original = helpers._tiny_multimodal_cfg().build(init_device="cpu").float()
    original.train()
    trimmed = deepcopy(original)
    batch = _batch(crop_counts)
    result = _trim_microbatch_image_padding(batch)
    outputs = []
    crop_work = []
    for model, inputs in [(original, batch), (trimmed, result)]:
        observed = []
        handle = model.vision.register_forward_pre_hook(
            lambda _, args, observed=observed: observed.append(args[0].shape[0])
        )
        output = model(
            inputs["input_ids"],
            images=inputs["images"],
            pooled_patches_idx=inputs["pooled_patches_idx"],
        )
        output.float().square().mean().backward()
        handle.remove()
        outputs.append(output)
        crop_work.append(sum(observed))
    assert crop_work == [len(crop_counts) * 9, len(crop_counts) * max(max(crop_counts), 1)]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
    for (name, first), (other_name, second) in zip(
        original.named_parameters(), trimmed.named_parameters()
    ):
        assert name == other_name
        if first.grad is None:
            assert second.grad is None
        else:
            torch.testing.assert_close(first.grad, second.grad, rtol=2e-4, atol=1e-6, msg=name)
    if crop_counts == (0,):
        for name, param in trimmed.named_parameters():
            if name.startswith(("vision.", "connector.")):
                assert param.grad is not None and not bool(param.grad.any())
