import numpy as np
import pytest
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.distributed.tensor import DTensor

from olmo_core.config import DType
from olmo_core.data.multimodal import MultimodalCollatorConfig
from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.eval import MultimodalLMEvaluator
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import TransformerDataParallelWrappingStrategy
from olmo_core.nn.vision import (
    MultimodalLMConfig,
    VisionConnectorConfig,
    VisionEncoderConfig,
    VisionEncoderType,
)
from olmo_core.optim import AdamWConfig, SkipStepAdamWConfig, SkipStepOptimizer
from olmo_core.testing.distributed import run_distributed_test
from olmo_core.train.train_module import (
    MultimodalTransformerTrainModuleConfig,
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
)

VOCAB_SIZE = 256
IMAGE_PATCH_TOKEN_ID = 1
SEQUENCE_LENGTH = 8


def _multimodal_config() -> MultimodalLMConfig:
    lm = TransformerConfig.olmo2_1M(vocab_size=VOCAB_SIZE)
    vision = VisionEncoderConfig(
        name=VisionEncoderType.openai,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_emb_dim=32,
        image_num_heads=2,
        image_num_key_value_heads=2,
        image_num_layers=2,
        image_head_dim=16,
        image_mlp_dim=64,
        image_num_pos=5,
        image_norm_eps=1e-5,
    )
    connector = VisionConnectorConfig.from_vision_encoder(
        vision,
        output_dim=lm.d_model,
        mlp_hidden_size=32,
    )
    return MultimodalLMConfig(
        lm=lm,
        vision=vision,
        connector=connector,
        image_patch_token_id=IMAGE_PATCH_TOKEN_ID,
    )


def _build_train_module(*, init_device: str = "cpu", **kwargs):
    model = _multimodal_config().build(init_device=init_device)
    optim = kwargs.pop("optim", AdamWConfig(lr=1e-4))
    kwargs.setdefault("new_component_init_seed", 6198)
    config = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=optim,
        vision_activation_checkpointing=False,
        connector_activation_checkpointing=False,
        **kwargs,
    )
    return config.build(model, device=torch.device("cpu"))


def _text_batch():
    return {
        "input_ids": torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]]),
        "labels": torch.tensor([[3, 4, 5, 6, 7, 8, 9, -100]]),
        "loss_masks": torch.tensor([[0.0, 1.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0]]),
        "router_token_mask": torch.ones((1, SEQUENCE_LENGTH), dtype=torch.bool),
    }


class _Trainer:
    def __init__(self):
        self.global_step = 1
        self.metrics = {}

    def record_metric(self, name, value, *, reduce_type=None, namespace=None, **kwargs):
        del kwargs
        self.metrics[f"{namespace}/{name}"] = (value, reduce_type)

    def record_ce_loss(self, value, *, reduce_type=None):
        self.metrics["train/CE loss"] = (value, reduce_type)


def test_generic_config_round_trip_and_selective_embedding_rows():
    config = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(lr=1e-4),
        freeze_params=["vision.*"],
        diagnostics_interval=10,
        train_embedding_rows=[3, 1],
        source_loss_mass_targets={"caption": 0.75, "native": 0.25},
        new_component_init_seed=6198,
    )
    serialized = config.as_dict()
    assert serialized["diagnostics_interval"] == 10
    assert serialized["train_embedding_rows"] == [3, 1]
    assert serialized["source_loss_mass_targets"] == {"caption": 0.75, "native": 0.25}
    assert serialized["new_component_init_seed"] == 6198

    train_module = _build_train_module(
        freeze_params=["vision.*"],
        train_embedding_rows=[3, 1],
    )
    assert train_module.multimodal_model is train_module.model
    assert train_module.train_embedding_rows == (1, 3)

    embedding_ids = torch.arange(VOCAB_SIZE).unsqueeze(0)
    train_module.multimodal_model.lm.embeddings(embedding_ids).sum().backward()
    grad = train_module.multimodal_model.lm.embeddings.weight.grad
    assert grad is not None
    assert torch.nonzero(grad.abs().sum(dim=1)).flatten().tolist() == [1, 3]

    train_module._set_model_mode("train")
    assert not train_module.multimodal_model.vision.training
    assert train_module.multimodal_model.connector.training
    assert train_module.multimodal_model.lm.training


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"freeze_params": ["missing.*"]}, "does not match"),
        ({"diagnostics_interval": 0}, "must be positive"),
        ({"train_embedding_rows": [1, 1]}, "unique IDs"),
        ({"train_embedding_rows": [VOCAB_SIZE]}, "outside the LM embedding"),
        ({"source_loss_mass_targets": {"caption": 0.4}}, "sum to one"),
        ({"new_component_init_seed": -1}, "non-negative integer"),
    ],
)
def test_generic_config_rejects_ambiguous_diagnostics_contracts(kwargs, message):
    with pytest.raises(OLMoConfigurationError, match=message):
        _build_train_module(**kwargs)


def test_multimodal_root_hides_activation_checkpoint_wrapper_names():
    model = _multimodal_config().build(init_device="cpu")
    model.lm.blocks["0"].register_buffer("logical_name_test_buffer", torch.ones(1))
    parameters_before = dict(model.named_parameters())
    buffers_before = dict(model.named_buffers())

    model.lm.apply_activation_checkpointing(TransformerActivationCheckpointingConfig().mode)

    parameters_after = dict(model.named_parameters())
    buffers_after = dict(model.named_buffers())
    assert parameters_after.keys() == parameters_before.keys()
    assert buffers_after.keys() == buffers_before.keys()
    assert all(parameters_after[name] is value for name, value in parameters_before.items())
    assert all(buffers_after[name] is value for name, value in buffers_before.items())
    assert not any(
        "_checkpoint_wrapped_module" in name
        for name in (*parameters_after.keys(), *buffers_after.keys())
    )
    assert "lm.blocks.0.logical_name_test_buffer" in model.state_dict()


@pytest.mark.parametrize("init_device", ["cpu", "meta"])
def test_activation_checkpointing_preserves_logical_names_and_trainability(init_device):
    model = _multimodal_config().build(init_device=init_device)
    initially_frozen_name = next(
        name for name, _ in model.named_parameters() if name.startswith("connector.")
    )
    dict(model.named_parameters())[initially_frozen_name].requires_grad_(False)

    train_module = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(lr=1e-4),
        freeze_params=["lm.blocks.0.attention.w_q.weight"],
        ac_config=TransformerActivationCheckpointingConfig(),
        compile_model=True,
        vision_activation_checkpointing=False,
        connector_activation_checkpointing=False,
        new_component_init_seed=6198,
    ).build(model, device=torch.device("cpu"))

    parameters = dict(train_module.model.named_parameters())
    state = train_module.state_dict(optim=False)["model"]
    assert not any("_checkpoint_wrapped_module" in name for name in parameters)
    assert set(parameters).issubset(state)
    assert not any("_checkpoint_wrapped_module" in name for name in state)
    assert not parameters["lm.blocks.0.attention.w_q.weight"].requires_grad
    assert parameters["lm.blocks.0.attention.w_k.weight"].requires_grad
    assert not parameters[initially_frozen_name].requires_grad


def test_generic_multimodal_eval_returns_summed_weighted_loss_and_response_logits():
    train_module = _build_train_module(response_logits_only=True)
    batch = _text_batch()
    original = {name: value.clone() for name, value in batch.items()}

    output = train_module.eval_batch(batch, return_response_logits=True)

    assert output.ce_loss.ndim == 0
    assert output.logits is not None
    assert output.logits.shape[0] == int((batch["loss_masks"] > 0).sum())
    assert not output.ce_loss.requires_grad
    assert not output.logits.requires_grad
    for name, value in original.items():
        torch.testing.assert_close(batch[name], value)

    evaluator = MultimodalLMEvaluator(
        name="generic-mm",
        batches=[],
        device=torch.device("cpu"),
    )
    evaluator.update_metrics(batch, output.ce_loss, output.logits)
    assert torch.isfinite(evaluator.compute_metrics()["CE loss"])


def test_generic_training_records_source_input_and_component_diagnostics():
    train_module = _build_train_module(
        diagnostics_interval=1,
        source_loss_mass_targets={"caption": 1.0},
    )
    trainer = _Trainer()
    train_module._trainer = trainer
    batch = {
        "input_ids": torch.tensor([[IMAGE_PATCH_TOKEN_ID, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long),
        "labels": torch.tensor([[3, 4, 5, 6, 7, 8, 9, -100]]),
        "loss_masks": torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]),
        "router_token_mask": torch.ones((1, SEQUENCE_LENGTH), dtype=torch.bool),
        "position_ids": torch.arange(SEQUENCE_LENGTH).unsqueeze(0),
        "token_type_ids": torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]),
        "source_names": ["caption"],
        "images": torch.randn(1, 1, 4, 14 * 14 * 3),
        "pooled_patches_idx": torch.arange(4).reshape(1, 1, 4),
        "image_crop_counts": torch.ones(1, dtype=torch.long),
        "pooled_token_counts": torch.ones(1, dtype=torch.long),
    }

    train_module.train_batch(batch)
    train_module.optim_step()

    assert "data/source/caption/loss_mass_share" in trainer.metrics
    assert "multimodal/text embedding RMS" in trainer.metrics
    assert "multimodal/connector output RMS" in trainer.metrics
    assert "multimodal/spliced image embedding RMS" in trainer.metrics
    assert "optim/vision grad norm" in trainer.metrics
    assert "optim/connector grad norm" in trainer.metrics
    assert "optim/LM sequence mixers grad norm" in trainer.metrics


def test_unpacked_source_metadata_is_collated_recorded_and_never_forwarded():
    example = {
        "input_ids": np.arange(SEQUENCE_LENGTH, dtype=np.int64),
        "labels": np.array([1, 2, 3, 4, 5, 6, 7, -100], dtype=np.int64),
        "loss_masks": np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.float32),
        "position_ids": np.arange(SEQUENCE_LENGTH, dtype=np.int64),
        "token_type_ids": np.zeros(SEQUENCE_LENGTH, dtype=np.int64),
        "images": np.zeros((0, 4, 14 * 14 * 3), dtype=np.float32),
        "pooled_patches_idx": np.zeros((0, 4), dtype=np.int64),
        "_source_name": "caption",
    }
    batch = MultimodalCollatorConfig(pad_token_id=0, pad_sequence_length=SEQUENCE_LENGTH).build()(
        [example]
    )
    assert batch["source_names"] == ["caption"]
    assert "pack_source_names" not in batch
    assert "example_ids" not in batch

    train_module = _build_train_module(
        diagnostics_interval=1, source_loss_mass_targets={"caption": 1.0}
    )
    trainer = _Trainer()
    train_module._trainer = trainer
    train_module._record_data_metrics(batch)
    assert "data/source/caption/loss_mass_share" in trainer.metrics

    train_module._forbid_packed_multimodal_metadata = True
    _, _, _, model_kwargs = train_module._prepare_batch(dict(batch))
    assert "router_token_mask" not in model_kwargs
    assert "source_names" not in model_kwargs
    assert "example_ids" not in model_kwargs

    packed_batch = dict(batch)
    packed_batch["example_ids"] = torch.zeros_like(batch["input_ids"])
    with pytest.raises(OLMoConfigurationError, match="forbids packed/branched metadata"):
        train_module._prepare_batch(packed_batch)


def test_skip_step_receives_global_mean_loss_and_grad_norm_each_step():
    train_module = _build_train_module(
        optim=SkipStepAdamWConfig(lr=1e-4, foreach=False, rolling_interval_length=8),
        max_grad_norm=1.0,
    )
    train_module._trainer = _Trainer()
    assert isinstance(train_module.optim, SkipStepOptimizer)

    for _ in range(2):
        train_module.train_batch(_text_batch())
        train_module.optim_step()
        train_module.zero_grads()

    assert len(train_module.optim._losses) == 2
    assert len(train_module.optim._grad_norms) == 2
    assert train_module.optim.latest_loss is not None
    assert train_module.optim.latest_grad_norm is not None
    assert torch.isfinite(train_module.optim.latest_loss)
    assert torch.isfinite(train_module.optim.latest_grad_norm)


def test_strict_vision_load_optimizer_identity_and_image_row_reset():
    train_module = _build_train_module()
    vision_state = {
        name: torch.full_like(value, 0.25)
        for name, value in train_module.multimodal_model.vision.state_dict().items()
    }
    train_module.load_vision_state_dict(vision_state)
    train_module.assert_vision_optimizer_state_synced()
    for name, value in train_module.multimodal_model.vision.state_dict().items():
        torch.testing.assert_close(value, vision_state[name], rtol=0, atol=0)
    bad_vision_state = dict(vision_state)
    bad_key = next(iter(bad_vision_state))
    bad_vision_state[bad_key] = bad_vision_state[bad_key].double()
    with pytest.raises(OLMoConfigurationError, match="dtype mismatch"):
        train_module.load_vision_state_dict(bad_vision_state)

    embeddings = train_module.multimodal_model.lm.embeddings.weight
    output = train_module.multimodal_model.lm.lm_head.w_out.weight
    before_embeddings = embeddings.detach().clone()
    before_output = output.detach().clone()
    train_module.reset_image_token_rows([IMAGE_PATCH_TOKEN_ID], seed=6198, reset_output_rows=False)
    assert not torch.equal(
        embeddings[IMAGE_PATCH_TOKEN_ID], before_embeddings[IMAGE_PATCH_TOKEN_ID]
    )
    torch.testing.assert_close(embeddings[2:], before_embeddings[2:], rtol=0, atol=0)
    torch.testing.assert_close(output, before_output, rtol=0, atol=0)


def test_eval_only_build_skips_optimizer_and_loads_strict_model_state():
    model = _multimodal_config().build(init_device="cpu")
    config = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(lr=1e-4),
        vision_activation_checkpointing=False,
        connector_activation_checkpointing=False,
        new_component_init_seed=6198,
    )
    train_module = config.build(model, device=torch.device("cpu"), eval_only=True)
    assert train_module.optim is None
    state = train_module.state_dict(optim=False)
    train_module.load_state_dict(state)
    assert torch.isfinite(train_module.eval_batch(_text_batch()).ce_loss)
    with pytest.raises(RuntimeError, match="eval_only"):
        train_module.train_batch(_text_batch())


def test_meta_build_and_strict_model_only_parent_load(tmp_path):
    parent = _multimodal_config().lm.build(init_device="cpu")
    parent.init_weights(
        device=torch.device("cpu"),
        max_seq_len=SEQUENCE_LENGTH,
        max_local_microbatch_size=SEQUENCE_LENGTH,
    )
    with torch.no_grad():
        parent.embeddings.weight.fill_(0.125)
    save_state_dict(
        tmp_path,
        {"model": dist_cp_sd.get_model_state_dict(parent)},
    )

    train_module = _build_train_module(
        init_device="meta",
        ac_config=TransformerActivationCheckpointingConfig(),
    )
    assert {param.device.type for param in train_module.model.parameters()} == {"cpu"}
    assert all(torch.isfinite(param).all() for param in train_module.model.parameters())
    before_vision = {
        name: value.detach().clone()
        for name, value in train_module.multimodal_model.vision.state_dict().items()
    }
    model_keys = set(train_module.state_dict(optim=False)["model"])
    parameter_keys = set(dict(train_module.model.named_parameters()))
    loaded_keys = {key for key in model_keys if key.startswith("lm.")}
    missing_keys = model_keys - loaded_keys
    loaded_parameter_keys = loaded_keys & parameter_keys
    mapping = {key: key.removeprefix("lm.") for key in loaded_keys}

    receipt = train_module.load_parent_model_state_dict(
        tmp_path,
        current_to_checkpoint_key_mapping=mapping,
        expected_loaded_model_keys=loaded_keys,
        expected_missing_model_keys=missing_keys,
        expected_loaded_parameter_keys=loaded_parameter_keys,
    )

    assert receipt["loaded_model_tensor_count"] == len(loaded_keys)
    assert receipt["loaded_parameter_count"] == len(loaded_parameter_keys)
    assert receipt["loaded_tensor_dtype_counts"] == {"torch.float32": len(loaded_keys)}
    assert receipt["loaded_tensor_layout_counts"] == {"torch.strided": len(loaded_keys)}
    torch.testing.assert_close(
        train_module.multimodal_model.lm.embeddings.weight,
        torch.full_like(train_module.multimodal_model.lm.embeddings.weight, 0.125),
    )
    for name, value in train_module.multimodal_model.vision.state_dict().items():
        torch.testing.assert_close(value, before_vision[name], rtol=0, atol=0)
    assert train_module.optim.state == {}
    assert train_module.state_dict_load_opts.strict
    assert not any(
        "_checkpoint_wrapped_module" in name for name in receipt["loaded_parameter_keys"]
    )


def test_meta_new_component_seed_is_explicit_and_reproducible():
    first = _build_train_module(init_device="meta", new_component_init_seed=6198)
    second = _build_train_module(init_device="meta", new_component_init_seed=6198)
    third = _build_train_module(init_device="meta", new_component_init_seed=6199)
    first_connector = first.multimodal_model.connector.state_dict()
    second_connector = second.multimodal_model.connector.state_dict()
    third_connector = third.multimodal_model.connector.state_dict()

    for name, value in first_connector.items():
        torch.testing.assert_close(value, second_connector[name], rtol=0, atol=0)
    assert any(
        not torch.equal(value, third_connector[name])
        for name, value in first_connector.items()
        if value.is_floating_point()
    )

    model = _multimodal_config().build(init_device="meta")
    with pytest.raises(OLMoConfigurationError, match="required for a meta-initialized"):
        MultimodalTransformerTrainModuleConfig(
            rank_microbatch_size=SEQUENCE_LENGTH,
            max_sequence_length=SEQUENCE_LENGTH,
            optim=AdamWConfig(lr=1e-4),
            vision_activation_checkpointing=False,
            connector_activation_checkpointing=False,
        ).build(model, device=torch.device("cpu"))


def test_parent_load_rejects_nonexact_missing_inventory(tmp_path):
    parent = _multimodal_config().lm.build(init_device="cpu")
    save_state_dict(tmp_path, {"model": dist_cp_sd.get_model_state_dict(parent)})
    train_module = _build_train_module()
    model_keys = set(train_module.state_dict(optim=False)["model"])
    loaded_keys = {key for key in model_keys if key.startswith("lm.")}
    missing_keys = model_keys - loaded_keys
    omitted = next(iter(missing_keys))

    with pytest.raises(OLMoConfigurationError, match="do not exactly partition"):
        train_module.load_parent_model_state_dict(
            tmp_path,
            current_to_checkpoint_key_mapping={key: key.removeprefix("lm.") for key in loaded_keys},
            expected_loaded_model_keys=loaded_keys,
            expected_missing_model_keys=missing_keys - {omitted},
            expected_loaded_parameter_keys=loaded_keys
            & set(dict(train_module.model.named_parameters())),
        )

    with pytest.raises(OLMoConfigurationError, match="cover the loaded inventory exactly"):
        mapping = {key: key.removeprefix("lm.") for key in loaded_keys}
        mapping.pop(next(iter(mapping)))
        train_module.load_parent_model_state_dict(
            tmp_path,
            current_to_checkpoint_key_mapping=mapping,
            expected_loaded_model_keys=loaded_keys,
            expected_missing_model_keys=missing_keys,
            expected_loaded_parameter_keys=loaded_keys
            & set(dict(train_module.model.named_parameters())),
        )


def test_parent_load_rejects_dtype_mismatch(tmp_path):
    parent = _multimodal_config().lm.build(init_device="cpu").double()
    save_state_dict(tmp_path, {"model": dist_cp_sd.get_model_state_dict(parent)})
    train_module = _build_train_module()
    model_keys = set(train_module.state_dict(optim=False)["model"])
    loaded_keys = {key for key in model_keys if key.startswith("lm.")}
    missing_keys = model_keys - loaded_keys

    with pytest.raises(OLMoConfigurationError, match="dtype mismatch"):
        train_module.load_parent_model_state_dict(
            tmp_path,
            current_to_checkpoint_key_mapping={key: key.removeprefix("lm.") for key in loaded_keys},
            expected_loaded_model_keys=loaded_keys,
            expected_missing_model_keys=missing_keys,
            expected_loaded_parameter_keys=loaded_keys
            & set(dict(train_module.model.named_parameters())),
        )


def _run_hsdp_meta_row_mask_step(parent_checkpoint_dir: str):
    model_config = _multimodal_config()
    model_config.lm.embed_scale = 3.0
    model_config.lm.embedding_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
    )
    model = model_config.build(init_device="meta")
    train_module = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(lr=1e-4, weight_decay=0.0),
        freeze_params=[
            "vision.*",
            "lm.embedding_norm.*",
            "lm.blocks.*",
            "lm.lm_head.*",
        ],
        train_embedding_rows=[IMAGE_PATCH_TOKEN_ID],
        new_component_init_seed=6198,
        diagnostics_interval=1,
        vision_activation_checkpointing=False,
        connector_activation_checkpointing=False,
        ac_config=TransformerActivationCheckpointingConfig(),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            num_replicas=1,
            shard_degree=2,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
    ).build(model, device=torch.device("cuda"))
    train_module._trainer = _Trainer()
    train_module.assert_vision_optimizer_state_synced()
    sharded_vision_state = dist_cp_sd.get_model_state_dict(
        train_module.multimodal_model.vision,
        options=dist_cp_sd.StateDictOptions(strict=True),
    )
    vision_state = {
        name: torch.full(value.shape, 0.25, dtype=value.dtype)
        for name, value in sharded_vision_state.items()
    }
    train_module.load_vision_state_dict(vision_state)
    reloaded_vision_state = dist_cp_sd.get_model_state_dict(
        train_module.multimodal_model.vision,
        options=dist_cp_sd.StateDictOptions(strict=True),
    )
    for name, value in reloaded_vision_state.items():
        local_value = value.to_local() if isinstance(value, DTensor) else value
        torch.testing.assert_close(local_value, torch.full_like(local_value, 0.25), rtol=0, atol=0)
    model_keys = set(train_module.state_dict(optim=False)["model"])
    parameter_keys = set(dict(train_module.model.named_parameters()))
    loaded_keys = {key for key in model_keys if key.startswith("lm.")}
    missing_keys = model_keys - loaded_keys
    receipt = train_module.load_parent_model_state_dict(
        parent_checkpoint_dir,
        current_to_checkpoint_key_mapping={key: key.removeprefix("lm.") for key in loaded_keys},
        expected_loaded_model_keys=loaded_keys,
        expected_missing_model_keys=missing_keys,
        expected_loaded_parameter_keys=loaded_keys & parameter_keys,
        process_group=train_module.dp_process_group,
    )
    assert receipt["loaded_tensor_dtype_counts"] == {"torch.float32": len(loaded_keys)}
    embedding_value = train_module.multimodal_model.lm.embeddings.weight.full_tensor()
    torch.testing.assert_close(embedding_value, torch.full_like(embedding_value, 0.125))
    train_module.reset_image_token_rows([IMAGE_PATCH_TOKEN_ID], seed=6198, reset_output_rows=False)
    embedding_before = train_module.multimodal_model.lm.embeddings.weight.full_tensor().clone()
    embedding_norm_weight = train_module.multimodal_model.lm.embedding_norm.weight
    assert isinstance(embedding_norm_weight, DTensor)
    assert not embedding_norm_weight.requires_grad
    connector_parameters = {
        name: parameter
        for name, parameter in train_module.multimodal_model.connector.named_parameters()
        if parameter.requires_grad
    }
    connector_before = {
        name: parameter.full_tensor().clone() for name, parameter in connector_parameters.items()
    }
    batch = {
        "input_ids": torch.tensor(
            [[IMAGE_PATCH_TOKEN_ID, 3, 4, 5, 6, 7, 8, 9]],
            dtype=torch.long,
            device="cuda",
        ),
        "labels": torch.tensor([[3, 4, 5, 6, 7, 8, 9, -100]], device="cuda"),
        "loss_masks": torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]], device="cuda"),
        "router_token_mask": torch.ones((1, SEQUENCE_LENGTH), dtype=torch.bool, device="cuda"),
        "position_ids": torch.arange(SEQUENCE_LENGTH, device="cuda").unsqueeze(0),
        "token_type_ids": torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], device="cuda"),
        "images": torch.randn(1, 1, 4, 14 * 14 * 3, device="cuda"),
        "pooled_patches_idx": torch.arange(4, device="cuda").reshape(1, 1, 4),
    }
    # Only rank 1 contributes a loss. The image-row gradient owned by rank 0 can therefore
    # be non-zero only if the embedding module's FSDP forward/backward hooks synchronize it.
    # A direct ``DTensor.full_tensor()`` embedding lookup leaves this row exactly zero.
    if torch.distributed.get_rank() == 0:
        batch["loss_masks"].zero_()

    train_module.train_batch(batch)
    grad = train_module.multimodal_model.lm.embeddings.weight.grad
    assert isinstance(grad, DTensor)
    full_grad = grad.full_tensor()
    nonzero_rows = torch.nonzero(full_grad.abs().sum(dim=1)).flatten().tolist()
    assert nonzero_rows == [IMAGE_PATCH_TOKEN_ID]
    train_module.optim_step()
    embedding_after = train_module.multimodal_model.lm.embeddings.weight.full_tensor()
    assert not torch.equal(
        embedding_after[IMAGE_PATCH_TOKEN_ID], embedding_before[IMAGE_PATCH_TOKEN_ID]
    )
    non_image_rows = torch.arange(VOCAB_SIZE) != IMAGE_PATCH_TOKEN_ID
    torch.testing.assert_close(
        embedding_after[non_image_rows],
        embedding_before[non_image_rows],
        rtol=0,
        atol=0,
    )
    assert any(
        not torch.equal(parameter.full_tensor(), connector_before[name])
        for name, parameter in connector_parameters.items()
    )
    assert "multimodal/connector output RMS" in train_module.trainer.metrics
    assert "optim/connector grad norm" in train_module.trainer.metrics
    assert "optim/input embeddings grad norm" in train_module.trainer.metrics
    assert "optim/vision grad norm" not in train_module.trainer.metrics
    assert "optim/LM sequence mixers grad norm" not in train_module.trainer.metrics


@pytest.mark.gpu
def test_generic_hsdp_meta_build_and_row_mask_step(tmp_path):
    model_config = _multimodal_config()
    model_config.lm.embed_scale = 3.0
    model_config.lm.embedding_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
    )
    parent = model_config.lm.build(init_device="cpu")
    parent.init_weights(
        device=torch.device("cpu"),
        max_seq_len=SEQUENCE_LENGTH,
        max_local_microbatch_size=SEQUENCE_LENGTH,
    )
    with torch.no_grad():
        parent.embeddings.weight.fill_(0.125)
    save_state_dict(tmp_path, {"model": dist_cp_sd.get_model_state_dict(parent)})
    run_distributed_test(
        _run_hsdp_meta_row_mask_step,
        func_args=[str(tmp_path)],
        world_size=2,
        backend="nccl",
        start_method="spawn",
    )
