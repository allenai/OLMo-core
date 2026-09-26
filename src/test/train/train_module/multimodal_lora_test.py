"""LoRA wiring on ``MultimodalTransformerTrainModule``.

Covers the three things that can silently go wrong end to end: which parameters end up in
the optimizer, whether a base (non-LoRA) checkpoint still loads, and whether the
``strict=False`` load that makes that possible would also swallow a genuinely missing
base weight.
"""

from typing import Dict

import pytest
import torch

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lora import LoRAConfig, LoRALinear, lora_param_names, merge_lora_
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.vision import (
    MultimodalLMConfig,
    VisionConnectorConfig,
    VisionEncoderConfig,
    VisionEncoderType,
)
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train.train_module import MultimodalTransformerTrainModuleConfig

BASE_VOCAB, EXTRA_VOCAB = 200, 8
IMAGE_PATCH_ID = BASE_VOCAB + 2
SEQ_LEN, N_POOLED = 16, 4
RANK = 4

LORA_TARGETS = [
    "lm.blocks.*.attention.w_q",
    "lm.blocks.*.attention.w_k",
    "lm.blocks.*.attention.w_v",
    "lm.blocks.*.attention.w_out",
    "lm.blocks.*.feed_forward.w1",
    "lm.blocks.*.feed_forward.w2",
    "lm.blocks.*.feed_forward.w3",
]


def _model_config() -> MultimodalLMConfig:
    lm = TransformerConfig.llama_like(
        d_model=16,
        vocab_size=BASE_VOCAB,
        n_layers=2,
        n_heads=2,
        n_extra_vocab=EXTRA_VOCAB,
        attn_backend=AttentionBackendName.torch,
    )
    vision = VisionEncoderConfig(
        name=VisionEncoderType.siglip,
        use_cls_token=False,
        patch_embedding_bias=True,
        use_pre_ln=False,
        image_default_input_size=(56, 56),
        image_patch_size=14,
        image_emb_dim=32,
        image_num_heads=2,
        image_num_key_value_heads=2,
        image_num_layers=2,
        image_head_dim=16,
        image_mlp_dim=64,
        image_num_pos=16,
        image_norm_eps=1e-5,
    )
    connector = VisionConnectorConfig.from_vision_encoder(
        vision, output_dim=lm.d_model, mlp_hidden_size=64
    )
    return MultimodalLMConfig(
        lm=lm, vision=vision, connector=connector, image_patch_token_id=IMAGE_PATCH_ID
    )


class _StubTrainer:
    def record_metric(self, *args, **kwargs) -> None:
        pass

    def record_ce_loss(self, *args, **kwargs) -> None:
        pass


def _build_model() -> torch.nn.Module:
    model = _model_config().build(init_device="cpu")
    model.lm.init_weights(max_seq_len=SEQ_LEN, device=torch.device("cpu"))
    model.vision.reset_parameters()
    model.connector.reset_parameters()
    return model


def _train_module(use_lora: bool, model=None):
    """Mirrors the two branches `Molmo2-Stage2.build_config` constructs."""
    if use_lora:
        group_overrides = [
            OptimGroupOverride(
                params=["vision_backbone.connector.*"],
                opts=dict(lr=5e-6, weight_decay=0.0, scheduler_name="connector"),
            ),
            OptimGroupOverride(
                params=["lm.*.lora_A", "lm.*.lora_B"],
                opts=dict(lr=2e-4, weight_decay=0.0, scheduler_name="lora"),
            ),
        ]
        freeze_params = ["lm.*", "vision_backbone.vision.*"]
        lora = LoRAConfig(rank=RANK, alpha=2.0 * RANK, target_modules=list(LORA_TARGETS))
    else:
        group_overrides = [
            OptimGroupOverride(
                params=["vision_backbone.connector.*"],
                opts=dict(lr=5e-6, weight_decay=0.0, scheduler_name="connector"),
            ),
            OptimGroupOverride(
                params=["vision_backbone.vision.*"],
                opts=dict(lr=5e-6, weight_decay=0.0, scheduler_name="vision"),
            ),
        ]
        freeze_params = None
        lora = None

    config = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=SEQ_LEN,
        max_sequence_length=SEQ_LEN,
        optim=AdamWConfig(lr=1e-5, group_overrides=group_overrides),
        freeze_params=freeze_params,
        lora=lora,
        z_loss_multiplier=1e-4,
        compile_model=False,
        response_logits_only=True,
    )
    train_module = config.build(model if model is not None else _build_model())
    train_module._trainer = _StubTrainer()  # type: ignore[assignment]
    return train_module


def _batch() -> Dict[str, torch.Tensor]:
    torch.manual_seed(0)
    input_ids = torch.randint(0, BASE_VOCAB, (1, SEQ_LEN))
    input_ids[0, 2 : 2 + N_POOLED] = IMAGE_PATCH_ID
    loss_masks = torch.zeros(1, SEQ_LEN)
    loss_masks[:, SEQ_LEN // 2 :] = 1.0
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "loss_masks": loss_masks,
        "images": torch.randn(1, 1, 16, 3 * 14 * 14),
        "pooled_patches_idx": torch.zeros(1, N_POOLED, 4, dtype=torch.long),
    }


def test_only_adapters_and_connector_are_optimized():
    tm = _train_module(use_lora=True)
    optimized = {id(p) for group in tm.optim.param_groups for p in group["params"]}
    trainable = {n for n, p in tm.model.named_parameters() if p.requires_grad}
    expected = set(lora_param_names(tm.model)) | {
        n for n, _ in tm.model.named_parameters() if n.startswith("vision_backbone.connector.")
    }
    assert trainable == expected
    assert len(optimized) == len(expected)
    # The whole LM and the whole vision encoder are frozen.
    assert not any(
        p.requires_grad
        for n, p in tm.model.named_parameters()
        if n.startswith("vision_backbone.vision.")
    )


def test_lora_uses_far_fewer_trainable_params_than_full_finetune():
    lora_tm = _train_module(use_lora=True)
    full_tm = _train_module(use_lora=False)
    n_lora = sum(p.numel() for p in lora_tm.model.parameters() if p.requires_grad)
    n_full = sum(p.numel() for p in full_tm.model.parameters() if p.requires_grad)
    assert n_lora < n_full / 2


def test_scheduler_groups_match_the_optimizer_groups():
    """A `scheduler_name` with no matching param group, or vice versa, is a silent misconfig."""
    tm = _train_module(use_lora=True)
    names = {g.get("scheduler_name") for g in tm.optim.param_groups}
    assert "lora" in names and "connector" in names
    assert "vision" not in names


def test_step_zero_forward_matches_the_unadapted_model():
    """`lora_B` starts at zero, so adapters must not perturb the initial forward at all.

    This is what makes "did the base checkpoint load correctly?" testable on a real run:
    a LoRA arm and a full-finetune arm must report the same step-0 loss.
    """
    torch.manual_seed(0)
    model = _build_model()
    batch = _batch()
    baseline = _train_module(use_lora=False, model=model)
    baseline.model.eval()
    with torch.no_grad():
        before = baseline.model(
            batch["input_ids"],
            images=batch["images"],
            pooled_patches_idx=batch["pooled_patches_idx"],
        ).clone()

    adapted = _train_module(use_lora=True, model=model)
    adapted.model.eval()
    with torch.no_grad():
        after = adapted.model(
            batch["input_ids"],
            images=batch["images"],
            pooled_patches_idx=batch["pooled_patches_idx"],
        )
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_train_step_updates_adapters_and_leaves_base_weights_alone():
    tm = _train_module(use_lora=True)
    w_q = tm.model.lm.blocks["0"].attention.w_q
    assert isinstance(w_q, LoRALinear)
    base_before = w_q.weight.detach().clone()
    lora_b_before = w_q.lora_B.detach().clone()

    tm.train_batch(_batch())
    tm.optim_step()

    torch.testing.assert_close(w_q.weight, base_before, rtol=0, atol=0)
    assert not torch.equal(w_q.lora_B, lora_b_before), "adapters did not move"


def test_merge_preserves_the_adapted_forward_and_the_plain_key_space():
    tm = _train_module(use_lora=True)
    tm.train_batch(_batch())
    tm.optim_step()

    batch = _batch()
    tm.model.eval()
    with torch.no_grad():
        adapted = tm.model(
            batch["input_ids"],
            images=batch["images"],
            pooled_patches_idx=batch["pooled_patches_idx"],
        ).clone()

    # Scoped to `lm.*`, which is where LoRA operates. The vision/connector names differ
    # between the two branches only because activation checkpointing wraps a trainable
    # submodule and skips a frozen one -- nothing to do with adapters.
    reference_keys = {n for n, _ in _build_model().named_parameters() if n.startswith("lm.")}
    merge_lora_(tm.model)
    assert {n for n, _ in tm.model.named_parameters() if n.startswith("lm.")} == reference_keys
    assert lora_param_names(tm.model) == []
    assert not any(isinstance(m, LoRALinear) for m in tm.model.modules())
    with torch.no_grad():
        merged = tm.model(
            batch["input_ids"],
            images=batch["images"],
            pooled_patches_idx=batch["pooled_patches_idx"],
        )
    torch.testing.assert_close(merged, adapted, rtol=1e-5, atol=1e-6)


def test_load_is_strict_without_lora_and_relaxed_with_it():
    assert _train_module(use_lora=False).state_dict_load_opts.strict is True
    assert _train_module(use_lora=True).state_dict_load_opts.strict is False


class _FakeMetadata:
    def __init__(self, keys):
        self.state_dict_metadata = {k: None for k in keys}


def test_missing_base_weight_is_rejected_even_under_the_relaxed_load():
    """`strict=False` is needed for the adapters; it must not also hide a real gap."""
    tm = _train_module(use_lora=True)
    lora_keys = set(lora_param_names(tm.model))
    all_keys = {f"model.{n}" for n, _ in tm.model.named_parameters()}

    # A checkpoint with everything except the adapters: fine, that is the normal case.
    ok = _FakeMetadata(all_keys - {f"model.{k}" for k in lora_keys})
    tm._check_lora_pruned_keys({}, ok)

    # Same, but also missing a real base weight.
    victim = "model.lm.blocks.0.attention.w_q.weight"
    assert victim in all_keys
    bad = _FakeMetadata(set(ok.state_dict_metadata) - {victim})
    with pytest.raises(RuntimeError, match="missing 1 non-LoRA model key"):
        tm._check_lora_pruned_keys({}, bad)


def test_checkpoint_key_space_is_full_finetune_plus_adapters_only():
    """The invariant the merge script and the relaxed load both rest on.

    Worth pinning because the two branches genuinely differ in *module* names: a LoRA run
    freezes the ViT, so the train module skips vision activation checkpointing and the
    vision submodules are not wrapped. `get_model_state_dict` strips the
    `_checkpoint_wrapped_module` segments, so the checkpoint key spaces still line up — but
    that is a property of the state-dict machinery, not something either side arranges.
    """
    import torch.distributed.checkpoint.state_dict as dist_cp_sd

    full = _train_module(use_lora=False)
    lora = _train_module(use_lora=True)
    full_keys = set(dist_cp_sd.get_model_state_dict(full.model, options=full.state_dict_save_opts))
    lora_keys = set(dist_cp_sd.get_model_state_dict(lora.model, options=lora.state_dict_save_opts))

    assert not any("_checkpoint_wrapped" in k for k in full_keys | lora_keys)
    assert full_keys - lora_keys == set()
    assert lora_keys - full_keys == set(lora_param_names(lora.model))
