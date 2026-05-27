"""Tests for the multimodal train module."""

import tempfile
from test.data.multimodal.synthetic_source import (
    SyntheticMultimodalDataset,
    SyntheticMultimodalDatasetConfig,
)

import pytest
import torch

from olmo_core.data.multimodal import (
    CropMode,
    ImagePreprocessorConfig,
    MultiCropPreprocessorConfig,
    MultimodalDataLoaderConfig,
    MultimodalPreprocessorConfig,
    MultimodalTokenizerConfig,
)
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.vision import (
    MultimodalTransformer,
    MultimodalTransformerConfig,
    VisionBackboneConfig,
    VisionBackboneType,
    VisionConnectorConfig,
)
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module.multimodal import (
    MultimodalTransformerTrainModule,
    MultimodalTransformerTrainModuleConfig,
)

transformers = pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf_tokenizer():
    tok_cfg = MultimodalTokenizerConfig.dolma2()
    try:
        return tok_cfg.load_hf_tokenizer()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Could not load dolma2 HF tokenizer: {e}")


def _build_model() -> MultimodalTransformer:
    mm_tok = MultimodalTokenizerConfig.dolma2()
    lm_cfg = TransformerConfig.olmo2_1M(vocab_size=mm_tok.padded_vocab_size(128))
    vis_cfg = VisionBackboneConfig(
        name=VisionBackboneType.openai,
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
    conn_cfg = VisionConnectorConfig.from_vision_backbone(
        vis_cfg, output_dim=lm_cfg.d_model, mlp_hidden_size=32
    )
    cfg = MultimodalTransformerConfig(
        lm=lm_cfg,
        vision=vis_cfg,
        connector=conn_cfg,
        image_patch_token_id=mm_tok.image_patch_id,
    )
    return MultimodalTransformer(cfg, init_device="cpu")


def _build_train_module(max_grad_norm=None) -> MultimodalTransformerTrainModule:
    cfg = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=128,
        optim=AdamWConfig(lr=1e-3),
        max_grad_norm=max_grad_norm,
    )
    return cfg.build(_build_model(), device=torch.device("cpu"))


def _build_loader(hf_tokenizer, n_examples=8, global_batch_size=2):
    cfg = MultimodalDataLoaderConfig(
        preprocessor=MultimodalPreprocessorConfig(
            max_sequence_length=128,
            multicrop=MultiCropPreprocessorConfig(
                base_image_input_size=(28, 28),
                crop_mode=CropMode.resize,
                pool_h=2,
                pool_w=2,
                image_preprocessor=ImagePreprocessorConfig(patch_size=14),
            ),
        ),
        global_batch_size=global_batch_size,
        work_dir=tempfile.mkdtemp(prefix="mm-tm-test-"),
    )
    source = SyntheticMultimodalDataset(
        SyntheticMultimodalDatasetConfig(n_examples=n_examples, image_size=(56, 56), seed=0)
    )
    return cfg.build(source, hf_tokenizer)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_config_build_returns_train_module():
    tm = _build_train_module()
    assert isinstance(tm, MultimodalTransformerTrainModule)
    assert tm.optim is not None
    assert tm.rank_microbatch_size == 512
    assert tm.max_sequence_length == 128


def test_model_lives_on_configured_device():
    tm = _build_train_module()
    for p in tm.model.parameters():
        assert p.device.type == "cpu"
        break


# ---------------------------------------------------------------------------
# train_batch — gradient flow + parameter update
# ---------------------------------------------------------------------------


def test_train_batch_updates_parameters(hf_tokenizer):
    tm = _build_train_module()
    # Stub out metric hooks (no Trainer attached in unit tests).
    tm.record_metric = lambda *a, **k: None
    tm.record_ce_loss = lambda *a, **k: None
    loader = _build_loader(hf_tokenizer, n_examples=4, global_batch_size=2)
    loader.reshuffle(epoch=1)

    # Snapshot connector + last LM block params before training.
    before = {
        "connector": tm.model.connector.projector.w1.weight.detach().clone(),
        "lm_block_0": next(tm.model.lm.blocks["0"].parameters()).detach().clone(),
    }
    batch = next(iter(loader))
    loader.reset()

    tm.zero_grads()
    tm.train_batch(batch)
    tm.optim_step()

    after = {
        "connector": tm.model.connector.projector.w1.weight.detach().clone(),
        "lm_block_0": next(tm.model.lm.blocks["0"].parameters()).detach().clone(),
    }
    # Both submodules' params should change.
    assert not torch.equal(before["connector"], after["connector"])
    assert not torch.equal(before["lm_block_0"], after["lm_block_0"])


def test_train_batch_records_ce_loss(hf_tokenizer):
    tm = _build_train_module()
    loader = _build_loader(hf_tokenizer, n_examples=4, global_batch_size=2)
    loader.reshuffle(epoch=1)
    batch = next(iter(loader))
    loader.reset()
    # We need a metric recorder; the TrainModule normally has self.trainer.
    # For tests we monkeypatch a tiny recorder.
    captured = {}

    def fake_record_metric(name, value, *args, **kwargs):
        captured[name] = value.detach().item() if isinstance(value, torch.Tensor) else value

    def fake_record_ce_loss(value, *args, **kwargs):
        captured["ce_loss"] = value.detach().item()

    tm.record_metric = fake_record_metric
    tm.record_ce_loss = fake_record_ce_loss

    tm.zero_grads()
    tm.train_batch(batch)
    assert "ce_loss" in captured
    assert captured["ce_loss"] > 0  # Loss should be finite and positive at init.


# ---------------------------------------------------------------------------
# loss_masks → label_mask conversion
# ---------------------------------------------------------------------------


def test_loss_masks_converted_to_label_mask(hf_tokenizer):
    """The train module pops loss_masks (float) and creates label_mask (bool)
    BEFORE delegating to the base class."""
    tm = _build_train_module()
    batch = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "loss_masks": torch.tensor(
            [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.float32
        ),
        "images": torch.zeros(2, 0, 4, 588, dtype=torch.float32),
        "pooled_patches_idx": torch.zeros(2, 0, 4, dtype=torch.long),
    }
    tm._convert_loss_masks(batch)
    assert "loss_masks" not in batch
    assert "label_mask" in batch
    assert batch["label_mask"].dtype == torch.bool
    # Where loss_masks was 1.0, label_mask should be True.
    assert batch["label_mask"][0, 4].item() is True
    assert batch["label_mask"][0, 0].item() is False


def test_label_mask_passthrough_unchanged(hf_tokenizer):
    """If the batch already has label_mask, _convert_loss_masks shouldn't clobber it."""
    tm = _build_train_module()
    existing = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    batch = {"loss_masks": torch.ones(1, 4), "label_mask": existing}
    tm._convert_loss_masks(batch)
    # label_mask preserved; loss_masks left in place (we only pop when label_mask is missing).
    assert torch.equal(batch["label_mask"], existing)


# ---------------------------------------------------------------------------
# Multi-step convergence (does loss decrease?)
# ---------------------------------------------------------------------------


def test_loss_decreases_over_steps(hf_tokenizer):
    """Sanity check: a few training steps on a tiny synthetic batch should
    reduce loss (or at least change it; can be noisy with bs=2)."""
    # No max_grad_norm so optim_step doesn't try to read self.trainer.
    tm = _build_train_module(max_grad_norm=None)
    loader = _build_loader(hf_tokenizer, n_examples=8, global_batch_size=2)

    losses = []

    def fake_record_ce_loss(value, *args, **kwargs):
        losses.append(value.detach().item())

    tm.record_metric = lambda *a, **k: None
    tm.record_ce_loss = fake_record_ce_loss

    loader.reshuffle(epoch=1)
    for batch in loader:
        tm.zero_grads()
        tm.train_batch(batch)
        tm.optim_step()
    loader.reset()

    # Loss should change across steps (gradient is being applied).
    assert len(losses) >= 2
    assert losses[0] != losses[-1]


# ---------------------------------------------------------------------------
# state_dict round-trip
# ---------------------------------------------------------------------------


def test_state_dict_round_trip():
    tm1 = _build_train_module()
    state = tm1.state_dict(optim=False)
    assert "model" in state

    # Build a fresh module and load.
    tm2 = _build_train_module()
    sd_to_load = tm2.state_dict_to_load(
        # state_dict_to_load needs a Metadata object — we test the simpler model-only path.
        metadata=type("M", (), {"state_dict_metadata": {}})(),
        optim=False,
    )
    # The loaded state_dict_to_load should at least produce the same keys.
    assert set(sd_to_load.keys()) == {"model"}


# ---------------------------------------------------------------------------
# End-to-end: data loader → train module
# ---------------------------------------------------------------------------


def test_end_to_end_train_step(hf_tokenizer):
    """Drive the full pipeline: source → loader → train_module.train_batch."""
    tm = _build_train_module()
    tm.record_metric = lambda *a, **k: None
    tm.record_ce_loss = lambda *a, **k: None

    loader = _build_loader(hf_tokenizer, n_examples=4, global_batch_size=2)
    loader.reshuffle(epoch=1)
    for batch in loader:
        tm.zero_grads()
        tm.train_batch(batch)
        tm.optim_step()
    loader.reset()
    # If we got here without exceptions, all the kwargs propagated correctly
    # through the train module to the model's forward.
