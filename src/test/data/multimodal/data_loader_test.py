"""Tests for the multimodal data loader."""

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


def _loader_cfg(global_batch_size: int = 4, num_workers: int = 0) -> MultimodalDataLoaderConfig:
    return MultimodalDataLoaderConfig(
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
        num_workers=num_workers,
        work_dir=tempfile.mkdtemp(prefix="mm-data-test-"),
    )


def _source(n_examples: int = 32, seed: int = 0) -> SyntheticMultimodalDataset:
    return SyntheticMultimodalDataset(
        SyntheticMultimodalDatasetConfig(n_examples=n_examples, image_size=(56, 56), seed=seed)
    )


# ---------------------------------------------------------------------------
# Basic batch shape / dtype
# ---------------------------------------------------------------------------


def test_yields_batches_of_expected_shape(hf_tokenizer):
    cfg = _loader_cfg(global_batch_size=4)
    loader = cfg.build(_source(n_examples=32), hf_tokenizer)
    loader.reshuffle(epoch=1)

    batches = list(loader)
    loader.reset()
    assert len(batches) == 32 // 4
    b = batches[0]
    assert b["input_ids"].shape[0] == 4  # rank_batch_size = 4 with dp_world_size=1
    assert b["images"].shape[0] == 4
    assert b["pooled_patches_idx"].shape[0] == 4


def test_batch_dtypes_match_model_contract(hf_tokenizer):
    cfg = _loader_cfg(global_batch_size=2)
    loader = cfg.build(_source(n_examples=8), hf_tokenizer)
    loader.reshuffle(epoch=1)
    b = next(iter(loader))
    loader.reset()
    assert b["input_ids"].dtype == torch.long
    assert b["loss_masks"].dtype == torch.float32
    assert b["images"].dtype == torch.float32
    assert b["pooled_patches_idx"].dtype == torch.long


def test_total_batches_matches_iter_count(hf_tokenizer):
    cfg = _loader_cfg(global_batch_size=4)
    loader = cfg.build(_source(n_examples=20), hf_tokenizer)
    loader.reshuffle(epoch=1)
    expected = loader.total_batches
    assert expected == 5
    actual = sum(1 for _ in loader)
    assert actual == expected


# ---------------------------------------------------------------------------
# Rank sharding
# ---------------------------------------------------------------------------


def test_rank_sharding_yields_disjoint_examples(hf_tokenizer):
    """Each rank sees a different stride of the source — no overlap."""
    cfg = _loader_cfg(global_batch_size=4)
    rank0 = cfg.build(_source(n_examples=16), hf_tokenizer, dp_world_size=2, dp_rank=0)
    rank1 = cfg.build(_source(n_examples=16), hf_tokenizer, dp_world_size=2, dp_rank=1)
    rank0.reshuffle(epoch=1)
    rank1.reshuffle(epoch=1)

    # rank_batch_size = global_batch_size / dp_world_size = 2.
    assert rank0.rank_batch_size == 2
    assert rank1.rank_batch_size == 2

    r0 = next(iter(rank0))["input_ids"]
    r1 = next(iter(rank1))["input_ids"]
    rank0.reset()
    rank1.reset()
    # The two ranks should see different examples → token sequences differ.
    assert not torch.equal(r0, r1)


# ---------------------------------------------------------------------------
# Reshuffle / epochs
# ---------------------------------------------------------------------------


def test_reshuffle_changes_examples_across_epochs(hf_tokenizer):
    cfg = _loader_cfg(global_batch_size=2)
    loader = cfg.build(_source(n_examples=8), hf_tokenizer)
    loader.reshuffle(epoch=1)
    epoch1 = next(iter(loader))["input_ids"]
    loader.reset()

    loader.reshuffle(epoch=2)
    epoch2 = next(iter(loader))["input_ids"]
    loader.reset()
    assert not torch.equal(epoch1, epoch2)


def test_reshuffle_same_epoch_same_order(hf_tokenizer):
    """Calling reshuffle with the same epoch twice produces the same iteration."""
    cfg = _loader_cfg(global_batch_size=2)
    loader = cfg.build(_source(n_examples=8), hf_tokenizer)
    loader.reshuffle(epoch=3)
    a = next(iter(loader))["input_ids"]
    loader.reset()
    loader.reshuffle(epoch=3)
    b = next(iter(loader))["input_ids"]
    loader.reset()
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# state_dict round-trip
# ---------------------------------------------------------------------------


def test_state_dict_round_trip_resumes(hf_tokenizer):
    cfg = _loader_cfg(global_batch_size=2)
    loader = cfg.build(_source(n_examples=8), hf_tokenizer)
    loader.reshuffle(epoch=1)

    # Consume two batches.
    it = iter(loader)
    next(it)
    next(it)
    state = loader.state_dict()
    assert state["batches_processed"] == 2

    # Build a fresh loader and load state.
    loader2 = cfg.build(_source(n_examples=8), hf_tokenizer)
    loader2.load_state_dict(state)
    assert loader2.batches_processed == 2
    assert loader2.epoch == state["epoch"]


# ---------------------------------------------------------------------------
# Mock batch
# ---------------------------------------------------------------------------


def test_mock_batch_shape_matches_model_contract(hf_tokenizer):
    cfg = _loader_cfg(global_batch_size=2)
    loader = cfg.build(_source(n_examples=8), hf_tokenizer)
    mock = loader.get_mock_batch()
    # Same keys as a real batch.
    real_keys = {"input_ids", "loss_masks", "images", "pooled_patches_idx"}
    assert set(mock.keys()) == real_keys
    assert mock["input_ids"].shape[0] == loader.rank_batch_size
    # Total <im_patch> count must equal B * n_pooled (model contract).
    patch_id = cfg.preprocessor.tokenizer.image_patch_id
    n_image_patches = (mock["input_ids"] == patch_id).sum().item()
    B, n_pooled, _ = mock["pooled_patches_idx"].shape
    assert n_image_patches == B * n_pooled


# ---------------------------------------------------------------------------
# Multi-worker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [0, 2])
def test_works_with_workers(hf_tokenizer, num_workers):
    cfg = _loader_cfg(global_batch_size=4, num_workers=num_workers)
    loader = cfg.build(_source(n_examples=16), hf_tokenizer)
    loader.reshuffle(epoch=1)
    n = 0
    for batch in loader:
        n += 1
        assert batch["input_ids"].shape[0] == 4
    loader.reset()
    assert n == 4  # 16 examples / batch 4


# ---------------------------------------------------------------------------
# End-to-end with the model
# ---------------------------------------------------------------------------


def test_end_to_end_with_model(hf_tokenizer):
    """Drive the loader through MultimodalTransformer.forward."""
    from olmo_core.nn.transformer.config import TransformerConfig
    from olmo_core.nn.vision import (
        MultimodalTransformer,
        MultimodalTransformerConfig,
        VisionBackboneConfig,
        VisionBackboneType,
        VisionConnectorConfig,
    )

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
    model = MultimodalTransformer(
        MultimodalTransformerConfig(
            lm=lm_cfg,
            vision=vis_cfg,
            connector=conn_cfg,
            image_patch_token_id=mm_tok.image_patch_id,
        ),
        init_device="cpu",
    )
    model.eval()

    loader = _loader_cfg(global_batch_size=2).build(_source(n_examples=4), hf_tokenizer)
    loader.reshuffle(epoch=1)
    for batch in loader:
        with torch.inference_mode():
            out = model(
                input_ids=batch["input_ids"],
                images=batch["images"],
                pooled_patches_idx=batch["pooled_patches_idx"],
            )
        assert torch.isfinite(out).all()
    loader.reset()
