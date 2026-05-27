"""
End-to-end test: synthetic dataset → preprocessor → collator → model.forward.

This is the load-bearing test for PR 4 — it verifies that the data pipeline
produces tensors with the exact shapes / dtypes / token-count invariants that
:class:`~olmo_core.nn.vision.MultimodalTransformer.forward` requires.
"""

import pytest
import torch

from olmo_core.data.multimodal import (
    CropMode,
    ImagePreprocessorConfig,
    MultiCropPreprocessorConfig,
    MultimodalCollator,
    MultimodalCollatorConfig,
    MultimodalPreprocessor,
    MultimodalPreprocessorConfig,
    MultimodalTokenizerConfig,
    SyntheticMultimodalDataset,
    SyntheticMultimodalDatasetConfig,
)
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.vision import (
    MultimodalTransformer,
    MultimodalTransformerConfig,
    VisionBackboneConfig,
    VisionBackboneType,
    VisionConnectorConfig,
)

transformers = pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Build a tiny end-to-end stack
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf_tokenizer():
    try:
        return MultimodalTokenizerConfig.dolma2().load_hf_tokenizer()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Could not load dolma2 HF tokenizer: {e}")


def _build_stack(crop_mode: CropMode):
    """Returns (preprocessor, collator, model, multimodal_tokenizer)."""
    mm_tok = MultimodalTokenizerConfig.dolma2()

    multicrop_cfg = MultiCropPreprocessorConfig(
        base_image_input_size=(28, 28),
        crop_mode=crop_mode,
        max_crops=4,
        overlap_margins=(0, 0),
        pool_h=2,
        pool_w=2,
        image_preprocessor=ImagePreprocessorConfig(patch_size=14),
    )
    prep_cfg = MultimodalPreprocessorConfig(
        tokenizer=mm_tok,
        multicrop=multicrop_cfg,
        max_sequence_length=256,
    )
    coll_cfg = MultimodalCollatorConfig(tokenizer=mm_tok)

    # Build the tiny multimodal model whose vocab is sized to fit our extended tokens.
    lm_vocab = mm_tok.padded_vocab_size(128)
    lm_cfg = TransformerConfig.olmo2_1M(vocab_size=lm_vocab)
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
    model_cfg = MultimodalTransformerConfig(
        lm=lm_cfg,
        vision=vis_cfg,
        connector=conn_cfg,
        image_patch_token_id=mm_tok.image_patch_id,
    )
    return prep_cfg, coll_cfg, model_cfg, mm_tok


# ---------------------------------------------------------------------------
# End-to-end shape / dtype check
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.fixture(autouse=True)
    def _setup(self, hf_tokenizer):
        self.tok = hf_tokenizer

    def _run_pipeline(self, crop_mode: CropMode, batch_size: int = 3, n_examples: int = 6):
        prep_cfg, coll_cfg, model_cfg, mm_tok = _build_stack(crop_mode)
        prep = MultimodalPreprocessor(prep_cfg, self.tok)
        coll = MultimodalCollator(coll_cfg)
        model = MultimodalTransformer(model_cfg, init_device="cpu")
        model.eval()

        ds = SyntheticMultimodalDataset(
            SyntheticMultimodalDatasetConfig(n_examples=n_examples, image_size=(56, 56), seed=0)
        )

        # Pull one batch.
        examples = []
        for ex in ds:
            prompt, response, image = ex
            examples.append(prep(prompt, response, image))
            if len(examples) == batch_size:
                break

        batch = coll(examples)
        with torch.inference_mode():
            out = model(
                input_ids=batch["input_ids"],
                images=batch["images"],
                pooled_patches_idx=batch["pooled_patches_idx"],
            )
        return out, batch, model_cfg

    def test_resize_mode_forward(self):
        out, batch, cfg = self._run_pipeline(CropMode.resize)
        B, S = batch["input_ids"].shape
        assert out.shape == (B, S, cfg.lm.vocab_size)
        assert torch.isfinite(out).all()

    def test_overlap_mode_forward(self):
        out, batch, cfg = self._run_pipeline(CropMode.overlap_and_resize)
        B, S = batch["input_ids"].shape
        assert out.shape == (B, S, cfg.lm.vocab_size)
        assert torch.isfinite(out).all()

    def test_with_labels_returns_loss(self):
        prep_cfg, coll_cfg, model_cfg, mm_tok = _build_stack(CropMode.resize)
        prep = MultimodalPreprocessor(prep_cfg, self.tok)
        coll = MultimodalCollator(coll_cfg)
        model = MultimodalTransformer(model_cfg, init_device="cpu")
        model.eval()

        ds = SyntheticMultimodalDataset(SyntheticMultimodalDatasetConfig(n_examples=4, seed=0))
        examples = [prep(p, r, i) for p, r, i in ds]
        batch = coll(examples)

        # Use input_ids as labels (typical autoregressive setup).
        from olmo_core.nn.lm_head import LMOutputWithLoss

        with torch.inference_mode():
            out = model(
                input_ids=batch["input_ids"],
                images=batch["images"],
                pooled_patches_idx=batch["pooled_patches_idx"],
                labels=batch["input_ids"],
            )
        assert isinstance(out, LMOutputWithLoss)
        assert out.loss.shape == ()
        assert torch.isfinite(out.loss)

    def test_text_only_batch_forward(self):
        """When all examples are text-only, the pipeline still produces a working batch."""
        prep_cfg, coll_cfg, model_cfg, mm_tok = _build_stack(CropMode.resize)
        prep = MultimodalPreprocessor(prep_cfg, self.tok)
        coll = MultimodalCollator(coll_cfg)
        model = MultimodalTransformer(model_cfg, init_device="cpu")
        model.eval()

        ds = SyntheticMultimodalDataset(
            SyntheticMultimodalDatasetConfig(n_examples=3, seed=0, text_only_fraction=1.0)
        )
        examples = [prep(p, r, i) for p, r, i in ds]
        batch = coll(examples)
        with torch.inference_mode():
            # No image kwargs → text-only forward.
            out = model(input_ids=batch["input_ids"])
        assert out.shape[0] == 3
        assert torch.isfinite(out).all()

    def test_variable_image_layout(self):
        """In overlap mode with jittered image sizes, the collator must pad
        n_pooled and the model splice must still satisfy its contract."""
        prep_cfg, coll_cfg, model_cfg, mm_tok = _build_stack(CropMode.overlap_and_resize)
        prep = MultimodalPreprocessor(prep_cfg, self.tok)
        coll = MultimodalCollator(coll_cfg)
        model = MultimodalTransformer(model_cfg, init_device="cpu")
        model.eval()

        ds = SyntheticMultimodalDataset(
            SyntheticMultimodalDatasetConfig(
                n_examples=4, image_size=(56, 56), image_size_jitter=28, seed=0
            )
        )
        examples = [prep(p, r, i) for p, r, i in ds]
        batch = coll(examples)
        with torch.inference_mode():
            out = model(
                input_ids=batch["input_ids"],
                images=batch["images"],
                pooled_patches_idx=batch["pooled_patches_idx"],
            )
        assert out.shape[0] == 4
        assert torch.isfinite(out).all()
