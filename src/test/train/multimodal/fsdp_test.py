"""
FSDP tests for MultimodalTransformer + MultimodalTransformerTrainModule.

Uses :func:`~olmo_core.testing.distributed.run_distributed_test` with the
``gloo`` backend to spawn 2 ranks on CPU, so this runs without GPUs and
covers the FSDP wrapping + materialize-then-init flow.
"""

import tempfile

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
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.vision import (
    MultimodalTransformer,
    MultimodalTransformerConfig,
    VisionBackboneConfig,
    VisionBackboneType,
    VisionConnectorConfig,
)
from olmo_core.optim import AdamWConfig
from olmo_core.testing import run_distributed_test
from olmo_core.train.train_module.multimodal import (
    MultimodalTransformerTrainModule,
    MultimodalTransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)

transformers = pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Model and loader builders (same shape as PR 5's tests)
# ---------------------------------------------------------------------------


def _tiny_pixmo_cap_sample(n: int = 4):
    """A tiny in-memory stand-in for a PixMo-Cap shard: deterministic
    ``(prompt, caption, PIL.Image)`` triples, no network or disk required.

    Returns a list (re-iterable) so the data loader can scan it more than once.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        arr = rng.integers(0, 256, size=(56, 56, 3), dtype=np.uint8)
        items.append(("Describe this image in detail.", f"A caption {i}.", Image.fromarray(arr)))
    return items


def _build_meta_model() -> MultimodalTransformer:
    """Build a tiny multimodal model on the meta device so FSDP can wrap it
    before materialization."""
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
    return MultimodalTransformer(cfg, init_device="meta")


# ---------------------------------------------------------------------------
# Single-rank tests (no torch.distributed): verify apply_fsdp / init_weights
# don't break the model when called as part of train module construction.
# ---------------------------------------------------------------------------


def test_init_weights_materializes_from_meta():
    """init_weights moves a meta-device model to a real device and runs
    reset_parameters on every submodule."""
    model = _build_meta_model()
    # Confirm meta first.
    assert any(p.device.type == "meta" for p in model.parameters())

    model.init_weights(device=torch.device("cpu"))

    # All params should now be on CPU and contain real values.
    for p in model.parameters():
        assert p.device.type == "cpu"
        assert torch.isfinite(p).all()


def test_train_module_without_dp_config_works_after_init_weights_change():
    """The single-device path (dp_config=None) still works exactly as PR 5."""
    mm_tok = MultimodalTokenizerConfig.dolma2()
    cfg = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=128,
        optim=AdamWConfig(lr=1e-3),
    )
    # Build the model on CPU (not meta) so the dp_config=None path works.
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
    tm = cfg.build(model, device=torch.device("cpu"))
    assert isinstance(tm, MultimodalTransformerTrainModule)
    assert tm.world_mesh is None


# ---------------------------------------------------------------------------
# Distributed (2-rank gloo) FSDP test
# ---------------------------------------------------------------------------


def _fsdp_wrap_and_init_only():
    """Body for ``run_distributed_test``: just apply_fsdp + init_weights,
    no train module, no data, no forward. Isolates the FSDP/materialization
    flow from the rest of the train stack."""
    import torch.distributed as dist

    from olmo_core.distributed.parallel import build_world_mesh, get_dp_model_mesh

    model = _build_meta_model()
    world_mesh = build_world_mesh(
        dp=TransformerDataParallelConfig(name=DataParallelType.fsdp), device_type="cpu"
    )
    dp_mesh = get_dp_model_mesh(world_mesh)
    model.apply_fsdp(dp_mesh=dp_mesh)
    model.init_weights(
        max_seq_len=128,
        device=torch.device("cpu"),
        world_mesh=world_mesh,
    )

    # Each rank's local params should be finite.
    for p in model.parameters():
        local = p.to_local() if hasattr(p, "to_local") else p
        assert torch.isfinite(local).all(), f"non-finite param on rank {dist.get_rank()}"


def test_fsdp_2rank_wrap_and_init():
    """Minimal: FSDP wrap + init_weights under 2-rank gloo."""
    # Use spawn to avoid inheriting CUDA state from the test process.
    run_distributed_test(
        _fsdp_wrap_and_init_only, world_size=2, backend="gloo", start_method="spawn"
    )


def _fsdp_smoke():
    """Body for ``run_distributed_test``: build model on meta, apply FSDP via
    the train module's dp_config path, then run a single forward + backward."""
    mm_tok = MultimodalTokenizerConfig.dolma2()
    try:
        hf_tok = mm_tok.load_hf_tokenizer()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Could not load dolma2 HF tokenizer: {e}")

    model = _build_meta_model()
    tm_cfg = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=128,
        optim=AdamWConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp),
    )
    tm = tm_cfg.build(model, device=torch.device("cpu"))
    # No Trainer is attached; stub metric hooks.
    tm.record_metric = lambda *a, **k: None
    tm.record_ce_loss = lambda *a, **k: None

    # Build a tiny loader (rank-aware via dp_world_size=2).
    loader_cfg = MultimodalDataLoaderConfig(
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
        global_batch_size=2,
        work_dir=tempfile.mkdtemp(prefix="mm-fsdp-test-"),
    )
    source = _tiny_pixmo_cap_sample(n=4)
    import torch.distributed as dist

    loader = loader_cfg.build(
        source,
        hf_tok,
        dp_world_size=dist.get_world_size(),
        dp_rank=dist.get_rank(),
    )
    loader.reshuffle(epoch=1)
    batch = next(iter(loader))
    loader.reset()

    # Run a training step. Should not raise.
    tm.zero_grads()
    tm.train_batch(batch)
    tm.optim_step()

    # Sanity: at least one connector param should be a DTensor (FSDP-sharded).
    from torch.distributed.tensor import DTensor

    found_dtensor = any(isinstance(p, DTensor) for p in tm.model.connector.parameters())
    assert found_dtensor, "expected FSDP to leave connector params as DTensors"


def test_fsdp_2rank_smoke():
    """Wraps + materializes + runs one training step under 2-rank gloo FSDP."""
    run_distributed_test(_fsdp_smoke, world_size=2, backend="gloo", start_method="spawn")
