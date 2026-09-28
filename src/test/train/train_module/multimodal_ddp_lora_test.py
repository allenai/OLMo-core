"""DDP (composable ``replicate``) for LoRA stage-2 training.

The multimodal DDP branch had zero test coverage before this file, and the failure modes
it risks are all distributed-only: a rank whose batch is all-text must still produce a
gradient for every reducer-registered param or DDP errors on the *next* step; gradient
averaging must compose with the global-weight loss divisor; and grad sync must be gated to
the final micro-batch through the composable mixin's ``set_requires_gradient_sync`` (the
legacy ``no_sync()`` branch never matches a ``replicate()``d model — its class is swapped
to ``torch.distributed._composable.replicate.DDP``, a different class from
``nn.parallel.DistributedDataParallel``).

All tests run on 2 gloo CPU processes via ``run_distributed_test``.
"""

from typing import Dict

import torch
import torch.distributed as dist
from torch.distributed._composable.replicate import DDP as ComposableDDP

from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lora import LoRAConfig, lora_param_names
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.vision import (
    MultimodalLMConfig,
    VisionConnectorConfig,
    VisionEncoderConfig,
    VisionEncoderType,
)
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.testing.distributed import run_distributed_test
from olmo_core.train.train_module import MultimodalTransformerTrainModuleConfig
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)

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
    # Seeded identically on every rank. Composable replicate would broadcast rank 0's
    # weights on the first forward anyway, but deterministic init keeps the single-process
    # parity test's reference model equal to the distributed ranks' models by construction.
    torch.manual_seed(0)
    model = _model_config().build(init_device="cpu")
    model.lm.init_weights(max_seq_len=SEQ_LEN, device=torch.device("cpu"))
    model.vision.reset_parameters()
    model.connector.reset_parameters()
    return model


def _train_module(dp: bool):
    """The LoRA branch of ``Molmo2-Stage2.build_config``, optionally under DDP."""
    config = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=SEQ_LEN,
        max_sequence_length=SEQ_LEN,
        optim=AdamWConfig(
            lr=1e-5,
            group_overrides=[
                OptimGroupOverride(
                    params=["vision_backbone.connector.*"],
                    opts=dict(lr=5e-6, weight_decay=0.0, scheduler_name="connector"),
                ),
                OptimGroupOverride(
                    params=["lm.*.lora_A", "lm.*.lora_B"],
                    opts=dict(lr=2e-4, weight_decay=0.0, scheduler_name="lora"),
                ),
            ],
        ),
        freeze_params=["lm.*", "vision_backbone.vision.*"],
        lora=LoRAConfig(rank=RANK, alpha=2.0 * RANK, target_modules=list(LORA_TARGETS)),
        dp_config=(TransformerDataParallelConfig(name=DataParallelType.ddp) if dp else None),
        z_loss_multiplier=1e-4,
        compile_model=False,
        response_logits_only=True,
    )
    train_module = config.build(_build_model(), device=torch.device("cpu"))
    train_module._trainer = _StubTrainer()  # type: ignore[assignment]
    return train_module


def _batch(seed: int, with_image: bool) -> Dict[str, torch.Tensor]:
    """One sequence. ``with_image=False`` mirrors the collator's all-text contract: a
    single dummy zero crop whose pooled indices are all -1, splicing nothing."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, BASE_VOCAB, (1, SEQ_LEN), generator=generator)
    if with_image:
        input_ids[0, 2 : 2 + N_POOLED] = IMAGE_PATCH_ID
        pooled = torch.zeros(1, N_POOLED, 4, dtype=torch.long)
        images = torch.randn(1, 1, 16, 3 * 14 * 14, generator=generator)
    else:
        input_ids[input_ids == IMAGE_PATCH_ID] = 0
        pooled = torch.full((1, N_POOLED, 4), -1, dtype=torch.long)
        images = torch.zeros(1, 1, 16, 3 * 14 * 14)
    loss_masks = torch.zeros(1, SEQ_LEN)
    loss_masks[:, SEQ_LEN // 2 :] = 1.0
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "loss_masks": loss_masks,
        "images": images,
        "pooled_patches_idx": pooled,
    }


def _trainable_grads(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }


# -- workers (run on 2 gloo processes) -----------------------------------------------


def _asymmetric_batches_worker():
    """Rank 0 trains on an image batch, rank 1 on all-text — two full steps.

    Two steps on purpose: with ``find_unused_parameters=False`` (the default the DDP
    branch runs with), a param that missed its gradient does not fail in that backward —
    DDP raises "Expected to have finished reduction in the prior iteration" on the *next*
    forward. One step would pass vacuously.
    """
    tm = _train_module(dp=True)
    assert isinstance(tm.model, ComposableDDP)

    for step in range(2):
        batch = _batch(seed=10 * step + dist.get_rank(), with_image=dist.get_rank() == 0)
        tm.train_batch(batch)

        # After the (single-microbatch) backward, grads are all-reduce-averaged: every
        # rank must hold identical trainable grads even though the batches differ.
        for name, grad in sorted(_trainable_grads(tm.model).items()):
            gathered = [torch.empty_like(grad) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, grad)
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0, msg=name)

        tm.optim_step()
        tm.zero_grads()


def test_asymmetric_batches_two_steps_no_unused_param_error():
    run_distributed_test(_asymmetric_batches_worker, world_size=2, backend="gloo")


def _grad_parity_worker(reference_path: str):
    """2-rank DDP over [b0, b1] must equal one process accumulating b0+b1.

    Pins the semantics the ``train_batch`` divisor comment relies on: each rank divides by
    ``global_weight / world_size`` and DDP *averages*, so the effective normalization is
    the global weight — identical to one process consuming both sequences as one batch
    split into micro-batches. The reference gradients are computed in the *parent* pytest
    process and passed in, because ``build_world_mesh`` is once-per-process, so a second
    (non-DDP) train module cannot be built inside a distributed worker.
    """
    reference_grads = torch.load(reference_path, weights_only=True)
    tm = _train_module(dp=True)
    tm.train_batch(_batch(seed=dist.get_rank(), with_image=True))
    ddp_grads = _trainable_grads(tm.model)

    assert set(ddp_grads) == set(reference_grads)
    for name in sorted(ddp_grads):
        torch.testing.assert_close(
            ddp_grads[name], reference_grads[name], rtol=1e-5, atol=1e-6, msg=name
        )


def _compute_reference_grads(path: str) -> None:
    """Non-distributed reference: both sequences as one 2-row batch, split by train_batch
    into 2 micro-batches of 1 (rank_microbatch_size == one sequence)."""
    reference = _train_module(dp=False)
    b0, b1 = _batch(seed=0, with_image=True), _batch(seed=1, with_image=True)
    combined = {k: torch.cat([b0[k], b1[k]], dim=0) for k in b0}
    reference.train_batch(combined)
    grads = _trainable_grads(reference.model)
    assert grads, "reference produced no trainable grads"
    torch.save(grads, path)


def test_ddp_grad_parity_with_single_process_accumulation(tmp_path):
    # The reference cannot be built inside a worker (`build_world_mesh` is once-per-
    # process, and the train module calls it whenever dist is initialized — even with
    # dp_config=None). It also cannot run in the parent before the fork: gloo workers
    # fork, and forking after the parent has executed a model backward (OpenMP/threadpool
    # state) deadlocks. So the reference runs in its own spawned process and hands its
    # grads over as a file.
    import torch.multiprocessing as mp

    path = str(tmp_path / "reference_grads.pt")
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_compute_reference_grads, args=(path,))
    proc.start()
    proc.join(timeout=300)
    assert proc.exitcode == 0, f"reference process exited {proc.exitcode}"

    run_distributed_test(_grad_parity_worker, world_size=2, backend="gloo", func_args=(path,))


def _microbatch_sync_gating_worker():
    """`_train_microbatch_context` must gate grad sync through the composable mixin.

    Before the fix this context's DDP branch checked ``isinstance(self.model,
    nn.parallel.DistributedDataParallel)`` — always False for a ``replicate()``d model —
    so every micro-batch all-reduced the full trainable gradient set.
    """
    from torch.distributed._composable.replicate import replicate as replicate_api

    tm = _train_module(dp=True)
    state = replicate_api.state(tm.model)

    with tm._train_microbatch_context(0, 2):
        assert state._no_sync is True, "non-final micro-batch must skip grad sync"
    with tm._train_microbatch_context(1, 2):
        assert state._no_sync is False, "final micro-batch must sync grads"
    # Single-microbatch steps always sync.
    with tm._train_microbatch_context(0, 1):
        assert state._no_sync is False

    # And the gated path still converges on the same synced grads: a 2-row batch splits
    # into 2 micro-batches; after train_batch the grads must match across ranks.
    b = {
        k: torch.cat([_batch(seed=7, with_image=True)[k], _batch(seed=8, with_image=True)[k]], 0)
        for k in _batch(seed=7, with_image=True)
    }
    tm.train_batch(b)
    for name, grad in sorted(_trainable_grads(tm.model).items()):
        gathered = [torch.empty_like(grad) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, grad)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0, msg=name)


def test_microbatch_sync_is_gated_to_the_final_microbatch():
    run_distributed_test(_microbatch_sync_gating_worker, world_size=2, backend="gloo")


def _checkpoint_key_space_worker():
    """The class swap and forward hooks must not leak into checkpoint keys."""
    import torch.distributed.checkpoint.state_dict as dist_cp_sd

    tm = _train_module(dp=True)
    keys = set(dist_cp_sd.get_model_state_dict(tm.model, options=tm.state_dict_save_opts))

    expected = {
        n.replace("._checkpoint_wrapped_module", "") for n, _ in tm.model.named_parameters()
    }
    assert keys == expected
    assert not any("_checkpoint_wrapped_module" in k for k in keys)
    assert not any(k.startswith("module.") for k in keys)
    assert set(lora_param_names(tm.model)) <= keys


def test_checkpoint_key_space_unchanged_under_ddp():
    run_distributed_test(_checkpoint_key_space_worker, world_size=2, backend="gloo")
