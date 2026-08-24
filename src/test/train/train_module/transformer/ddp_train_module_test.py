"""Tests for :class:`OLMoDDPTrainModule` config and construction."""

from typing import Optional

import pytest
import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.train.train_module import OLMoDDPTrainModule, OLMoDDPTrainModuleConfig
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerPipelineParallelConfig,
)


def test_moe_v2_train_module_config_roundtrips():
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
    )
    restored = OLMoDDPTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.optim.lr == 1e-3


def test_moe_v2_train_module_config_roundtrips_with_parallelism():
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, reduce_grads_in_fp32=False
        ),
        pp_config=TransformerPipelineParallelConfig(degree=2),
    )
    restored = OLMoDDPTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.dp_config is not None and restored.dp_config.reduce_grads_in_fp32 is False
    assert restored.pp_config is not None and restored.pp_config.degree == 2


def _tiny_model_config(
    *,
    d_model: int = 64,
    n_layers: int = 2,
    dtype: DType = DType.float32,
    router_bias_gamma: Optional[float] = None,
    global_load_balancing: bool = False,
) -> OLMoDDPModelConfig:
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    return OLMoDDPModelConfig(
        init_seed=0,
        d_model=d_model,
        recompute_each_block=False,
        vocab_size=128,
        n_layers=n_layers,
        name=TransformerType.moe_fused_v2,
        block=OLMoDDPTransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            attention=AttentionConfig(
                name=AttentionType.default, n_heads=4, bias=False, use_flash=False, dtype=dtype
            ),
            routed_experts=RoutedExpertsConfig(
                d_model=d_model, hidden_size=128, num_experts=4, bias=False, dtype=dtype
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=d_model,
                num_experts=4,
                top_k=2,
                dtype=dtype,
                bias_gamma=router_bias_gamma,
                global_load_balancing=global_load_balancing,
            ),
            shared_experts=None,
            layer_norm=layer_norm,
        ),
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
    )


def _run_construct_no_ep():
    model = _tiny_model_config().build(init_device="cpu")
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
    )
    # eval_only=True skips the optimizer build (its fp32-master-param setup is exercised on GPU);
    # this covers the world-mesh build + data-parallel wrapping with no expert parallelism.
    train_module = config.build(model, device=torch.device("cpu"), eval_only=True)

    assert len(train_module.model_parts) == 1  # no pipeline parallelism
    assert train_module.dp_world_size == 2
    assert train_module.world_mesh["dense"] is not None
    assert train_module.moe_mesh is None  # no expert parallelism


def test_moe_v2_train_module_construction_no_ep():
    run_distributed_test(
        _run_construct_no_ep,
        world_size=2,
        backend="gloo",
        start_method="spawn",
    )


def _run_construct_ep():
    # bf16 params → the fused optimizer maintains fp32 master params (its realistic config); a pure
    # fp32 model instead takes the optimizer's "expect fp32 param" branch.
    model = _tiny_model_config(dtype=DType.bfloat16).build(init_device="cuda")
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
        ep_config=TransformerExpertParallelConfig(degree=2),
    )
    # Full build (eval_only=False): wires expert parallelism through the train module (moe mesh +
    # apply_ep sharding the experts across the two ranks + DP wrapping) and builds the optimizer.
    train_module = config.build(model, device=torch.device("cuda"), eval_only=False)

    assert len(train_module.model_parts) == 1  # no pipeline parallelism
    assert train_module.moe_mesh is not None
    assert train_module.ep_mp_group is not None
    assert train_module.optim is not None
    assert train_module.num_flops_per_token(seq_len=512) > 0


@requires_multi_gpu
def test_moe_v2_train_module_construction_ep():
    run_distributed_test(
        _run_construct_ep,
        world_size=2,
        backend="nccl",
        start_method="spawn",
    )


def test_moe_v2_train_module_config_reset_optimizer_states_roundtrips():
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        reset_optimizer_states_on_resume=True,
    )
    restored = OLMoDDPTrainModuleConfig.from_dict(config.as_dict())
    assert restored == config
    assert restored.reset_optimizer_states_on_resume is True
    # The resume flag is distinct from the generic on-load flag, which stays at its default.
    assert restored.reset_optimizer_states_on_load is False


def _run_rejects_per_microbatch_allreduce():
    model = _tiny_model_config().build(init_device="cpu")
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp, only_allreduce_last_microbatch=False
        ),
    )
    # MultiGroupDistributedDataParallel reduces each bucket once per accumulation window, so
    # per-micro-batch all-reduce is unsupported and must be rejected up front.
    with pytest.raises(OLMoConfigurationError, match="only_allreduce_last_microbatch"):
        config.build(model, device=torch.device("cpu"), eval_only=True)


def test_moe_v2_train_module_rejects_per_microbatch_allreduce():
    run_distributed_test(
        _run_rejects_per_microbatch_allreduce,
        world_size=2,
        backend="gloo",
        start_method="spawn",
    )


_MOMENT_SUFFIXES = (".exp_avg", ".exp_avg_sq")


def _build_ddp_train_module_for_checkpoint(*, router_bias_gamma: Optional[float] = None):
    model = _tiny_model_config(dtype=DType.bfloat16, router_bias_gamma=router_bias_gamma).build(
        init_device="cuda"
    )
    config = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=512,
        max_sequence_length=512,
        optim=OLMoDDPOptimizerConfig(lr=1e-3),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
    )
    return config.build(model, device=torch.device("cuda"), eval_only=False)


def _run_resume_resets_optimizer_moments(save_dir):
    # Save a checkpoint carrying non-zero optimizer moments, then verify that the resume flag
    # (threaded through as reset_optimizer_states_on_load) actually controls whether those moments
    # are restored or discarded on load.
    tm = _build_ddp_train_module_for_checkpoint()
    assert tm.optim is not None
    for key, state in tm.optim.states.items():
        if key.endswith(_MOMENT_SUFFIXES):
            state.to_local().fill_(0.5)
    tm.save_state_dict_direct(save_dir)

    # Reset on load: only the main params are restored, so freshly zero-initialized moments stay zero.
    tm_reset = _build_ddp_train_module_for_checkpoint()
    assert tm_reset.optim is not None
    tm_reset.load_state_dict_direct(save_dir, reset_optimizer_states_on_load=True)
    for key, state in tm_reset.optim.states.items():
        if key.endswith(_MOMENT_SUFFIXES):
            assert torch.count_nonzero(state.to_local()) == 0, key

    # No reset: the saved (non-zero) moments are restored.
    tm_restore = _build_ddp_train_module_for_checkpoint()
    assert tm_restore.optim is not None
    tm_restore.load_state_dict_direct(save_dir, reset_optimizer_states_on_load=False)
    restored_any_moment = any(
        key.endswith(_MOMENT_SUFFIXES) and torch.count_nonzero(state.to_local()) > 0
        for key, state in tm_restore.optim.states.items()
    )
    assert restored_any_moment


@requires_multi_gpu
def test_moe_v2_train_module_resume_resets_optimizer_moments(tmp_path):
    run_distributed_test(
        _run_resume_resets_optimizer_moments,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(str(tmp_path / "checkpoint"),),
    )


def _score_bias_buffers(train_module):
    return {
        name: buf
        for model_part in train_module.model_parts
        for name, buf in model_part.named_buffers()
        if name.endswith("score_bias") and buf is not None
    }


def _run_direct_checkpoint_restores_buffers(save_dir):
    # Persistent buffers (the router's aux-loss-free score_bias) are model state updated outside
    # the optimizer; the direct checkpoint must round-trip them.
    tm = _build_ddp_train_module_for_checkpoint(router_bias_gamma=1e-3)
    mutated = _score_bias_buffers(tm)
    assert mutated, "expected at least one score_bias buffer with bias_gamma set"
    saved = {}
    for name, buf in mutated.items():
        buf.copy_(torch.arange(buf.numel(), device=buf.device, dtype=buf.dtype) + 1.0)
        saved[name] = buf.detach().clone()
    tm.save_state_dict_direct(save_dir)

    tm_restored = _build_ddp_train_module_for_checkpoint(router_bias_gamma=1e-3)
    restored = _score_bias_buffers(tm_restored)
    # Freshly built buffers are zero-initialized, so they must differ before the load.
    for name, buf in restored.items():
        assert torch.count_nonzero(buf) == 0, name
    tm_restored.load_state_dict_direct(save_dir, reset_optimizer_states_on_load=False)
    for name, expected in saved.items():
        torch.testing.assert_close(restored[name], expected)


@requires_multi_gpu
def test_moe_v2_train_module_direct_checkpoint_restores_buffers(tmp_path):
    run_distributed_test(
        _run_direct_checkpoint_restores_buffers,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(str(tmp_path / "checkpoint"),),
    )


@pytest.mark.parametrize(
    "value, multiple, expected",
    [
        ([4, 2, 3], 2, [4, 2, 3, 3]),
        ([4, 2], 2, [4, 2]),
        ((4, 2, 3), 2, (4, 2, 3, 3)),
    ],
)
def test_pad_pp_batch_dim_pads_batch_leading_sequences(value, multiple, expected):
    # Per-instance metadata such as 'max_doc_lens' is a Python list, so it has to be padded
    # alongside the tensors it accompanies or it ends up shorter than the padded batch.
    assert OLMoDDPTrainModule._pad_pp_batch_dim(value, multiple) == expected


def test_pad_pp_batch_dim_pads_tensors_by_repeating_the_last_instance():
    value = torch.tensor([[1, 2], [3, 4], [5, 6]])
    padded = OLMoDDPTrainModule._pad_pp_batch_dim(value, 2)
    assert torch.equal(padded, torch.tensor([[1, 2], [3, 4], [5, 6], [5, 6]]))


def test_split_pp_dry_run_model_kwargs_slices_batch_leading_values():
    # The independent PP dry run splits model kwargs itself, so per-instance metadata has to be
    # sliced there too rather than handed whole to every microbatch.
    kwargs = {
        "segment_ids": torch.arange(8).reshape(4, 2),
        "max_doc_lens": [4, 2, 3, 4],
        "cp_original_seq_len": 2,
    }
    split = OLMoDDPTrainModule._split_pp_dry_run_model_kwargs(
        kwargs,
        original_batch_size=4,
        micro_batch_size=2,
        num_microbatches=2,
    )

    assert torch.equal(split[0]["segment_ids"], kwargs["segment_ids"][0:2])
    assert torch.equal(split[1]["segment_ids"], kwargs["segment_ids"][2:4])
    assert split[0]["max_doc_lens"] == [4, 2]
    assert split[1]["max_doc_lens"] == [3, 4]
    # Scalars are broadcast rather than sliced.
    assert split[0]["cp_original_seq_len"] == split[1]["cp_original_seq_len"] == 2


def _run_global_lb_group_is_stage_local():
    train_module = OLMoDDPTrainModule.__new__(OLMoDDPTrainModule)
    train_module.world_mesh = {}
    train_module._build_world_mesh(
        dp=TransformerDataParallelConfig(name=DataParallelType.ddp),
        pp=TransformerPipelineParallelConfig(degree=2),
        device_type="cpu",
    )
    pp_rank = train_module.dense_mesh["pp"].get_local_rank()

    model = _tiny_model_config(global_load_balancing=True).build(init_device="cpu")
    # Call apply_dp directly rather than building the whole train module: the PP path installs
    # CUDA events, which this CPU-only test cannot do. This is the call that hands the routers
    # their load-balancing group, which is what the test is about.
    model.apply_dp(dp_mesh=train_module.dense_mesh["dp"], ep_mesh=None)

    expected_size = dist.get_world_size() // 2
    for block in model.blocks.values():
        lb_group = block.routed_experts_router.lb_process_group
        assert lb_group is not None, "global load balancing needs a group after apply_dp"
        assert dist.get_world_size(lb_group) == expected_size

        # Every member has to be a data-parallel replica of this rank's own pipeline stage.
        # A group spanning stages would average expert counts over unrelated layers, which
        # nothing else would catch: the reduction still succeeds, it just balances the wrong thing.
        gathered = [torch.zeros(1, dtype=torch.long) for _ in range(expected_size)]
        dist.all_gather(gathered, torch.tensor([pp_rank], dtype=torch.long), group=lb_group)
        assert {int(t.item()) for t in gathered} == {
            pp_rank
        }, f"rank {dist.get_rank()} shares a load-balancing group with another pipeline stage"


def test_global_lb_group_is_stage_local():
    # 2 pipeline stages x 2 data-parallel replicas.
    run_distributed_test(
        _run_global_lb_group_is_stage_local,
        world_size=4,
        backend="gloo",
        start_method="spawn",
    )
