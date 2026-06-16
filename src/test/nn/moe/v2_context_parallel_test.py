import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import pytest
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import DType
from olmo_core.distributed.parallel.context_parallel import ContextParallelConfig
from olmo_core.distributed.parallel.data_parallel import DataParallelConfig, DataParallelType
from olmo_core.distributed.parallel.expert_parallel import ExpertParallelConfig
from olmo_core.distributed.parallel.pipeline_parallel import (
    PipelineParallelConfig,
    PipelineP2PBackend,
    PipelineScheduleType,
    PipelineSplitStyle,
)
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.testing import requires_grouped_gemm, requires_multi_gpu, run_distributed_test
from olmo_core.optim.moe_optimizer import MoEFusedV2OptimizerConfig
from olmo_core.train.train_module.transformer.pipeline.pipeline_schedule import (
    CustomScheduleInterleaved1F1B,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerPipelineParallelConfig,
)
from olmo_core.train.train_module.transformer.moe_train_module import MoEV2TransformerTrainModule

from .v2_block_no_sync_test import (
    _init_block_params,
    _install_deterministic_topk_router,
    _install_forced_router,
)


def _build_block(
    init_device: str = "cpu",
    *,
    ep_no_sync: bool = False,
    ep_no_sync_use_rowwise_all_to_all: bool = False,
) -> OLMoDDPTransformerBlock:
    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )
    return OLMoDDPTransformerBlock(
        d_model=128,
        block_idx=0,
        n_layers=1,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=2,
            n_kv_heads=2,
            bias=False,
            use_flash=False,
            dtype=DType.float32,
        ),
        attention_norm=layer_norm,
        routed_experts_router=MoERouterConfigV2(
            d_model=128,
            num_experts=4,
            top_k=1,
            gating_function=MoERouterGatingFunction.softmax,
            uniform_expert_assignment=True,
            lb_loss_weight=0.01,
            z_loss_weight=1e-3,
            dtype=DType.float32,
        ),
        shared_experts_router=None,
        shared_experts=None,
        routed_experts=RoutedExpertsConfig(
            d_model=128,
            hidden_size=256,
            num_experts=4,
            bias=False,
            dtype=DType.float32,
        ),
        feed_forward_norm=layer_norm,
        ep_no_sync=ep_no_sync,
        ep_no_sync_use_rowwise_all_to_all=ep_no_sync_use_rowwise_all_to_all,
        ep_no_sync_rowwise_nblocks=128,
        init_device=init_device,
    )


def _build_model(
    init_device: str = "cuda",
    n_layers: int = 1,
    *,
    ep_no_sync: bool = False,
    ep_no_sync_use_rowwise_all_to_all: bool = False,
):
    from olmo_core.nn.lm_head import LMHeadConfig
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
    from olmo_core.nn.transformer import (
        MoEFusedV2TransformerConfig,
        TransformerBlockType,
        TransformerType,
    )

    layer_norm = LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.float32,
    )
    config = MoEFusedV2TransformerConfig(
        name=TransformerType.moe_fused_v2,
        d_model=128,
        vocab_size=64,
        n_layers=n_layers,
        recompute_each_block=False,
        recompute_all_blocks_by_chunk=False,
        lm_head=LMHeadConfig(bias=False, dtype=DType.float32),
        block=OLMoDDPTransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            sequence_mixer=AttentionConfig(
                name=AttentionType.default,
                n_heads=2,
                n_kv_heads=2,
                bias=False,
                rope=RoPEConfig(name=RoPEType.default, theta=10_000, full_precision=True),
                use_flash=False,
                dtype=DType.float32,
            ),
            attention_norm=layer_norm,
            routed_experts_router=MoERouterConfigV2(
                d_model=128,
                num_experts=4,
                top_k=1,
                gating_function=MoERouterGatingFunction.softmax,
                uniform_expert_assignment=False,
                lb_loss_weight=None,
                z_loss_weight=None,
                dtype=DType.float32,
            ),
            shared_experts_router=None,
            shared_experts=None,
            routed_experts=RoutedExpertsConfig(
                d_model=128,
                hidden_size=256,
                num_experts=4,
                bias=False,
                dtype=DType.float32,
            ),
            feed_forward_norm=layer_norm,
            ep_no_sync=ep_no_sync,
            ep_no_sync_use_rowwise_all_to_all=ep_no_sync_use_rowwise_all_to_all,
            ep_no_sync_rowwise_nblocks=128,
        ),
    )
    return config.build(init_device=init_device)


def _init_module_params(module: torch.nn.Module) -> None:
    torch.manual_seed(1234)
    with torch.no_grad():
        for param in module.parameters():
            if param.is_floating_point():
                param.normal_(mean=0.0, std=0.02)


def _new_train_module_for_mesh() -> MoEV2TransformerTrainModule:
    module = MoEV2TransformerTrainModule.__new__(MoEV2TransformerTrainModule)
    module.world_mesh = {}
    module.pp_group = None
    module.dp_group = None
    module.tp_group = None
    module.cp_group = None
    module.ep_dp_group = None
    module.ep_mp_group = None
    module.dense_dp_cp_group = None
    module.expert_param_group = None
    return module


def _gather_ulysses_sequence(local_tensor: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size(cp_group))]
    dist.all_gather(gathered, local_tensor.contiguous(), group=cp_group)
    return torch.cat(gathered, dim=1)


def _assert_ep_dispatch_folds_context_parallel(module: MoEV2TransformerTrainModule) -> None:
    cp_ranks = set(dist.get_process_group_ranks(module.cp_group))
    ep_mp_ranks = set(dist.get_process_group_ranks(module.ep_mp_group))
    assert cp_ranks <= ep_mp_ranks


def _run_moe_v2_cp_dp_mesh_groups() -> None:
    module = _new_train_module_for_mesh()

    module._build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.ddp),
        cp=ContextParallelConfig(degree=2),
        device_type="cpu",
    )

    assert module.world_mesh["dense"].shape == (2, 2)
    assert module.world_mesh["dense"].mesh_dim_names == ("dp", "cp")
    assert module.world_mesh["moe"] is None

    assert dist.get_world_size(module.dp_group) == 2
    assert dist.get_world_size(module.cp_group) == 2
    assert dist.get_world_size(module.dense_dp_cp_group) == 4


def test_moe_v2_cp_dp_mesh_groups_cpu() -> None:
    run_distributed_test(
        _run_moe_v2_cp_dp_mesh_groups,
        backend="gloo",
        world_size=4,
        start_method="spawn",
    )


def _run_moe_v2_cp_ep_mesh_groups() -> None:
    module = _new_train_module_for_mesh()

    module._build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.ddp),
        cp=ContextParallelConfig(degree=2),
        ep=ExpertParallelConfig(degree=4),
        device_type="cpu",
    )

    assert module.world_mesh["dense"].shape == (2, 2)
    assert module.world_mesh["dense"].mesh_dim_names == ("dp", "cp")
    assert module.world_mesh["moe"].shape == (1, 4)
    assert module.world_mesh["moe"].mesh_dim_names == ("ep_dp", "ep_mp")

    assert dist.get_world_size(module.dp_group) == 2
    assert dist.get_world_size(module.cp_group) == 2
    assert dist.get_world_size(module.dense_dp_cp_group) == 4
    assert dist.get_world_size(module.ep_mp_group) == 4
    assert dist.get_world_size(module.ep_dp_group) == 1
    assert module.expert_param_group is module.ep_dp_group

    _assert_ep_dispatch_folds_context_parallel(module)


def test_moe_v2_cp_ep_mesh_groups_cpu() -> None:
    run_distributed_test(
        _run_moe_v2_cp_ep_mesh_groups,
        backend="gloo",
        world_size=4,
        start_method="spawn",
    )


def _run_moe_v2_cp_pp_mesh_groups() -> None:
    module = _new_train_module_for_mesh()

    module._build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.ddp),
        cp=ContextParallelConfig(degree=2),
        pp=PipelineParallelConfig(degree=2),
        device_type="cpu",
    )

    assert module.world_mesh["dense"].shape == (2, 1, 2)
    assert module.world_mesh["dense"].mesh_dim_names == ("pp", "dp", "cp")
    assert module.world_mesh["moe"] is None

    assert dist.get_world_size(module.pp_group) == 2
    assert dist.get_world_size(module.dp_group) == 1
    assert dist.get_world_size(module.cp_group) == 2
    assert dist.get_world_size(module.dense_dp_cp_group) == 2


def test_moe_v2_cp_pp_mesh_groups_cpu() -> None:
    run_distributed_test(
        _run_moe_v2_cp_pp_mesh_groups,
        backend="gloo",
        world_size=4,
        start_method="spawn",
    )


def _run_moe_v2_cp_ep_pp_mesh_groups() -> None:
    module = _new_train_module_for_mesh()

    module._build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.ddp),
        cp=ContextParallelConfig(degree=2),
        ep=ExpertParallelConfig(degree=4),
        pp=PipelineParallelConfig(degree=2),
        device_type="cpu",
    )

    assert module.world_mesh["dense"].shape == (2, 2, 2)
    assert module.world_mesh["dense"].mesh_dim_names == ("pp", "dp", "cp")
    assert module.world_mesh["moe"].shape == (2, 1, 4)
    assert module.world_mesh["moe"].mesh_dim_names == ("pp", "ep_dp", "ep_mp")

    assert dist.get_world_size(module.pp_group) == 2
    assert dist.get_world_size(module.dp_group) == 2
    assert dist.get_world_size(module.cp_group) == 2
    assert dist.get_world_size(module.dense_dp_cp_group) == 4
    assert dist.get_world_size(module.ep_mp_group) == 4
    assert dist.get_world_size(module.ep_dp_group) == 1
    assert module.expert_param_group is module.ep_dp_group

    _assert_ep_dispatch_folds_context_parallel(module)


def test_moe_v2_cp_ep_pp_mesh_groups_cpu() -> None:
    run_distributed_test(
        _run_moe_v2_cp_ep_pp_mesh_groups,
        backend="gloo",
        world_size=8,
        start_method="spawn",
    )


def test_pipeline_split_replicates_context_parallel_metadata() -> None:
    schedule = CustomScheduleInterleaved1F1B.__new__(CustomScheduleInterleaved1F1B)
    schedule._n_microbatches = 2
    schedule._args_chunk_spec = None
    schedule._kwargs_chunk_spec = None

    input_ids = torch.arange(16).view(4, 4)
    labels = input_ids.clone()
    args_split, kwargs_split = schedule._split_inputs(
        (input_ids,),
        {
            "labels": labels,
            "cp_already_sharded": True,
            "cp_original_seq_len": 8,
        },
    )

    assert [args[0].shape for args in args_split] == [torch.Size([2, 4]), torch.Size([2, 4])]
    assert [kwargs["labels"].shape for kwargs in kwargs_split] == [
        torch.Size([2, 4]),
        torch.Size([2, 4]),
    ]
    assert [kwargs["cp_already_sharded"] for kwargs in kwargs_split] == [True, True]
    assert [kwargs["cp_original_seq_len"] for kwargs in kwargs_split] == [8, 8]


def _run_moe_v2_folded_cp_ep_forward_backward() -> None:
    module = _new_train_module_for_mesh()
    module._build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.ddp),
        cp=ContextParallelConfig(degree=2),
        ep=ExpertParallelConfig(degree=4),
        device_type="cuda",
    )

    cp_group_ranks = set(dist.get_process_group_ranks(module.cp_group))
    ep_group_ranks = set(dist.get_process_group_ranks(module.ep_mp_group))
    assert dist.get_world_size(module.cp_group) == 2
    assert dist.get_world_size(module.ep_mp_group) == 4
    assert cp_group_ranks <= ep_group_ranks

    block = _build_block(init_device="cuda")
    cp_config = TransformerContextParallelConfig.ulysses(degree=2)
    block.apply_cp(module.world_mesh["dense"]["cp"], ring=cp_config.ring, uly=cp_config.uly)
    block.apply_ep(module.world_mesh["moe"])
    _init_block_params(block)
    _install_deterministic_topk_router(block)
    block.train()

    torch.manual_seed(1234 + dist.get_rank())
    x = torch.randn(
        1,
        4,
        block.d_model,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    loss = y.float().square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    expert_grad = block.routed_experts.w_up_gate.grad
    assert expert_grad is not None
    assert torch.isfinite(expert_grad).all()
    assert expert_grad.abs().sum() > 0


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
@requires_multi_gpu
@requires_grouped_gemm
def test_moe_v2_folded_cp_ep_forward_backward_cuda() -> None:
    run_distributed_test(
        _run_moe_v2_folded_cp_ep_forward_backward,
        backend="nccl",
        world_size=4,
        start_method="spawn",
    )


def _run_moe_v2_folded_cp_rowwise_ep_forward_backward() -> None:
    module = _new_train_module_for_mesh()
    module._build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.ddp),
        cp=ContextParallelConfig(degree=2),
        ep=ExpertParallelConfig(degree=4),
        device_type="cuda",
    )

    block = _build_block(
        init_device="cuda",
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
    )
    cp_config = TransformerContextParallelConfig.ulysses(degree=2)
    block.apply_cp(module.world_mesh["dense"]["cp"], ring=cp_config.ring, uly=cp_config.uly)
    block.apply_ep(module.world_mesh["moe"])
    _init_block_params(block)
    _install_deterministic_topk_router(block)
    block.train()

    torch.manual_seed(2468 + dist.get_rank())
    x = torch.randn(
        1,
        4,
        block.d_model,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    loss = y.float().square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    expert_grad = block.routed_experts.w_up_gate.grad
    assert expert_grad is not None
    assert torch.isfinite(expert_grad).all()
    assert expert_grad.abs().sum() > 0


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
@requires_multi_gpu
@requires_grouped_gemm
def test_moe_v2_folded_cp_rowwise_ep_forward_backward_cuda() -> None:
    run_distributed_test(
        _run_moe_v2_folded_cp_rowwise_ep_forward_backward,
        backend="nccl",
        world_size=4,
        start_method="spawn",
    )


def _run_moe_v2_folded_cp_ep_model_forward_backward() -> None:
    module = _new_train_module_for_mesh()
    module._build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.ddp),
        cp=ContextParallelConfig(degree=2),
        ep=ExpertParallelConfig(degree=4),
        device_type="cuda",
    )

    model = _build_model(init_device="cuda")
    cp_config = TransformerContextParallelConfig.ulysses(degree=2)
    model.apply_cp(module.world_mesh["dense"]["cp"], ring=cp_config.ring, uly=cp_config.uly)
    model.apply_ep(
        dp_mesh=module.world_mesh["dense"]["dp"],
        ep_mesh=module.world_mesh["moe"],
    )
    _init_module_params(model)
    for block in model.blocks.values():
        assert isinstance(block, OLMoDDPTransformerBlock)
        _install_deterministic_topk_router(block)
    model.train()

    torch.manual_seed(4321)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), device="cuda")
    labels = input_ids.clone()
    lm_output = model(
        input_ids,
        labels=labels,
        loss_reduction="sum",
        loss_div_factor=float(labels.numel()),
        return_logits=False,
    )

    assert lm_output.logits is None
    assert torch.isfinite(lm_output.loss)
    assert torch.isfinite(lm_output.ce_loss)

    lm_output.loss.backward()

    assert model.embeddings.weight.grad is not None
    assert torch.isfinite(model.embeddings.weight.grad).all()

    first_block = next(iter(model.blocks.values()))
    assert isinstance(first_block, OLMoDDPTransformerBlock)
    expert_grad = first_block.routed_experts.w_up_gate.grad
    assert expert_grad is not None
    assert torch.isfinite(expert_grad).all()
    assert expert_grad.abs().sum() > 0


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
@requires_multi_gpu
@requires_grouped_gemm
def test_moe_v2_folded_cp_ep_model_forward_backward_cuda() -> None:
    run_distributed_test(
        _run_moe_v2_folded_cp_ep_model_forward_backward,
        backend="nccl",
        world_size=4,
        start_method="spawn",
    )


def _run_moe_v2_folded_cp_ep_logits_parity_cuda() -> None:
    ref_module = _new_train_module_for_mesh()
    ref_module._build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.ddp),
        ep=ExpertParallelConfig(degree=4),
        device_type="cuda",
    )

    cp_module = _new_train_module_for_mesh()
    cp_module._build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.ddp),
        cp=ContextParallelConfig(degree=2),
        ep=ExpertParallelConfig(degree=4),
        device_type="cuda",
    )

    ref_model = _build_model(init_device="cuda")
    ref_model.apply_ep(
        dp_mesh=ref_module.world_mesh["dense"]["dp"],
        ep_mesh=ref_module.world_mesh["moe"],
    )

    cp_model = _build_model(init_device="cuda")
    cp_config = TransformerContextParallelConfig.ulysses(degree=2)
    cp_model.apply_cp(cp_module.world_mesh["dense"]["cp"], ring=cp_config.ring, uly=cp_config.uly)
    cp_model.apply_ep(
        dp_mesh=cp_module.world_mesh["dense"]["dp"],
        ep_mesh=cp_module.world_mesh["moe"],
    )

    _init_module_params(ref_model)
    _init_module_params(cp_model)
    for model in (ref_model, cp_model):
        for block in model.blocks.values():
            assert isinstance(block, OLMoDDPTransformerBlock)
            _install_forced_router(block)
        model.eval()

    torch.manual_seed(2026)
    input_ids = torch.randint(0, ref_model.vocab_size, (1, 8), device="cuda")

    with torch.no_grad():
        ref_logits = ref_model(input_ids, return_logits=True)
        cp_logits = cp_model(input_ids, return_logits=True)

    assert isinstance(ref_logits, torch.Tensor)
    assert isinstance(cp_logits, torch.Tensor)
    assert ref_logits.shape == (1, 8, ref_model.vocab_size)
    assert cp_logits.shape == (1, 4, cp_model.vocab_size)

    gathered_cp_logits = _gather_ulysses_sequence(cp_logits, cp_module.cp_group)
    torch.testing.assert_close(gathered_cp_logits, ref_logits, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
@requires_multi_gpu
@requires_grouped_gemm
def test_moe_v2_folded_cp_ep_logits_parity_cuda() -> None:
    run_distributed_test(
        _run_moe_v2_folded_cp_ep_logits_parity_cuda,
        backend="nccl",
        world_size=4,
        start_method="spawn",
    )


def _run_moe_v2_folded_cp_rowwise_ep_train_module_cuda() -> None:
    model = _build_model(
        init_device="meta",
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
    )
    train_module = MoEV2TransformerTrainModule(
        model=model,
        optim=MoEFusedV2OptimizerConfig(),
        rank_microbatch_size=8,
        max_sequence_length=8,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
        cp_config=TransformerContextParallelConfig.ulysses(degree=2),
        ep_config=ExpertParallelConfig(degree=4),
        device=torch.device("cuda"),
    )

    assert train_module._cp_local_rank_microbatch_size(8) == 4
    assert train_module.world_mesh["dense"].shape == (2, 2)
    assert train_module.world_mesh["moe"].shape == (1, 4)

    for model_part in train_module.model_parts:
        unwrapped = getattr(model_part, "module", model_part)
        for block in unwrapped.blocks.values():
            assert isinstance(block, OLMoDDPTransformerBlock)
            assert block.ep_no_sync
            assert block.ep_no_sync_use_rowwise_all_to_all
            _install_deterministic_topk_router(block)

    recorded_ce_losses: list[torch.Tensor] = []

    def record_ce_loss(value, *args, **kwargs):
        del args, kwargs
        recorded_ce_losses.append(value.detach().clone())

    trainer = SimpleNamespace(
        global_batch_size=16,
        dp_process_group=train_module.dp_process_group,
        record_ce_loss=record_ce_loss,
        record_metric=lambda *args, **kwargs: None,
    )
    train_module._attach_trainer(trainer)

    torch.manual_seed(1357)
    input_ids = torch.randint(0, model.vocab_size, (1, 8), device="cuda")
    train_module.train_batch(
        {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }
    )

    assert recorded_ce_losses
    assert torch.isfinite(recorded_ce_losses[-1])

    expert_grads = []
    for model_part in train_module.model_parts:
        unwrapped = getattr(model_part, "module", model_part)
        for block in unwrapped.blocks.values():
            assert isinstance(block, OLMoDDPTransformerBlock)
            for param in (block.routed_experts.w_up_gate, block.routed_experts.w_down):
                grad = getattr(param, "_main_grad_fp32", None)
                if grad is None:
                    grad = param.grad
                if grad is not None:
                    expert_grads.append(grad)

    assert expert_grads
    assert all(torch.isfinite(grad).all() for grad in expert_grads)
    assert any(grad.abs().sum() > 0 for grad in expert_grads)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
@requires_multi_gpu
@requires_grouped_gemm
def test_moe_v2_folded_cp_rowwise_ep_train_module_cuda() -> None:
    run_distributed_test(
        _run_moe_v2_folded_cp_rowwise_ep_train_module_cuda,
        backend="nccl",
        world_size=4,
        start_method="spawn",
    )


def _run_moe_v2_folded_cp_ep_pp_train_module_cuda() -> None:
    os.environ["OLMO_PP_SCHEDULE_PLOT"] = "0"

    model = _build_model(init_device="meta", n_layers=4)
    train_module = MoEV2TransformerTrainModule(
        model=model,
        optim=MoEFusedV2OptimizerConfig(),
        rank_microbatch_size=8,
        max_sequence_length=8,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
        cp_config=TransformerContextParallelConfig.ulysses(degree=2),
        ep_config=ExpertParallelConfig(degree=4),
        pp_config=TransformerPipelineParallelConfig(
            degree=2,
            schedule=PipelineScheduleType.custom_interleaved_1F1B,
            style=PipelineSplitStyle.loop,
            use_custom_stage_implementation=True,
            p2p_backend=PipelineP2PBackend.nccl,
        ),
        device=torch.device("cuda"),
    )

    assert train_module.world_mesh["dense"].shape == (2, 2, 2)
    assert train_module.world_mesh["moe"].shape == (2, 1, 4)
    assert dist.get_world_size(train_module.dense_dp_cp_group) == 4
    assert dist.get_world_size(train_module.ep_mp_group) == 4
    assert dist.get_world_size(train_module.expert_param_group) == 1

    for model_part in train_module.model_parts:
        unwrapped = getattr(model_part, "module", model_part)
        for block in unwrapped.blocks.values():
            assert isinstance(block, OLMoDDPTransformerBlock)
            _install_deterministic_topk_router(block)

    recorded_ce_losses: list[torch.Tensor] = []

    def record_ce_loss(value, *args, **kwargs):
        del args, kwargs
        recorded_ce_losses.append(value.detach().clone())

    trainer = SimpleNamespace(
        global_batch_size=32,
        dp_process_group=train_module.dp_process_group,
        record_ce_loss=record_ce_loss,
        record_metric=lambda *args, **kwargs: None,
    )
    train_module._attach_trainer(trainer)

    torch.manual_seed(6789)
    input_ids = torch.randint(0, model.vocab_size, (2, 8), device="cuda")
    batch = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
    }
    train_module.train_batch(batch)

    assert recorded_ce_losses
    assert torch.isfinite(recorded_ce_losses[-1])

    expert_grads = []
    for model_part in train_module.model_parts:
        unwrapped = getattr(model_part, "module", model_part)
        for block in unwrapped.blocks.values():
            assert isinstance(block, OLMoDDPTransformerBlock)
            for param in (block.routed_experts.w_up_gate, block.routed_experts.w_down):
                grad = getattr(param, "_main_grad_fp32", None)
                if grad is None:
                    grad = param.grad
                if grad is not None:
                    expert_grads.append(grad)

    assert expert_grads
    assert all(torch.isfinite(grad).all() for grad in expert_grads)
    assert any(grad.abs().sum() > 0 for grad in expert_grads)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Requires at least 8 GPUs")
@requires_multi_gpu
@requires_grouped_gemm
def test_moe_v2_folded_cp_ep_pp_train_module_cuda() -> None:
    run_distributed_test(
        _run_moe_v2_folded_cp_ep_pp_train_module_cuda,
        backend="nccl",
        world_size=8,
        start_method="spawn",
    )


def _run_moe_v2_folded_cp_rowwise_ep_pp_train_module_cuda() -> None:
    os.environ["OLMO_PP_SCHEDULE_PLOT"] = "0"

    model = _build_model(
        init_device="meta",
        n_layers=4,
        ep_no_sync=True,
        ep_no_sync_use_rowwise_all_to_all=True,
    )
    train_module = MoEV2TransformerTrainModule(
        model=model,
        optim=MoEFusedV2OptimizerConfig(),
        rank_microbatch_size=8,
        max_sequence_length=8,
        dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp),
        cp_config=TransformerContextParallelConfig.ulysses(degree=2),
        ep_config=ExpertParallelConfig(degree=4),
        pp_config=TransformerPipelineParallelConfig(
            degree=2,
            schedule=PipelineScheduleType.custom_interleaved_1F1B,
            style=PipelineSplitStyle.loop,
            use_custom_stage_implementation=True,
            p2p_backend=PipelineP2PBackend.nccl,
        ),
        device=torch.device("cuda"),
    )

    assert train_module._cp_local_rank_microbatch_size(8) == 4
    assert train_module.world_mesh["dense"].shape == (2, 2, 2)
    assert train_module.world_mesh["moe"].shape == (2, 1, 4)

    for model_part in train_module.model_parts:
        unwrapped = getattr(model_part, "module", model_part)
        for block in unwrapped.blocks.values():
            assert isinstance(block, OLMoDDPTransformerBlock)
            assert block.ep_no_sync
            assert block.ep_no_sync_use_rowwise_all_to_all
            _install_deterministic_topk_router(block)

    recorded_ce_losses: list[torch.Tensor] = []

    def record_ce_loss(value, *args, **kwargs):
        del args, kwargs
        recorded_ce_losses.append(value.detach().clone())

    trainer = SimpleNamespace(
        global_batch_size=32,
        dp_process_group=train_module.dp_process_group,
        record_ce_loss=record_ce_loss,
        record_metric=lambda *args, **kwargs: None,
    )
    train_module._attach_trainer(trainer)

    torch.manual_seed(9753)
    input_ids = torch.randint(0, model.vocab_size, (2, 8), device="cuda")
    train_module.train_batch(
        {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }
    )

    assert recorded_ce_losses
    assert torch.isfinite(recorded_ce_losses[-1])

    expert_grads = []
    for model_part in train_module.model_parts:
        unwrapped = getattr(model_part, "module", model_part)
        for block in unwrapped.blocks.values():
            assert isinstance(block, OLMoDDPTransformerBlock)
            for param in (block.routed_experts.w_up_gate, block.routed_experts.w_down):
                grad = getattr(param, "_main_grad_fp32", None)
                if grad is None:
                    grad = param.grad
                if grad is not None:
                    expert_grads.append(grad)

    assert expert_grads
    assert all(torch.isfinite(grad).all() for grad in expert_grads)
    assert any(grad.abs().sum() > 0 for grad in expert_grads)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Requires at least 8 GPUs")
@requires_multi_gpu
@requires_grouped_gemm
def test_moe_v2_folded_cp_rowwise_ep_pp_train_module_cuda() -> None:
    run_distributed_test(
        _run_moe_v2_folded_cp_rowwise_ep_pp_train_module_cuda,
        backend="nccl",
        world_size=8,
        start_method="spawn",
    )


def _run_moe_v2_block_apply_cp() -> None:
    cp_mesh = DeviceMesh(
        device_type="cpu",
        mesh=torch.arange(dist.get_world_size(), dtype=torch.int),
        mesh_dim_names=("cp",),
    )
    cp_config = TransformerContextParallelConfig.ulysses(degree=dist.get_world_size())

    block = _build_block()
    block.apply_cp(cp_mesh, ring=cp_config.ring, uly=cp_config.uly)

    assert block.attention.cp_enabled
    assert block.routed_experts_router is not None
    assert block.routed_experts_router.cp_mesh is cp_mesh


def test_moe_v2_block_apply_cp_cpu() -> None:
    run_distributed_test(
        _run_moe_v2_block_apply_cp,
        backend="gloo",
        world_size=2,
        start_method="spawn",
    )
