"""Joint no-EP routing/gradient-collective qualification with the sharded optimizer."""

import os
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe.emo import EmoRouterConfig
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.parallel import MultiGroupDistributedDataParallel
from olmo_core.nn.transformer import TransformerBlockType
from olmo_core.optim.moe_optimizer import OLMoDDPOptimizer
from olmo_core.testing import run_distributed_test


def _run_joint_parity(compiled):
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
    stacks = []
    # Keep the already-qualified expert math identical in both stacks.
    for key in (
        "OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE",
        "OLMO_PROFILE_SWIGLU_PAIRWISE",
        "OLMO_PROFILE_EMO_DOCUMENT_POOL",
        "OLMO_PROFILE_EMO_TOP16",
        "OLMO_PROFILE_ROUNDED_WGRAD",
        "OLMO_PROFILE_RS_SINGLE_PARAM_FAST_PATH",
    ):
        os.environ[key] = "1"
    for enabled in (False, True):
        os.environ["OLMO_PROFILE_DDP_DEFER_REPLICATED_REDUCTIONS"] = str(int(enabled))
        os.environ["OLMO_PROFILE_LB_COUNT_OVERLAP"] = str(int(enabled))
        torch.manual_seed(431)
        block = OLMoDDPTransformerBlockConfig(
            name=TransformerBlockType.moe_fused_v2,
            attention=AttentionConfig(
                name=AttentionType.default,
                n_heads=4,
                bias=False,
                use_flash=False,
                dtype=DType.bfloat16,
            ),
            layer_norm=LayerNormConfig(name=LayerNormType.rms, bias=False, dtype=DType.float32),
            routed_experts=RoutedExpertsConfig(
                d_model=512, hidden_size=1024, num_experts=512, bias=False, dtype=DType.bfloat16
            ),
            routed_experts_router=MoERouterConfigV2(
                d_model=512,
                num_experts=512,
                top_k=16,
                dtype=DType.float32,
                lb_loss_weight=0.01,
                z_loss_weight=1e-5,
                global_load_balancing=True,
                emo=EmoRouterConfig(
                    eos_token_id=0, min_document_expert_pool=16, max_document_expert_pool=512
                ),
            ),
            shared_experts=None,
        ).build(d_model=512, block_idx=0, n_layers=1, init_device=str(device))
        with torch.no_grad():
            for name, param in block.named_parameters():
                if param.ndim == 1 and "norm" in name:
                    param.fill_(1)
                else:
                    param.normal_(0, 0.02)
        # Production materializes the entire model in BF16 before optimizer build;
        # router arithmetic still explicitly casts to FP32 inside its forward.
        block.to(dtype=torch.bfloat16)
        router = block.routed_experts_router
        router.set_load_balancing_process_group(dist.group.WORLD)
        assert router.load_balancing_loss is not None
        assert router.z_loss is not None
        assert router.global_batch_size_per_expert is not None
        if compiled:
            block.compile(dynamic=False)
        ddp = MultiGroupDistributedDataParallel(
            block,
            init_sync=False,
            accumulate_grads_in_fp32=True,
            reduce_grads_in_fp32=True,
            use_reduce_scatter=True,
            bucket_cap_mb=1,
        )
        optimizer = OLMoDDPOptimizer(
            [{"named_params": dict(ddp.named_parameters()), "pg": "dp"}],
            world_mesh={"dense": mesh, "moe": None},
            dp_group=dist.group.WORLD,
            model_has_grad_accum_fp32_buffer=True,
            use_distributed=True,
            lr=1e-3,
            betas=(0.9, 0.95),
            max_grad_norm=1.0,
        )
        ddp.configure_reduce_scatter_params(optimizer.normal_params_with_sharded_optimizer_state())
        assert any(bucket.reduce_scatter for bucket in ddp._grad_buckets)
        assert any(not bucket.reduce_scatter for bucket in ddp._grad_buckets)
        stacks.append((ddp, optimizer, router))
    segment_ids = torch.arange(256, device=device).div(32, rounding_mode="floor").reshape(1, -1)
    for step in range(3):
        torch.manual_seed(781 + step + rank)
        inputs = [torch.randn(1, 256, 512, device=device, dtype=torch.bfloat16) for _ in range(8)]
        losses = []
        for enabled, (ddp, optimizer, router) in enumerate(stacks):
            os.environ["OLMO_PROFILE_LB_COUNT_OVERLAP"] = str(enabled)
            values = []
            torch.manual_seed(913 + step + rank)
            for index, x in enumerate(inputs):
                with ddp.no_sync() if index < 7 else nullcontext():
                    out = ddp(x, loss_div_factor=256.0 * 8, segment_ids=segment_ids)
                    loss = out.float().square().mean() / 8
                    loss.backward()
                    values.append(loss.detach())
            ddp.finalize_grad_reduce()
            losses.append(torch.stack(values))
            optimizer.step()
            assert not bool(optimizer._step_skipped.item())
        torch.testing.assert_close(losses[1], losses[0], rtol=2e-5, atol=1e-7)
        for (name, ref), (_, new) in zip(
            stacks[0][0].named_parameters(), stacks[1][0].named_parameters()
        ):
            torch.testing.assert_close(new, ref, rtol=2e-5, atol=1e-7, msg=name)
            ref_grad = getattr(ref, "_olmo_ddp_reduced_grad_shard", ref._main_grad_fp32)
            new_grad = getattr(new, "_olmo_ddp_reduced_grad_shard", new._main_grad_fp32)
            torch.testing.assert_close(new_grad, ref_grad, rtol=2e-5, atol=1e-7, msg=name)
        ref_opt, new_opt = stacks[0][1], stacks[1][1]
        assert ref_opt.states.keys() == new_opt.states.keys()
        for name in ref_opt.states:
            torch.testing.assert_close(
                new_opt.states[name].to_local(),
                ref_opt.states[name].to_local(),
                rtol=2e-5,
                atol=1e-7,
                msg=name,
            )
        torch.testing.assert_close(
            stacks[1][2].global_batch_size_per_expert,
            stacks[0][2].global_batch_size_per_expert,
            rtol=0,
            atol=0,
        )
        for ddp, _, router in stacks:
            ddp.zero_grad(set_to_none=False)
            router.reset_metrics()


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
def test_no_ep_overlap_with_deferred_reductions_and_sharded_adam(compiled):
    """Exercise actual no-EP aux-loss attachment over3x8 microbatches and both reduction kinds."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    run_distributed_test(
        _run_joint_parity,
        backend="nccl",
        start_method="spawn",
        func_args=(compiled,),
    )
