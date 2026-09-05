"""Exercise the opt-in batched-count model assembly with actual no-EP blocks and DDP."""

import os
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.moe.emo import EmoRouterConfig
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.testing import requires_multi_gpu, run_distributed_test
from olmo_core.train.train_module import OLMoDDPTrainModuleConfig
from olmo_core.train.train_module.transformer import TransformerDataParallelConfig

from .ddp_train_module_test import _tiny_model_config


def _run_batched_model_parity(compiled):
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    stacks = []
    for key in (
        "OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE",
        "OLMO_PROFILE_SWIGLU_PAIRWISE",
        "OLMO_PROFILE_EMO_DOCUMENT_POOL",
        "OLMO_PROFILE_ROUNDED_WGRAD",
        "OLMO_PROFILE_RS_SINGLE_PARAM_FAST_PATH",
    ):
        os.environ[key] = "1"
    os.environ["OLMO_PROFILE_LB_COUNT_OVERLAP"] = "0"
    for enabled in (False, True):
        os.environ["OLMO_PROFILE_DDP_DEFER_REPLICATED_REDUCTIONS"] = str(int(enabled))
        os.environ["OLMO_PROFILE_LB_COUNT_BATCHED"] = str(int(enabled))
        torch.manual_seed(331)
        config = _tiny_model_config(d_model=512, n_layers=2, dtype=DType.bfloat16)
        config.block.routed_experts.num_experts = 16
        config.block.routed_experts.hidden_size = 1024
        router = config.block.routed_experts_router
        router.num_experts, router.top_k = 16, 4
        router.global_load_balancing = True
        router.lb_loss_weight, router.z_loss_weight = 0.01, 1e-5
        router.emo = EmoRouterConfig(
            eos_token_id=0, min_document_expert_pool=4, max_document_expert_pool=16
        )
        model = config.build(init_device="cuda")
        module = OLMoDDPTrainModuleConfig(
            rank_microbatch_size=256,
            max_sequence_length=128,
            compile_model=compiled,
            optim=OLMoDDPOptimizerConfig(lr=1e-3),
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.ddp, use_reduce_scatter=True
            ),
        ).build(model, device=device, eval_only=False)
        assert module.optim is not None
        assert not module.model.module.recompute_each_block
        stacks.append(module)
    for update in range(3):
        torch.manual_seed(931 + update + rank)
        batches = [torch.randint(0, 128, (2, 128), device=device) for _ in range(8)]
        outputs = []
        for enabled, module in enumerate(stacks):
            os.environ["OLMO_PROFILE_LB_COUNT_BATCHED"] = str(enabled)
            values = []
            torch.manual_seed(1123 + update + rank)
            for index, tokens in enumerate(batches):
                with module.model.no_sync() if index < 7 else nullcontext():
                    out = module.model(
                        tokens,
                        labels=tokens.roll(-1, 1),
                        loss_reduction="sum",
                        loss_div_factor=256.0 * 8,
                        z_loss_multiplier=1e-5,
                    )
                    out.loss.backward()
                    values.append(out.ce_loss.detach())
            module.model.finalize_grad_reduce()
            outputs.append(torch.stack(values))
        torch.testing.assert_close(outputs[1], outputs[0], rtol=2e-5, atol=1e-7)
        for (name, ref), (_, new) in zip(
            stacks[0].model.named_parameters(), stacks[1].model.named_parameters()
        ):
            ref_grad = getattr(ref, "_olmo_ddp_reduced_grad_shard", ref._main_grad_fp32)
            new_grad = getattr(new, "_olmo_ddp_reduced_grad_shard", new._main_grad_fp32)
            torch.testing.assert_close(new_grad, ref_grad, rtol=2e-5, atol=1e-7, msg=name)
        for module in stacks:
            module.optim.step()
            assert not bool(module.optim._step_skipped.item())
        for (name, ref), (_, new) in zip(
            stacks[0].model.named_parameters(), stacks[1].model.named_parameters()
        ):
            torch.testing.assert_close(new, ref, rtol=2e-5, atol=1e-7, msg=name)
        assert stacks[0].optim.states.keys() == stacks[1].optim.states.keys()
        for name in stacks[0].optim.states:
            torch.testing.assert_close(
                stacks[1].optim.states[name].to_local(),
                stacks[0].optim.states[name].to_local(),
                rtol=2e-5,
                atol=1e-7,
                msg=name,
            )
        for module in stacks:
            module.zero_grads()
            for block in module.model.module.blocks.values():
                block.routed_experts_router.reset_metrics()


@requires_multi_gpu
@pytest.mark.parametrize("compiled", [False, True])
def test_batched_lb_model_and_optimizer_parity(compiled):
    """Test complete two-layer model, CE/z/LB, mixed gradient collectives and three updates."""
    run_distributed_test(
        _run_batched_model_parity,
        world_size=2,
        backend="nccl",
        start_method="spawn",
        func_args=(compiled,),
    )
