"""Explicit EP opt-in: two real rowwise blocks, matching gradients and Adam states."""

import os
from contextlib import nullcontext
from test.nn.moe.v2.rounded_wgrad_ep_test import _build_model

import pytest
import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import unhide_from_torch
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.testing import run_distributed_test
from olmo_core.train.train_module import OLMoDDPTrainModuleConfig
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)


def _run_batched_ep_parity(
    ep_degree, width, hidden, experts, compiled, checkpoint_kda=False, balanced=False
):
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    for key in (
        "OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE",
        "OLMO_PROFILE_SWIGLU_PAIRWISE",
        "OLMO_PROFILE_EMO_DOCUMENT_POOL",
        "OLMO_PROFILE_EMO_TOP16",
        "OLMO_PROFILE_ROUNDED_WGRAD",
        "OLMO_PROFILE_ROUNDED_WGRAD_EP",
        "OLMO_PROFILE_RS_SINGLE_PARAM_FAST_PATH",
        "OLMO_EP_NO_SYNC_FORBID_RUNTIME_SYMM_ALLOC",
    ):
        os.environ[key] = "1"
    for key in (
        "OLMO_PROFILE_DDP_DEFER_REPLICATED_REDUCTIONS",
        "OLMO_PROFILE_LB_COUNT_OVERLAP",
        "OLMO_PROFILE_LB_COUNT_BATCHED",
        "OLMO_PROFILE_LB_COUNT_BATCHED_EP",
    ):
        os.environ[key] = "0"
    assert not (checkpoint_kda and balanced)
    production_kda = checkpoint_kda or balanced
    sequence, batch, micros = (8192, 4, 2) if production_kda else (256, 1, 8)
    if balanced:
        batch = 3
    if production_kda:
        from kernel_fun._common import support

        support.MIN_CTAS = 128
    stacks = []
    for enabled in range(2):
        config = OLMoDDPTrainModuleConfig(
            rank_microbatch_size=sequence * batch,
            max_sequence_length=sequence,
            compile_model=compiled,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.ddp,
                accumulate_grads_in_fp32=True,
                reduce_grads_in_fp32=True,
                use_reduce_scatter=True,
                bucket_cap_mb=1,
            ),
            ep_config=TransformerExpertParallelConfig(degree=ep_degree),
            optim=OLMoDDPOptimizerConfig(
                lr=1e-3,
                betas=(0.9, 0.95),
                max_grad_norm=1.0,
                dtype=DType.float32,
                use_distributed=True,
                compile=compiled,
            ),
        )
        if production_kda:
            from examples.olmo_ddp.olmoe3_small_medium_models import build_model_config

            model = build_model_config("medium", eos_token_id=0, vocab_size=256)
            # Two real production-width KDA+latent-MoE blocks, not a full-model
            # throughput test. Preserve private EP outputs and BF16/FP32 handling.
            model.init_seed = 12536
            model.n_layers = 2
            model.block_overrides = {}
            model.block.checkpoint_attn = bool(enabled and checkpoint_kda)
            model.block.ep.capacity_factor = 8.0  # Dropless parity fixture only.
            model.validate()
            model = model.build(init_device="meta")
        else:
            model = _build_model(width, hidden, experts, n_layers=2)
        tm = config.build(model, device=device)
        assert dist.get_world_size(tm.ep_mp_group) == ep_degree
        assert dist.get_world_size(tm.ep_dp_group) == 8 // ep_degree
        stacks.append(tm)

    def compare_states():
        ref, candidate = stacks
        ref_params = dict(ref.model.named_parameters())
        candidate_params = dict(candidate.model.named_parameters())
        assert ref_params.keys() == candidate_params.keys()
        for name in ref_params:
            torch.testing.assert_close(
                candidate_params[name], ref_params[name], rtol=2e-5, atol=1e-7, msg=name
            )
        assert ref.optim.states.keys() == candidate.optim.states.keys()
        for name in ref.optim.states:
            torch.testing.assert_close(
                candidate.optim.states[name].to_local(),
                ref.optim.states[name].to_local(),
                rtol=2e-5,
                atol=1e-7,
                msg=name,
            )

    compare_states()
    for step in range(3):
        torch.manual_seed(913 + step + rank)
        if balanced:
            from olmo_core.data.utils import split_batch_balanced

            sizes = [3, 3, 3, 3, 2, 2] * (2 if step == 1 else 1)
            full = torch.randint(1, 256, (sum(sizes), sequence), device=device)
            full[:, 31::32] = 0
            batches = list(full.split(sizes, dim=0))  # Explicit reference partition.
            split = split_batch_balanced(
                {"input_ids": full, "metadata": list(range(sum(sizes)))},
                3,
                partition_unit_instances=16,
            )
            assert [p["input_ids"].shape[0] for p in split] == sizes
            assert sum([p["metadata"] for p in split], []) == list(range(sum(sizes)))
            candidate_batches = [p["input_ids"] for p in split]
        else:
            sizes = [batch] * micros
            batches = [torch.randint(1, 256, (batch, sequence), device=device) for _ in sizes]
            for tokens in batches:
                tokens[:, 31::32] = 0
            candidate_batches = batches
        losses = []
        for enabled, tm in enumerate(stacks):
            os.environ["OLMO_PROFILE_LB_COUNT_BATCHED"] = str(enabled if not production_kda else 0)
            os.environ["OLMO_PROFILE_LB_COUNT_BATCHED_EP"] = str(
                enabled if not production_kda else 0
            )
            values = []
            torch.manual_seed(781 + step + rank)
            for micro, tokens in enumerate(candidate_batches if enabled else batches):
                diagnostic_sync = os.environ.get("OLMOE3_PARITY_DIAGNOSTIC_SYNC", "0") == "1"
                if diagnostic_sync and rank == 0:
                    print(
                        "EP_DIAGNOSTIC_FORWARD",
                        step,
                        enabled,
                        micro,
                        tuple(tokens.shape),
                        flush=True,
                    )
                with tm.model.no_sync() if micro < len(sizes) - 1 else nullcontext():
                    out = tm.model(
                        tokens,
                        labels=tokens.roll(-1, 1),
                        loss_reduction="sum",
                        loss_div_factor=float(sequence * sum(sizes)),
                        z_loss_multiplier=1e-5,
                    )
                    if diagnostic_sync:
                        torch.cuda.synchronize()
                        if rank == 0:
                            print("EP_DIAGNOSTIC_BACKWARD", step, enabled, micro, flush=True)
                    out.loss.backward()
                    if diagnostic_sync:
                        torch.cuda.synchronize()
                    values.append(out.ce_loss.detach())
            tm.model.finalize_grad_reduce()
            losses.append(torch.stack(values))
            for block in tm.model.module.routed_blocks():
                dropped = block._ep_no_sync_rowwise_drop_tokens_sum
                assert dropped is not None and unhide_from_torch(dropped).item() == 0
        torch.testing.assert_close(losses[1], losses[0], rtol=2e-5, atol=1e-7)
        for (name, ref), (_, candidate) in zip(
            stacks[0].model.named_parameters(), stacks[1].model.named_parameters()
        ):
            ref_grad = getattr(ref, "_olmo_ddp_reduced_grad_shard", ref._main_grad_fp32)
            candidate_grad = getattr(
                candidate, "_olmo_ddp_reduced_grad_shard", candidate._main_grad_fp32
            )
            torch.testing.assert_close(candidate_grad, ref_grad, rtol=2e-5, atol=1e-7, msg=name)
        for ref, candidate in zip(
            stacks[0].model.module.routed_blocks(), stacks[1].model.module.routed_blocks()
        ):
            for field in ("global_batch_size_per_expert", "load_balancing_loss", "z_loss"):
                torch.testing.assert_close(
                    getattr(candidate.routed_experts_router, field),
                    getattr(ref.routed_experts_router, field),
                    rtol=0 if field == "global_batch_size_per_expert" else 2e-5,
                    atol=0 if field == "global_batch_size_per_expert" else 1e-7,
                    msg=field,
                )
        for tm in stacks:
            tm.optim.step()
            assert not bool(tm.optim._step_skipped.item())
        compare_states()
        for tm in stacks:
            tm.model.zero_grad(set_to_none=(step == 1))
            for block in tm.model.module.routed_blocks():
                block.routed_experts_router.reset_metrics()
        if rank == 0:
            print(
                f"{'BALANCED' if balanced else 'KDA_AC' if checkpoint_kda else 'BATCHED'}_EP_PARITY "
                f"ep={ep_degree} width={width} hidden={hidden} "
                f"experts={experts} compiled={compiled} update={step+1} microbatches={sizes}: "
                "CE, gradients, parameters, Adam states, counts and auxiliary metrics pass",
                flush=True,
            )


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "ep_degree,width,hidden,experts", [(4, 512, 1024, 64), (8, 768, 1536, 512)]
)
def test_rowwise_batched_counts_and_sharded_adam(ep_degree, width, hidden, experts, compiled):
    """Cover nontrivial EP-DP reductions and the medium's local expert dimensions.

    This uses two ordinary attention blocks, not the full production KDA stack.
    Production timing retains capacity1.25; this fixture is deliberately dropless.
    """
    if torch.cuda.device_count() < 8:
        pytest.skip("requires8 CUDA GPUs")
    run_distributed_test(
        _run_batched_ep_parity,
        world_size=8,
        backend="nccl",
        start_method="spawn",
        func_args=(ep_degree, width, hidden, experts, compiled),
    )
