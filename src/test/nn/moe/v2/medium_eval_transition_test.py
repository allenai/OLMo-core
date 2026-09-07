"""Bounded EP8 train/eval transition regression at production medium dimensions."""

import os

import pytest
import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.testing import run_distributed_test
from olmo_core.train.train_module import OLMoDDPTrainModuleConfig
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)


def _exercise_eval_transitions():
    from kernel_fun._common import support

    from examples.olmo_ddp.olmoe3_small_medium_models import build_model_config

    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    support.MIN_CTAS = 128
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
    for key in ("OLMO_PROFILE_LB_COUNT_BATCHED", "OLMO_PROFILE_LB_COUNT_BATCHED_EP"):
        os.environ[key] = "0"
    model = build_model_config("medium", eos_token_id=0, vocab_size=256)
    model.init_seed = 12536
    model.n_layers = 2
    model.block_overrides = {}
    model.validate()
    tm = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=2 * 8192,
        max_sequence_length=8192,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            accumulate_grads_in_fp32=True,
            reduce_grads_in_fp32=True,
            use_reduce_scatter=True,
            bucket_cap_mb=1,
        ),
        ep_config=TransformerExpertParallelConfig(degree=8),
        optim=OLMoDDPOptimizerConfig(
            lr=9.2e-4,
            betas=(0.9, 0.95),
            max_grad_norm=1.0,
            dtype=DType.float32,
            use_distributed=True,
            compile=False,
        ),
    ).build(model.build(init_device="meta"), device=torch.device("cuda", rank))
    torch.manual_seed(781 + rank)
    random = torch.randint(1, 256, (2, 8192), device="cuda")
    random[:, 31::32] = 0
    repeated = torch.full_like(random, 17 + rank)
    repeated[:, 1023::1024] = 0
    for cycle in range(3):
        print(f"EP_EVAL_TRANSITION rank={rank} cycle={cycle} train", flush=True)
        tm.model.train()
        out = tm.model(
            random,
            labels=random.roll(-1, 1),
            loss_reduction="sum",
            loss_div_factor=float(random.numel()),
            z_loss_multiplier=1e-5,
        )
        out.loss.backward()
        tm.model.finalize_grad_reduce()
        assert torch.isfinite(out.ce_loss).all()
        tm.optim.step()
        assert not bool(tm.optim._step_skipped.item())
        tm.model.zero_grad(set_to_none=True)
        del out
        tm.model.eval()
        with torch.no_grad():
            for repeat in range(2):
                for index, tokens in enumerate((random, repeated, repeated)):
                    print(
                        f"EP_EVAL_TRANSITION rank={rank} cycle={cycle} eval={repeat}/{index}",
                        flush=True,
                    )
                    out = tm.eval_batch({"input_ids": tokens}, labels=tokens.roll(-1, 1))
                    assert torch.isfinite(out.ce_loss).all()
                    del out
                # Match the evaluator's cleanup between consecutive passes.
                from olmo_core.utils import gc_cuda

                gc_cuda()
        dist.barrier()
    print(f"EP_EVAL_TRANSITION_PASS rank={rank}", flush=True)


@pytest.mark.gpu
def test_medium_compiled_train_eval_transitions():
    """Stress synced eval routing, repeated passes and skewed inputs; no Weka writes."""
    if torch.cuda.device_count() < 8:
        pytest.skip("requires eight CUDA GPUs")
    run_distributed_test(
        _exercise_eval_transitions,
        world_size=8,
        backend="nccl",
        start_method="spawn",
    )
