"""Layer drop under FSDP2 (2 ranks) and on GatedDeltaNet blocks. GPU only."""

import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F

from olmo_core.config import DType
from olmo_core.distributed.utils import get_full_tensor, get_rank
from olmo_core.nn.attention import GatedDeltaNetConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.testing import requires_gpu, requires_multi_gpu, run_distributed_test
from olmo_core.testing.utils import requires_fla
from olmo_core.utils import get_default_device, seed_all

VOCAB = 1000


def _ce(model, ids):
    logits = model(ids[:, :-1])
    return F.cross_entropy(logits.reshape(-1, VOCAB).float(), ids[:, 1:].reshape(-1))


def run_fsdp2_layer_drop_matches_single_process():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = get_default_device()
    seed_all(0)
    cfg = TransformerConfig.olmo2_190M(
        vocab_size=VOCAB, n_layers=4, fused_ops=False, dtype=DType.float32
    )
    model = cfg.build(init_device=dev)
    model.init_weights(max_seq_len=64, device=dev)
    ref = copy.deepcopy(model)
    model.apply_fsdp()
    for m in (model, ref):
        m.enable_layer_drop(p=0.5, seed=0)
        m.train()
    # SGD, not Adam: Adam's first step is ~lr * sign(g), which turns reduction-order noise in
    # near-zero grads into lr-sized parameter differences. The grads are compared directly below.
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.05)
    rank, world = get_rank(), dist.get_world_size()
    gen = torch.Generator().manual_seed(1)
    seen = []
    for step in range(1, 7):
        model._layer_drop.step = ref._layer_drop.step = step
        skipped = model._layer_drop.skipped()
        seen.append(skipped)
        batch = torch.randint(0, VOCAB, (2 * world, 33), generator=gen).to(dev)  # 2 micro-batches/rank
        local = batch[2 * rank : 2 * rank + 2]
        for mb in range(2):
            (_ce(model, local[mb : mb + 1]) / 2).backward()
        ref_loss = sum(_ce(ref, batch[i : i + 1]) for i in range(batch.shape[0])) / batch.shape[0]
        ref_loss.backward()
        for k, blk in model.blocks.items():
            has = [p.grad is not None for p in blk.parameters()]
            if int(k) in skipped:
                assert not any(has), (step, k)
            else:
                assert all(has), (step, k)
        for (n, p), (_, rp) in zip(model.named_parameters(), ref.named_parameters()):
            if p.grad is None:
                assert rp.grad is None, n  # skipped in both
                continue
            g = get_full_tensor(p.grad)
            assert torch.isfinite(g).all(), n
            torch.testing.assert_close(g, rp.grad, atol=1e-6, rtol=1e-3, msg=lambda m: f"{n} step {step}: {m}")
        opt.step()
        ref_opt.step()
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)
        for (n, p), (_, rp) in zip(model.named_parameters(), ref.named_parameters()):
            torch.testing.assert_close(
                get_full_tensor(p).detach(), rp.detach(), atol=1e-5, rtol=1e-4, msg=lambda m: f"{n} step {step}: {m}"
            )
    assert any(seen) and len(set(seen)) > 1, seen  # the draw skipped something and varied
    # eval runs every block: FSDP output equals the reference model with no skipping
    model.eval()
    ref.eval()
    with torch.no_grad():
        ids = torch.randint(0, VOCAB, (1, 16), generator=gen).to(dev)
        torch.testing.assert_close(model(ids), ref(ids), atol=2e-5, rtol=1e-4)


@requires_multi_gpu
def test_fsdp2_layer_drop_matches_single_process():
    run_distributed_test(
        run_fsdp2_layer_drop_matches_single_process, backend="nccl", start_method="spawn"
    )


@requires_gpu
@requires_fla
def test_gdn_blocks_are_skipped_whole():
    torch.manual_seed(0)
    ln = LayerNormConfig(name=LayerNormType.rms, bias=False)
    cfg = TransformerConfig(
        d_model=256,
        vocab_size=1024,
        n_layers=3,
        block=TransformerBlockConfig(
            name=TransformerBlockType.reordered_norm,
            sequence_mixer=GatedDeltaNetConfig(n_heads=8),
            layer_norm=ln,
            feed_forward=FeedForwardConfig(hidden_size=512, bias=False),
        ),
        lm_head=LMHeadConfig(layer_norm=ln, bias=False),
    )
    m = cfg.build(init_device="cuda")
    m.init_weights()
    m.enable_layer_drop(fixed={1})
    m.train()
    ids = torch.randint(0, 1024, (2, 64), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(ids)
        h = m.embeddings(ids)
        h = m.blocks["2"](m.blocks["0"](h))
        want = m.lm_head(h)
    torch.testing.assert_close(out, want)
    out.float().sum().backward()
    assert all(p.grad is None for p in m.blocks["1"].parameters())
    assert all(p.grad is not None for p in m.blocks["2"].parameters())
