"""
Wall-clock train-step benchmark: sparse-landmark vs full attention, at the exact CPT/SFT settings
(Qwen3-4B all-attention and Qwen3.5-4B hybrid, 64k per rank, full AC, bf16, fused-linear loss).

Measures fwd+bwd time per step on one GPU (the per-rank compute that dominates multi-node step
time; optimizer + comms are attention-agnostic and mostly overlapped). Reports ms/step,
tokens/sec/GPU, peak memory, and the sparse/dense ratio.

Run on a GPU node (horton H200):
  python bench_train_step.py --configs qwen3_4b:dense qwen3_4b:sparse --seq-lens 16384 32768 65536
"""

import argparse
import gc
import json
import sys
import time

import torch

from olmo_core.data import TokenizerConfig
from olmo_core.nn.attention import AttentionBackendName, AttentionType
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)

MEM_FREQ = 63
BLOCK_SIZE = MEM_FREQ + 1


def build_model(model_name: str, attn: str, seq_len: int, ac: bool = True):
    if model_name == "qwen3_4b":
        tok = TokenizerConfig.qwen3()
        kwargs = dict(
            vocab_size=tok.padded_vocab_size(),
            attn_backend=AttentionBackendName.flash_2,
        )
        if attn == "sparse":
            kwargs.update(sparse_landmark=True, mem_freq=MEM_FREQ)
        elif attn == "fast":
            kwargs.update(fast_landmark=True, mem_freq=MEM_FREQ)
        elif attn != "dense":
            raise ValueError(attn)
        cfg = TransformerConfig.qwen3_4B(**kwargs)
    elif model_name == "qwen3_5_4b":
        tok = TokenizerConfig.qwen3_5()
        cfg = TransformerConfig.qwen3_5_4B(
            vocab_size=tok.padded_vocab_size(),
            attn_backend=AttentionBackendName.flash_2,
        )
        if attn == "sparse":
            mixer = cfg.block["attn"].sequence_mixer  # type: ignore[index]
            mixer.name = AttentionType.sparse_landmark
            mixer.mem_freq = MEM_FREQ
            mixer.num_landmarks = 1
        elif attn != "dense":
            raise ValueError(attn)
    else:
        raise ValueError(model_name)

    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear

    model = cfg.build(init_device="cuda")
    model.init_weights(max_seq_len=seq_len, device=torch.device("cuda"))
    model = model.to(torch.bfloat16)
    if ac:
        model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
    model.train()
    return model, tok


def bench_one(model_name: str, attn: str, seq_len: int, warmup: int, steps: int, batch: int, ac: bool = True):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tok = build_model(model_name, attn, seq_len, ac=ac)

    vocab = tok.vocab_size
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab - 1000, (batch, seq_len), device="cuda")
    if attn in ("sparse", "fast"):
        # landmark token at the end of every block, as LandmarkInstanceSource lays it out
        pos = torch.arange(seq_len, device="cuda")
        input_ids[:, (pos % BLOCK_SIZE) == (BLOCK_SIZE - 1)] = vocab - 1
    labels = input_ids.clone()

    def one_step():
        out = model(input_ids, labels=labels, z_loss_multiplier=1e-5)
        loss = out.loss if hasattr(out, "loss") else out[0]
        loss.backward()
        model.zero_grad(set_to_none=True)
        return loss

    for i in range(warmup):
        loss = one_step()
        torch.cuda.synchronize()
        print(f"  warmup {i}: loss={loss.item():.4f}", flush=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    res = dict(
        model=model_name,
        attn=attn,
        ac=ac,
        seq_len=seq_len,
        batch=batch,
        ms_per_step=dt * 1000,
        tokens_per_sec=batch * seq_len / dt,
        peak_mem_gb=peak_gb,
    )
    print(json.dumps(res), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["qwen3_4b:dense", "qwen3_4b:sparse"])
    p.add_argument("--seq-lens", nargs="+", type=int, default=[65536])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--no-ac", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    results = []
    for seq_len in args.seq_lens:
        for cfg in args.configs:
            model_name, attn = cfg.split(":")
            print(f"=== {model_name} {attn} T={seq_len} ===", flush=True)
            try:
                results.append(bench_one(model_name, attn, seq_len, args.warmup, args.steps, args.batch, ac=not args.no_ac))
            except Exception as e:
                print(f"FAILED {cfg} T={seq_len}: {type(e).__name__}: {e}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    # summary table
    print("\nmodel\tattn\tT\tms/step\ttok/s/GPU\tpeakGB")
    for r in results:
        print(f"{r['model']}\t{r['attn']}\t{r['seq_len']}\t{r['ms_per_step']:.0f}\t{r['tokens_per_sec']:.0f}\t{r['peak_mem_gb']:.1f}")


if __name__ == "__main__":
    main()
