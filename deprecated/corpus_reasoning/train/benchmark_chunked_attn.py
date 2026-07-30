"""Micro-benchmark: chunked-attention training throughput, flex vs sdpa.

Runs a real forward+backward over a batch of real training examples using each
backend, reports step time and tokens/sec. Uses the same ChunkedDataset +
build_prompt pipeline as training, so the data format (unified-format JSONL,
alpaca wrapping, doc boundary tokens) is identical to production training.

Usage:
    python scripts/train/benchmark_chunked_attn.py \\
        --config configs/hotpotqa_chunked_qboth_lora.yml \\
        --num-steps 15 --warmup 3
"""

import argparse
import time
import torch
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from corpus_reasoning.lib.chunked_attention import setup_tokenizer
from corpus_reasoning.train.train_chunked_fast import (
    ChunkedDataset, FlexChunkedCollator, SdpaChunkedCollator, build_block_mask,
    verify_mask_structure,
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg, attn_impl, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], attn_implementation=attn_impl, torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if cfg.get("adapter") == "lora":
        lora_config = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    return model.cuda()


def run_backend(cfg, attn_impl, batches, num_steps, warmup, verify=False):
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_tokenizer(tokenizer)

    model = build_model(cfg, attn_impl, tokenizer)
    model.train()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    torch.cuda.synchronize()
    total = 0.0
    total_tokens = 0

    for step in range(warmup + num_steps):
        batch = batches[step % len(batches)]
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

        if attn_impl == "flex_attention":
            chunk_ids = batch.pop("chunk_ids")
            if verify and step == 0:
                verify_mask_structure(chunk_ids)
            batch["attention_mask"] = build_block_mask(chunk_ids)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(**batch)
        out.loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if step >= warmup:
            total += dt
            total_tokens += batch["input_ids"].numel()
        print(f"  [{attn_impl}] step {step}: {dt*1000:.1f} ms")

    del model, optim
    torch.cuda.empty_cache()

    avg = total / num_steps
    tok_s = total_tokens / total
    return avg, tok_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-steps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--micro-batch-size", type=int, default=None)
    ap.add_argument("--backends", default="sdpa,flex")
    ap.add_argument("--data-path", default=None, help="Override datasets[0].path from config")
    ap.add_argument("--base-model", default=None, help="Override base_model from config")
    ap.add_argument("--sequence-len", type=int, default=None, help="Override sequence_len from config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.micro_batch_size is not None:
        cfg["micro_batch_size"] = args.micro_batch_size
    if args.data_path is not None:
        cfg["datasets"][0]["path"] = args.data_path
    if args.base_model is not None:
        cfg["base_model"] = args.base_model
    if args.sequence_len is not None:
        cfg["sequence_len"] = args.sequence_len

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    doc_start_id, doc_end_id = setup_tokenizer(tokenizer)

    data_path = cfg["datasets"][0]["path"]
    dataset = ChunkedDataset(
        data_path, tokenizer, cfg.get("sequence_len", 8192),
        doc_start_id, doc_end_id,
        query_position=cfg.get("query_position", "after"),
        train_on_inputs=cfg.get("train_on_inputs", False),
        task=cfg.get("task", "retrieval"),
        use_titles=not cfg.get("no_titles", False),
        before_dummy=cfg.get("before_dummy", 0),
        after_dummy=cfg.get("after_dummy", 0),
    )
    print(f"Loaded {len(dataset)} examples from {data_path}")

    mbs = cfg.get("micro_batch_size", 1)
    # Pre-fetch raw features once; each backend re-collates with its own collator
    # so both see identical source examples but backend-appropriate batch fields.
    raw_features = [dataset[i] for i in range(args.batches * mbs)]
    batch_features = [raw_features[i*mbs:(i+1)*mbs] for i in range(args.batches)]

    results = {}
    backends = args.backends.split(",")

    if "sdpa" in backends:
        print("\n=== Benchmarking SDPA (dense 4D mask) ===")
        collator = SdpaChunkedCollator(doc_start_id, doc_end_id, tokenizer.pad_token_id)
        batches = [collator(bf) for bf in batch_features]
        avg, tok_s = run_backend(cfg, "sdpa", batches, args.num_steps, args.warmup)
        results["sdpa"] = (avg, tok_s)

    if "flex" in backends:
        print("\n=== Benchmarking FlexAttention (BlockMask) ===")
        collator = FlexChunkedCollator(doc_start_id, doc_end_id, tokenizer.pad_token_id)
        batches = [collator(bf) for bf in batch_features]
        avg, tok_s = run_backend(cfg, "flex_attention", batches, args.num_steps, args.warmup, verify=True)
        results["flex"] = (avg, tok_s)

    print("\n=== RESULTS ===")
    for name, (avg, tok_s) in results.items():
        print(f"  {name:6s}: {avg*1000:7.1f} ms/step   {tok_s:10,.0f} tok/s")
    if "sdpa" in results and "flex" in results:
        speedup = results["sdpa"][0] / results["flex"][0]
        print(f"\n  flex speedup vs sdpa: {speedup:.2f}x")


if __name__ == "__main__":
    main()
