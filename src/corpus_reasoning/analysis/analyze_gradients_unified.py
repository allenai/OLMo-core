"""Per-layer, per-token gradient norm analysis on unified-format data.

Wrapper around the same hidden-state hook approach as analyze_gradients.py, but
takes a unified-format JSONL (documents/queries/answers/...) and a task name,
runs build_prompt to get the alpaca-wrapped prompt, then captures
per-layer / per-token L2 gradient norms with hidden-state backward hooks.

Usage:
    python scripts/analysis/analyze_gradients_unified.py \
        --base-model Qwen/Qwen3.5-0.8B-Base \
        --data data/contradiction_train_pubmed_both_n20_k3.jsonl \
        --task contradiction --query-position both \
        --example-idx 0 \
        --output outputs/grad_contradiction_n20_idx0.json
"""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from corpus_reasoning.lib.chunked_attention import (
    build_chunked_causal_mask,
    setup_tokenizer,
    wrap_documents,
)
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.lib.io import load_jsonl


def prepare(example, task, query_position, tokenizer, max_len, cot_mode="label"):
    prompt, output = build_prompt(
        example,
        task=task,
        query_position=query_position,
        cot_mode=cot_mode,
    )
    wrapped_prompt = wrap_documents(prompt)
    full_text = wrapped_prompt + output + tokenizer.eos_token

    enc = tokenizer(
        full_text, truncation=True, max_length=max_len,
        return_tensors="pt", padding=False,
    )
    input_ids = enc.input_ids.squeeze(0)

    # Mask labels for prompt tokens
    labels = input_ids.clone()
    response_marker = tokenizer.encode("### Response:\n", add_special_tokens=False)
    ids_list = input_ids.tolist()
    response_start = len(ids_list)
    for i in range(len(ids_list) - len(response_marker) + 1):
        if ids_list[i:i + len(response_marker)] == response_marker:
            response_start = i + len(response_marker)
            break
    labels[:response_start] = -100
    return input_ids, labels, response_start


def classify_tokens(input_ids, tokenizer, doc_start_id, doc_end_id, response_start):
    ids = input_ids.tolist()
    types = []
    in_doc = False
    question_tokens = tokenizer.encode("Question:", add_special_tokens=False)

    for i, tid in enumerate(ids):
        if i >= response_start:
            types.append("output")
        elif tid == doc_start_id:
            in_doc = True
            types.append("document")
        elif tid == doc_end_id:
            in_doc = False
            types.append("document")
        elif in_doc:
            types.append("document")
        else:
            if (i + len(question_tokens) <= len(ids)
                    and ids[i:i + len(question_tokens)] == question_tokens):
                types.append("query")
            elif i > 0 and types[-1] == "query":
                types.append("query")
            else:
                types.append("instruction")
    return types


def run(model, input_ids, labels, attention_mask, device):
    model.train()
    model.zero_grad()

    input_ids = input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    grads = {}

    def make_hook(idx):
        def hook(module, gin, gout):
            if gout[0] is not None:
                g = gout[0].detach().float()
                grads[idx] = g.squeeze(0).norm(dim=-1).cpu().tolist()
        return hook

    base = model
    if hasattr(base, "base_model"):
        base = base.base_model
        if hasattr(base, "model"):
            base = base.model
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        layers = base.model.layers
    elif hasattr(base, "layers"):
        layers = base.layers
    else:
        raise ValueError("can't find layers")

    hooks = [layer.register_full_backward_hook(make_hook(i))
             for i, layer in enumerate(layers)]

    out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    out.loss.backward()
    for h in hooks:
        h.remove()
    return grads, out.loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora-path", default="")
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--query-position", default="after")
    ap.add_argument("--cot-mode", default="label")
    ap.add_argument("--example-idx", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--standard-attention", action="store_true",
                    help="If set, no chunked mask (full causal).")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-targets", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    doc_start_id, doc_end_id = setup_tokenizer(tok)

    print(f"Loading {args.base_model} (sdpa, bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tok))

    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        print(f"Loaded LoRA from {args.lora_path}")
    else:
        cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=0.0, target_modules=args.lora_targets,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)

    model = model.cuda()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    device = next(model.parameters()).device

    examples = load_jsonl(args.data)
    print(f"Loaded {len(examples)} examples from {args.data}")

    end = min(args.example_idx + args.max_examples, len(examples))
    out_results = []
    for idx in range(args.example_idx, end):
        ex = examples[idx]
        input_ids, labels, response_start = prepare(
            ex, args.task, args.query_position, tok, args.max_len,
            cot_mode=args.cot_mode,
        )
        if args.standard_attention:
            attn_mask = None  # let HF build the default causal mask
        else:
            attn_mask = build_chunked_causal_mask(input_ids, doc_start_id, doc_end_id)

        types = classify_tokens(input_ids, tok, doc_start_id, doc_end_id, response_start)
        tokens = [tok.decode([t]) for t in input_ids.tolist()]

        grads, loss = run(model, input_ids, labels, attn_mask, device)
        print(f"  ex {idx}: loss={loss:.4f}, n_tokens={len(input_ids)}")
        for ttype in ("instruction", "document", "query", "output"):
            ix = [i for i, t in enumerate(types) if t == ttype]
            if not ix:
                continue
            avg_per_layer = []
            for k, v in grads.items():
                if isinstance(k, int):
                    avg_per_layer.append(sum(v[i] for i in ix) / len(ix))
            if avg_per_layer:
                print(f"    {ttype:11s} ({len(ix):4d} tok)  "
                      f"avg grad norm = {sum(avg_per_layer)/len(avg_per_layer):.6f}")

        out_results.append({
            "example_idx": idx,
            "loss": loss,
            "num_tokens": len(input_ids),
            "tokens": tokens,
            "token_types": types,
            "task": args.task,
            "query_position": args.query_position,
            "attention_type": "standard" if args.standard_attention else "chunked",
            "gradient_norms": {str(k): v for k, v in grads.items()},
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "data": args.data,
            "task": args.task,
            "query_position": args.query_position,
            "results": out_results,
        }, f)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
