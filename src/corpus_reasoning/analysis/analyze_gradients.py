"""Gradient analysis: which tokens/layers are most affected by a training step.

Given a model and a training example, computes the gradient of the loss w.r.t.
hidden states at each transformer layer. Outputs per-layer, per-token gradient
magnitudes as a JSON file for visualization.

Works with any config: chunked/standard attention, query before/after.

Usage:
    python scripts/analyze_gradients.py \
        --config configs/nq_rag_chunked_qbefore.yml \
        --example-idx 0 \
        --output outputs/gradient_analysis.json

    # With a LoRA checkpoint:
    python scripts/analyze_gradients.py \
        --config configs/nq_rag_chunked_qbefore.yml \
        --lora-path outputs/nq-rag-chunked-qbefore \
        --example-idx 0 \
        --output outputs/gradient_analysis.json

    # Standard attention (no chunked mask):
    python scripts/analyze_gradients.py \
        --config configs/nq_rag_std_qafter.yml \
        --standard-attention \
        --example-idx 0 \
        --output outputs/gradient_analysis.json
"""

import argparse
import json
import yaml
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from corpus_reasoning.lib.io import load_jsonl, ALPACA_TEMPLATE
from corpus_reasoning.lib.chunked_attention import (
    setup_tokenizer, wrap_documents, build_chunked_causal_mask, reorder_query,
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_example(ex, tokenizer, query_position, max_len, doc_start_id, doc_end_id):
    """Prepare a single training example: tokenize, build labels and mask info."""
    input_text = reorder_query(ex["input"], query_position)
    wrapped_input = wrap_documents(input_text)
    prompt = ALPACA_TEMPLATE.format(instruction=ex["instruction"], input=wrapped_input)
    full_text = prompt + ex["output"] + tokenizer.eos_token

    encoding = tokenizer(
        full_text, truncation=True, max_length=max_len,
        return_tensors="pt", padding=False,
    )
    input_ids = encoding.input_ids.squeeze(0)

    # Labels: -100 for prompt tokens, actual IDs for output
    labels = input_ids.clone()
    response_marker = tokenizer.encode("### Response:\n", add_special_tokens=False)
    ids_list = input_ids.tolist()
    response_start = len(ids_list)
    for i in range(len(ids_list) - len(response_marker) + 1):
        if ids_list[i:i + len(response_marker)] == response_marker:
            response_start = i + len(response_marker)
            break
    labels[:response_start] = -100

    return input_ids, labels


def get_token_labels(input_ids, tokenizer, doc_start_id, doc_end_id):
    """Classify each token as 'instruction', 'document', 'query', or 'output'."""
    ids = input_ids.tolist()
    token_types = []
    in_doc = False
    response_marker = tokenizer.encode("### Response:\n", add_special_tokens=False)
    response_start = len(ids)
    for i in range(len(ids) - len(response_marker) + 1):
        if ids[i:i + len(response_marker)] == response_marker:
            response_start = i + len(response_marker)
            break

    question_tokens = tokenizer.encode("Question:", add_special_tokens=False)

    for i, tid in enumerate(ids):
        if i >= response_start:
            token_types.append("output")
        elif tid == doc_start_id:
            in_doc = True
            token_types.append("document")
        elif tid == doc_end_id:
            in_doc = False
            token_types.append("document")
        elif in_doc:
            token_types.append("document")
        else:
            # Check if this is part of a question
            if i + len(question_tokens) <= len(ids) and ids[i:i + len(question_tokens)] == question_tokens:
                token_types.append("query")
            elif i > 0 and token_types[-1] == "query":
                # Continue query until next section
                token_types.append("query")
            else:
                token_types.append("instruction")

    return token_types


def run_gradient_analysis(model, input_ids, labels, attention_mask, device):
    """Run forward+backward and collect per-layer, per-token gradient magnitudes."""
    model.train()
    model.zero_grad()

    input_ids = input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Register hooks to capture hidden state gradients
    hidden_grads = {}

    def make_hook(layer_idx):
        def hook_fn(module, grad_input, grad_output):
            # grad_output is tuple; first element is gradient of output hidden states
            if grad_output[0] is not None:
                # (batch, seq_len, hidden_dim) -> per-token L2 norm
                grad = grad_output[0].detach().float()
                hidden_grads[layer_idx] = grad.squeeze(0).norm(dim=-1).cpu().tolist()
        return hook_fn

    hooks = []
    # Find transformer layers
    if hasattr(model, 'base_model'):
        # PEFT model
        base = model.base_model
        if hasattr(base, 'model'):
            base = base.model
    else:
        base = model

    if hasattr(base, 'model') and hasattr(base.model, 'layers'):
        layers = base.model.layers
    elif hasattr(base, 'layers'):
        layers = base.layers
    else:
        raise ValueError("Cannot find transformer layers in model")

    for i, layer in enumerate(layers):
        hooks.append(layer.register_full_backward_hook(make_hook(i)))

    # Also capture embedding gradient
    def emb_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            grad = grad_output[0].detach().float()
            hidden_grads["embedding"] = grad.squeeze(0).norm(dim=-1).cpu().tolist()

    if hasattr(base, 'model') and hasattr(base.model, 'embed_tokens'):
        hooks.append(base.model.embed_tokens.register_full_backward_hook(emb_hook))
    elif hasattr(base, 'embed_tokens'):
        hooks.append(base.embed_tokens.register_full_backward_hook(emb_hook))

    # Forward + backward
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    loss = outputs.loss
    loss.backward()

    # Clean up hooks
    for h in hooks:
        h.remove()

    return hidden_grads, loss.item()


def main():
    parser = argparse.ArgumentParser(description="Gradient analysis per layer/token")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--lora-path", type=str, default="", help="LoRA checkpoint path")
    parser.add_argument("--example-idx", type=int, default=0, help="Index of training example")
    parser.add_argument("--standard-attention", action="store_true",
                        help="Use standard causal attention instead of chunked")
    parser.add_argument("--output", type=str, default="outputs/gradient_analysis.json")
    parser.add_argument("--max-examples", type=int, default=1,
                        help="Number of examples to analyze")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = cfg["datasets"][0]["path"]
    query_position = cfg.get("query_position", "after")
    max_len = cfg.get("sequence_len", 8192)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    doc_start_id, doc_end_id = setup_tokenizer(tokenizer)

    # Load model
    print(f"Loading {cfg['base_model']} with attn_implementation=sdpa")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        print(f"Loaded LoRA from {args.lora_path}")
    else:
        # Apply fresh LoRA so gradients flow through LoRA params
        lora_config = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=0.0,  # no dropout for analysis
            target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model = model.cuda()
    device = next(model.parameters()).device

    # Disable gradient checkpointing for hook compatibility
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()

    # Load data
    examples = load_jsonl(data_path)
    print(f"Loaded {len(examples)} examples from {data_path}")

    results = []
    end_idx = min(args.example_idx + args.max_examples, len(examples))
    for idx in range(args.example_idx, end_idx):
        ex = examples[idx]
        print(f"\nAnalyzing example {idx}...")
        input_ids, labels = prepare_example(
            ex, tokenizer, query_position, max_len, doc_start_id, doc_end_id
        )

        # Build attention mask
        if args.standard_attention:
            seq_len = len(input_ids)
            dtype = torch.bfloat16
            min_val = torch.finfo(dtype).min
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), min_val, dtype=dtype), diagonal=1
            ).unsqueeze(0).unsqueeze(0)
        else:
            attention_mask = build_chunked_causal_mask(
                input_ids, doc_start_id, doc_end_id,
            )

        # Decode tokens for display
        tokens = [tokenizer.decode([tid]) for tid in input_ids]
        token_types = get_token_labels(input_ids, tokenizer, doc_start_id, doc_end_id)

        # Run gradient analysis
        hidden_grads, loss = run_gradient_analysis(
            model, input_ids, labels, attention_mask, device
        )

        print(f"  Loss: {loss:.4f}, Tokens: {len(input_ids)}")
        print(f"  Layers with gradients: {sorted(k for k in hidden_grads if k != 'embedding')}")

        result = {
            "example_idx": idx,
            "loss": loss,
            "num_tokens": len(input_ids),
            "tokens": tokens,
            "token_types": token_types,
            "query_position": query_position,
            "attention_type": "standard" if args.standard_attention else "chunked",
            "gradient_norms": {},
        }

        # Convert layer indices to strings for JSON
        for layer_key, norms in hidden_grads.items():
            result["gradient_norms"][str(layer_key)] = norms

        results.append(result)

        # Print summary per token type
        for ttype in ["instruction", "document", "query", "output"]:
            indices = [i for i, t in enumerate(token_types) if t == ttype]
            if not indices:
                continue
            # Average gradient norm across layers for these tokens
            layer_avgs = []
            for layer_key, norms in hidden_grads.items():
                if layer_key == "embedding":
                    continue
                avg = sum(norms[i] for i in indices) / len(indices)
                layer_avgs.append(avg)
            if layer_avgs:
                overall_avg = sum(layer_avgs) / len(layer_avgs)
                print(f"  {ttype}: {len(indices)} tokens, avg grad norm = {overall_avg:.6f}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": args.config,
            "lora_path": args.lora_path,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
