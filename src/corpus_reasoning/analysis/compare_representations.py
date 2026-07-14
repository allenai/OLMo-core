"""Compare hidden representations between two model checkpoints.

Given two checkpoints (e.g. base vs fine-tuned) and a set of test examples,
extracts hidden states at each transformer layer and computes per-layer,
per-token L2 distance. Shows which representations changed the most.

Works with any config: chunked/standard attention, query before/after.

Usage:
    # Compare base model vs LoRA checkpoint:
    python scripts/compare_representations.py \
        --base-model NousResearch/Llama-3.2-1B \
        --lora-path outputs/nq-rag-chunked-qbefore \
        --eval-data data/nq_train_k20_random_2500.jsonl \
        --query-position before \
        --max-examples 5 \
        --output outputs/repr_comparison.json

    # Compare two LoRA checkpoints:
    python scripts/compare_representations.py \
        --base-model NousResearch/Llama-3.2-1B \
        --lora-path outputs/nq-rag-chunked-qbefore \
        --lora-path-2 outputs/nq-rag-chunked-qafter \
        --eval-data data/nq_train_k20_random_2500.jsonl \
        --max-examples 5 \
        --output outputs/repr_comparison.json

    # Standard attention:
    python scripts/compare_representations.py \
        --base-model NousResearch/Llama-3.2-1B \
        --lora-path outputs/nq-rag-std-qafter \
        --eval-data data/nq_train_k20_random_2500.jsonl \
        --standard-attention \
        --max-examples 5 \
        --output outputs/repr_comparison.json
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from corpus_reasoning.lib.io import load_jsonl, ALPACA_TEMPLATE
from corpus_reasoning.lib.chunked_attention import (
    setup_tokenizer, wrap_documents, build_chunked_causal_mask, reorder_query,
)


def load_model_pair(base_model_name, lora_path, lora_path_2, tokenizer):
    """Load two models for comparison. Returns (model_a, model_b, label_a, label_b)."""
    print(f"Loading base model: {base_model_name}")
    model_a = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model_a.resize_token_embeddings(len(tokenizer))

    if lora_path_2:
        # Compare two LoRA checkpoints
        model_b = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        model_b.resize_token_embeddings(len(tokenizer))
        model_a = PeftModel.from_pretrained(model_a, lora_path)
        model_a = model_a.merge_and_unload()
        model_b = PeftModel.from_pretrained(model_b, lora_path_2)
        model_b = model_b.merge_and_unload()
        label_a, label_b = Path(lora_path).name, Path(lora_path_2).name
    elif lora_path:
        # Compare base vs LoRA
        model_b = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        model_b.resize_token_embeddings(len(tokenizer))
        model_b = PeftModel.from_pretrained(model_b, lora_path)
        model_b = model_b.merge_and_unload()
        label_a, label_b = "base", Path(lora_path).name
    else:
        raise ValueError("Must provide --lora-path (and optionally --lora-path-2)")

    model_a = model_a.cuda().eval()
    model_b = model_b.cuda().eval()
    return model_a, model_b, label_a, label_b

def prepare_input(ex, tokenizer, query_position, max_len, doc_start_id, doc_end_id):
    """Prepare an example for inference (prompt only, no output)."""
    input_text = reorder_query(ex["input"], query_position)
    wrapped_input = wrap_documents(input_text)
    prompt = ALPACA_TEMPLATE.format(instruction=ex["instruction"], input=wrapped_input)

    encoding = tokenizer(
        prompt, truncation=True, max_length=max_len,
        return_tensors="pt", padding=False,
    )
    return encoding.input_ids.squeeze(0)


def get_token_types(input_ids, tokenizer, doc_start_id, doc_end_id):
    """Classify tokens by type."""
    ids = input_ids.tolist()
    token_types = []
    in_doc = False

    response_marker = tokenizer.encode("### Response:\n", add_special_tokens=False)
    response_start = len(ids)
    for i in range(len(ids) - len(response_marker) + 1):
        if ids[i:i + len(response_marker)] == response_marker:
            response_start = i + len(response_marker)
            break

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
            token_types.append("instruction")

    # Refine: find "Question:" tokens within instruction
    question_str = "Question:"
    question_toks = tokenizer.encode(question_str, add_special_tokens=False)
    for i in range(len(ids) - len(question_toks) + 1):
        if ids[i:i + len(question_toks)] == question_toks and token_types[i] == "instruction":
            # Mark from here to the next doc or response as query
            for j in range(i, min(response_start, len(ids))):
                if token_types[j] == "document":
                    break
                token_types[j] = "query"

    return token_types


@torch.no_grad()
def extract_hidden_states(model, input_ids, attention_mask, device):
    """Extract hidden states from all layers. Returns dict {layer_idx: (seq_len, hidden_dim)}."""
    input_ids = input_ids.unsqueeze(0).to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # outputs.hidden_states is a tuple of (num_layers + 1) tensors
    # Index 0 = embedding output, index i = output of layer i
    hidden_states = {}
    for i, hs in enumerate(outputs.hidden_states):
        hidden_states[i] = hs.squeeze(0).float().cpu()

    return hidden_states


def compute_distances(hidden_a, hidden_b):
    """Compute per-layer, per-token L2 distances between two sets of hidden states."""
    distances = {}
    for layer_idx in hidden_a:
        if layer_idx not in hidden_b:
            continue
        # (seq_len, hidden_dim)
        diff = hidden_a[layer_idx] - hidden_b[layer_idx]
        # Per-token L2 norm
        distances[layer_idx] = diff.norm(dim=-1).tolist()

    # Also compute cosine distances
    cosine_distances = {}
    for layer_idx in hidden_a:
        if layer_idx not in hidden_b:
            continue
        a = hidden_a[layer_idx]
        b = hidden_b[layer_idx]
        cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        cosine_distances[layer_idx] = (1 - cos_sim).tolist()

    return distances, cosine_distances


def main():
    parser = argparse.ArgumentParser(description="Compare representations between checkpoints")
    parser.add_argument("--base-model", type=str, default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--lora-path", type=str, default="",
                        help="First LoRA checkpoint (compared against base, or against --lora-path-2)")
    parser.add_argument("--lora-path-2", type=str, default="",
                        help="Second LoRA checkpoint (if comparing two fine-tuned models)")
    parser.add_argument("--eval-data", type=str, required=True, help="JSONL eval/train data")
    parser.add_argument("--query-position", type=str, default="after",
                        choices=["before", "after", "both"])
    parser.add_argument("--standard-attention", action="store_true",
                        help="Use standard causal attention instead of chunked")
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--output", type=str, default="outputs/repr_comparison.json")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    doc_start_id, doc_end_id = setup_tokenizer(tokenizer)

    # Load model pair
    model_a, model_b, label_a, label_b = load_model_pair(
        args.base_model, args.lora_path, args.lora_path_2, tokenizer
    )
    device = next(model_a.parameters()).device
    print(f"Comparing: {label_a} vs {label_b}")

    # Load data
    examples = load_jsonl(args.eval_data)
    if len(examples) > args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Analyzing {len(examples)} examples")

    results = []
    for idx, ex in enumerate(tqdm(examples, desc="Comparing representations")):
        input_ids = prepare_input(
            ex, tokenizer, args.query_position, args.max_len,
            doc_start_id, doc_end_id,
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

        # Extract hidden states from both models
        hidden_a = extract_hidden_states(model_a, input_ids, attention_mask, device)
        hidden_b = extract_hidden_states(model_b, input_ids, attention_mask, device)

        # Compute distances
        l2_distances, cosine_distances = compute_distances(hidden_a, hidden_b)

        tokens = [tokenizer.decode([tid]) for tid in input_ids]
        token_types = get_token_types(input_ids, tokenizer, doc_start_id, doc_end_id)

        result = {
            "example_idx": idx,
            "num_tokens": len(input_ids),
            "tokens": tokens,
            "token_types": token_types,
            "query_position": args.query_position,
            "attention_type": "standard" if args.standard_attention else "chunked",
            "l2_distances": {str(k): v for k, v in l2_distances.items()},
            "cosine_distances": {str(k): v for k, v in cosine_distances.items()},
        }
        results.append(result)

        # Print summary
        print(f"\n  Example {idx}: {len(input_ids)} tokens")
        for ttype in ["instruction", "document", "query", "output"]:
            indices = [i for i, t in enumerate(token_types) if t == ttype]
            if not indices:
                continue
            # Average L2 distance across layers
            layer_avgs = []
            for layer_key, dists in l2_distances.items():
                if layer_key == 0:  # skip embedding
                    continue
                avg = sum(dists[i] for i in indices) / len(indices)
                layer_avgs.append(avg)
            if layer_avgs:
                overall = sum(layer_avgs) / len(layer_avgs)
                print(f"    {ttype}: {len(indices)} tokens, avg L2 dist = {overall:.6f}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model_a": label_a,
            "model_b": label_b,
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "lora_path_2": args.lora_path_2,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
