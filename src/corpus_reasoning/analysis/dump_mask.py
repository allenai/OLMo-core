"""Dump an eval-time attention mask and input for a HELMET NQ example."""

import torch

from transformers import AutoTokenizer
from corpus_reasoning.lib.chunked_attention import setup_tokenizer, wrap_documents, build_chunked_causal_mask, find_chunk_spans
from corpus_reasoning.eval.evaluate import load_helmet_examples

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
doc_start_id, doc_end_id = setup_tokenizer(tokenizer)

# Load exactly as evaluate.py does for chunked backends
raw_examples = load_helmet_examples("nq", num_docs=20, max_samples=5, shots=2, wrap_docs=True)
ex = raw_examples[0]
prompt = ex["prompt"]

# Tokenize exactly as evaluate_chunked.py does
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.squeeze(0)
ids_list = input_ids.tolist()
spans = find_chunk_spans(input_ids, doc_start_id, doc_end_id)
seq_len = len(ids_list)

print(f"Tokens: {seq_len}, Document chunks: {len(spans)}")
print(f"Answers: {ex['answers']}")

# --- Write input file ---
with open("outputs/debug_eval_input.txt", "w") as f:
    f.write("=" * 100 + "\n")
    f.write("EVAL PROMPT (full text as sent to tokenizer)\n")
    f.write(f"Answers: {ex['answers']}\n")
    f.write("=" * 100 + "\n\n")
    f.write(prompt)
    f.write("\n\n")

    f.write("=" * 100 + "\n")
    f.write(f"TOKEN-BY-TOKEN ({seq_len} tokens)\n")
    f.write(f"Document chunks: {len(spans)}\n")
    for i, (s, e) in enumerate(spans):
        f.write(f"  doc{i}: tokens [{s}, {e})  length={e-s}\n")
    f.write("=" * 100 + "\n\n")

    for i, tid in enumerate(ids_list):
        tok_str = tokenizer.decode([tid]).replace("\n", "\\n")
        region = ""
        for idx, (s, e) in enumerate(spans):
            if s <= i < e:
                region = f"doc{idx}"
                break
        f.write(f"  [{i:>5}] id={tid:>6}  region={region:<10}  token=|{tok_str}|\n")

print(f"Input written to outputs/debug_eval_input.txt")

# --- Build mask ---
mask = build_chunked_causal_mask(input_ids, doc_start_id, doc_end_id)
bool_mask = (mask.squeeze() == 0)

with open("outputs/debug_eval_mask.txt", "w") as f:
    f.write(f"Attention mask: {seq_len} x {seq_len}\n")
    f.write(f"# = can attend, . = masked\n")
    f.write(f"Document spans: {spans}\n\n")

    def get_region(pos):
        for idx, (s, e) in enumerate(spans):
            if s <= pos < e:
                return f"d{idx}"
        return "qry"

    # --- Sampled overview of full mask ---
    sample_cols = sorted(set(
        list(range(0, seq_len, 50)) +
        [s for s, e in spans] + [e-1 for s, e in spans] +
        [seq_len - 1]
    ))
    sample_cols = [c for c in sample_cols if c < seq_len]

    sample_rows = sorted(set(
        list(range(0, seq_len, 50)) +
        [s for s, e in spans] + [e-1 for s, e in spans] +
        [(s+e)//2 for s, e in spans] +
        [seq_len - 1]
    ))
    sample_rows = [r for r in sample_rows if r < seq_len]

    f.write("SAMPLED MASK (every 50th token + all doc boundaries)\n")
    f.write(f"{'':>18} |")
    for c in sample_cols:
        f.write(f"{c:>5}")
    f.write("\n")
    f.write(f"{'':>18} |")
    for c in sample_cols:
        f.write(f"{'---':>5}")
    f.write("\n")

    for r in sample_rows:
        rgn = get_region(r)
        f.write(f"  {r:>5} ({rgn:>4})     |")
        for c in sample_cols:
            if c > r:
                f.write("     ")
            else:
                val = bool_mask[r, c].item()
                f.write("    #" if val else "    .")
        f.write("\n")

    # --- Dense mask around first two doc boundaries ---
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("DENSE MASK: around doc0/doc1 boundary\n")
    f.write("Each char = one token. # = attend, . = masked\n")
    f.write("=" * 80 + "\n\n")

    window_start = max(0, spans[0][0] - 3)
    window_end = min(seq_len, spans[1][1] + 3)

    # Column header
    f.write(f"{'':>18} |")
    for c in range(window_start, window_end):
        if c % 10 == 0:
            f.write(str((c // 10) % 10))
        else:
            f.write(" ")
    f.write("\n")
    f.write(f"{'':>18} |")
    for c in range(window_start, window_end):
        f.write(str(c % 10))
    f.write("\n")
    f.write(f"{'':>18} |")
    f.write("-" * (window_end - window_start))
    f.write("\n")

    for r in range(window_start, window_end):
        rgn = get_region(r)
        f.write(f"  {r:>5} ({rgn:>4})     |")
        for c in range(window_start, window_end):
            if c > r:
                f.write(" ")
            else:
                val = bool_mask[r, c].item()
                f.write("#" if val else ".")
        f.write("\n")

    # --- Dense mask at the end: last doc + query tokens ---
    f.write("\n\n" + "=" * 80 + "\n")
    last_s, last_e = spans[-1]
    f.write(f"DENSE MASK: last doc (doc{len(spans)-1}) + query tokens at end\n")
    f.write("Each char = one token. # = attend, . = masked\n")
    f.write("=" * 80 + "\n\n")

    window_start2 = max(0, last_s - 3)
    window_end2 = seq_len

    f.write(f"{'':>18} |")
    for c in range(window_start2, window_end2):
        if c % 10 == 0:
            f.write(str((c // 10) % 10))
        else:
            f.write(" ")
    f.write("\n")
    f.write(f"{'':>18} |")
    for c in range(window_start2, window_end2):
        f.write(str(c % 10))
    f.write("\n")
    f.write(f"{'':>18} |")
    f.write("-" * (window_end2 - window_start2))
    f.write("\n")

    for r in range(window_start2, window_end2):
        rgn = get_region(r)
        # For query tokens, show whether they attend to sampled earlier positions
        f.write(f"  {r:>5} ({rgn:>4})     |")
        for c in range(window_start2, window_end2):
            if c > r:
                f.write(" ")
            else:
                val = bool_mask[r, c].item()
                f.write("#" if val else ".")
        f.write("\n")

print(f"Mask written to outputs/debug_eval_mask.txt")
print(f"\nView with:")
print(f"  less outputs/debug_eval_input.txt")
print(f"  less outputs/debug_eval_mask.txt")
