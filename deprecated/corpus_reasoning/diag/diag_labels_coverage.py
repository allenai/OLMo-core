"""Check label coverage across the contradiction + hpqa training datasets.

For each example in each training file, tokenize it through the same path as
ChunkedDataset (sequence_len=16384) and count:
  - total tokens
  - tokens with labels != -100
  - whether the "### Response:\\n" marker was found

If many examples have 0 labels kept, that explains the plateau: they contribute
zero gradient and the model only learns from the subset whose answer survives
truncation.
"""

import os, sys
# sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
from transformers import AutoTokenizer
from corpus_reasoning.lib.chunked_attention import setup_tokenizer
from corpus_reasoning.train.train_chunked_fast import ChunkedDataset


DATASETS = [
    ("contradiction", "data/contradiction_train_pubmed_both_n100_k3.jsonl"),
    ("hpqa",          "data/hotpotqa_train_k100_bridge_hn98_2500.jsonl"),
]

TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
if TOK.pad_token is None:
    TOK.pad_token = TOK.eos_token
doc_start_id, doc_end_id = setup_tokenizer(TOK)

for name, path in DATASETS:
    task = "contradiction" if name == "contradiction" else "qa"
    ds = ChunkedDataset(
        path, TOK, max_len=16384,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        query_position="both", train_on_inputs=False, task=task,
    )
    n = len(ds)
    # Sample up to 200 examples for speed (random-stride).
    stride = max(1, n // 200)
    kept = []
    truncated = 0
    full = 0
    total_lens = []
    for i in range(0, n, stride):
        ex = ds[i]
        t = ex["input_ids"].numel()
        k = int((ex["labels"] != -100).sum().item())
        kept.append(k)
        total_lens.append(t)
        if k == 0:
            truncated += 1
        else:
            full += 1
    print(f"=== {name} ({n} examples, sampled {len(kept)}) ===")
    print(f"  avg tokens: {sum(total_lens)/len(total_lens):.0f}  "
          f"max tokens: {max(total_lens)}  min: {min(total_lens)}")
    print(f"  labels_kept: avg={sum(kept)/len(kept):.1f}  "
          f"zero_kept (truncated): {truncated}/{len(kept)}  "
          f"non_zero_kept: {full}/{len(kept)}")
    # Show distribution of kept counts for the non-zero examples.
    nz = sorted([k for k in kept if k > 0])
    if nz:
        print(f"  non-zero labels_kept percentiles: "
              f"p10={nz[len(nz)//10]}  p50={nz[len(nz)//2]}  "
              f"p90={nz[min(len(nz)-1, 9*len(nz)//10)]}  "
              f"max={nz[-1]}")
    print()
