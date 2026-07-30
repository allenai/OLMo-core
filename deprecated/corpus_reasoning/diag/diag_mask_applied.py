"""Diagnostic: is the chunked 4D attention_mask actually being applied?

Loads one contradiction example, runs the base Qwen3.5 model forward under three
masks:
  A) chunked (SdpaChunkedCollator output)
  B) plain causal (triangular, -inf on upper tri)
  C) "all attend, same causal" (identical to B but built from the SdpaChunkedCollator
      with the "standard" pattern)

Then compares logit outputs token-by-token. If A ≈ B, the model is IGNORING the
custom 4D mask.
"""

import os, sys
# sys.path hack removed: corpus_reasoning is a package on PYTHONPATH=src
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from corpus_reasoning.lib.chunked_attention import (
    setup_tokenizer, AttentionPattern, PAD_CHUNK_ID, FREE_CHUNK_ID,
    build_dense_bool_mask,
)
from corpus_reasoning.train.train_chunked_fast import (
    ChunkedDataset, SdpaChunkedCollator, build_chunk_ids,
)


MODEL = "Qwen/Qwen3.5-0.8B-Base"
DATA = "data/contradiction_train_pubmed_both_n100_k3.jsonl"

torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
doc_start_id, doc_end_id = setup_tokenizer(tok)

print("Loading model (bf16, sdpa) ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
).cuda().eval()
model.resize_token_embeddings(len(tok))

pattern_chunked = AttentionPattern(name="chunked")
pattern_standard = AttentionPattern(name="standard")

ds = ChunkedDataset(
    DATA, tok, max_len=16384,
    doc_start_id=doc_start_id, doc_end_id=doc_end_id,
    query_position="both", train_on_inputs=False, task="contradiction",
)
# Use a fresh-bright-line example with a known contradiction.
ex = ds[0]
print(f"seq_len = {ex['input_ids'].size(0)}, labels_kept = {(ex['labels'] != -100).sum().item()}")
label_pos_dbg = (ex['labels'] != -100).nonzero().flatten().tolist()
if label_pos_dbg:
    print(f"First label positions: {label_pos_dbg[:10]} ... last: {label_pos_dbg[-5:]}")
    ids_list = ex['input_ids'].tolist()
    print(f"Learned text: {tok.decode([ids_list[p] for p in label_pos_dbg])!r}")

dtype = torch.bfloat16
min_val = torch.finfo(dtype).min

chunked_collator = SdpaChunkedCollator(doc_start_id, doc_end_id, tok.pad_token_id, pattern_chunked)
batch_chunked = chunked_collator([ex])
std_collator = SdpaChunkedCollator(doc_start_id, doc_end_id, tok.pad_token_id, pattern_standard)
batch_std = std_collator([ex])

assert torch.equal(batch_chunked["input_ids"], batch_std["input_ids"]), "ids differ"
ids = batch_chunked["input_ids"].cuda()
labs = batch_chunked["labels"].cuda()
mask_chunked = batch_chunked["attention_mask"].cuda()
mask_causal  = batch_std["attention_mask"].cuda()

# Sanity: diffs in the masks
print(f"chunked mask 0-entries: {(mask_chunked == 0.0).sum().item()}, "
      f"causal mask 0-entries: {(mask_causal == 0.0).sum().item()}")

def forward(attn_mask):
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn_mask)
    return out.logits.float()

print("Running forward A: chunked mask ...")
logits_A = forward(mask_chunked)
print("Running forward B: causal mask ...")
logits_B = forward(mask_causal)
print("Running forward C: NO explicit mask (pass 2D all-ones) ...")
# This mimics raw generation — HF will build a default causal mask internally.
attn2d = torch.ones_like(ids, dtype=torch.long)
logits_C = forward(attn2d)

diff_AB = (logits_A - logits_B).abs()
diff_AC = (logits_A - logits_C).abs()
diff_BC = (logits_B - logits_C).abs()

print(f"\nLogit diffs (max / mean over all B,S,V):")
print(f"  chunked vs causal-4D:  max={diff_AB.max().item():.4f}  mean={diff_AB.mean().item():.6f}")
print(f"  chunked vs 2D-default: max={diff_AC.max().item():.4f}  mean={diff_AC.mean().item():.6f}")
print(f"  causal-4D vs 2D-def  : max={diff_BC.max().item():.4f}  mean={diff_BC.mean().item():.6f}")

# Specifically look at logit diff restricted to LABEL positions (where loss is
# computed). If the mask is respected, FREE (label) tokens under chunked vs
# causal-4D should still differ (because context ↔ context attention differs,
# which propagates up into the FREE representations).
label_pos = (labs != -100)
label_pos_flat = label_pos[0]  # (S,)
if label_pos_flat.any():
    print(f"\nRestricted to {label_pos_flat.sum().item()} label positions:")
    A_lab = logits_A[0, label_pos_flat]
    B_lab = logits_B[0, label_pos_flat]
    C_lab = logits_C[0, label_pos_flat]
    print(f"  chunked vs causal-4D:  max={(A_lab-B_lab).abs().max().item():.4f}  mean={(A_lab-B_lab).abs().mean().item():.6f}")
    print(f"  chunked vs 2D-default: max={(A_lab-C_lab).abs().max().item():.4f}  mean={(A_lab-C_lab).abs().mean().item():.6f}")

# Also compute the loss under each mask to simulate the trainer.
def loss_under(attn_mask):
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn_mask, labels=labs)
    return out.loss.item()

print(f"\nLoss (base model, no LoRA):")
print(f"  chunked mask: {loss_under(mask_chunked):.4f}")
print(f"  causal-4D:    {loss_under(mask_causal):.4f}")
print(f"  2D-default:   {loss_under(attn2d):.4f}")
