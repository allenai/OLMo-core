"""Debug: why does the runtime content fingerprint miss the gold table?

Compares the olmo NumpyPaddedFSLDataset instance-0 token bytes against a fresh
tokenize-time render of JSONL row 0, and checks both fingerprints against the gold
table. Pinpoints any divergence (start/end offset, value mismatch, doc count).
"""
import json

import torch
from transformers import AutoTokenizer

from corpus_reasoning.lib import chunked_attention as _ca
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.lib.olmo_flex_attention import build_roles
from corpus_reasoning.lib.olmo_gold_grad_mask import (
    content_fingerprint, content_fingerprint_from_row)

BASE = "Qwen/Qwen3-0.6B-Base"
PREF = "data/.cache/olmo/contradiction_train_pubmed_both_n20_k3_contradiction_qboth_b7457e555311b081"
DATA = "data/contradiction_train_pubmed_both_n20_k3.jsonl"

meta = json.load(open(PREF + "_meta.json"))
gold = json.load(open(PREF + "_gold.json"))
eos = meta["eos_token_id"]
DS, DE = meta["doc_start_id"], meta["doc_end_id"]
print(f"meta: eos={eos} pad={meta['pad_token_id']} DS={DS} DE={DE} "
      f"wrap={meta.get('wrap_docs')} n_examples={meta.get('n_examples')} "
      f"gold_entries={len(gold)}")

# --- olmo dataset instance 0 ---
from olmo_core.data import NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType

tok_cfg = TokenizerConfig(vocab_size=151936, eos_token_id=eos,
                          pad_token_id=meta["pad_token_id"], bos_token_id=None,
                          identifier=BASE)
dc = NumpyPaddedFSLDatasetConfig(
    paths=[PREF + "_tokens.npy"], label_mask_paths=[PREF + "_label_mask.npy"],
    sequence_length=2048, tokenizer=tok_cfg, dtype=NumpyDatasetDType.uint32,
    work_dir="/tmp/olmo_cache_dbg")
ds = dc.build()
ds.prepare()
inst = ds[0]["input_ids"].tolist()
fe = inst.index(eos)
rt_content = inst[:fe + 1]
rt_fp = content_fingerprint_from_row(inst, eos)
roles = build_roles(torch.tensor([inst]), DS, DE, eos, mode="chunked")[0]
n_docs = int(roles.max().item()) + 1
print(f"\nds[0]: total_len={len(inst)} first_eos={fe} content_len={len(rt_content)} "
      f"n_docs_found={n_docs}")
print(f"  first10={rt_content[:10]}")
print(f"  last10={rt_content[-10:]}")
print(f"  runtime_fp={rt_fp[:16]}...  in_table={rt_fp in gold}")

# --- fresh tokenize of JSONL row 0 ---
tok = AutoTokenizer.from_pretrained(BASE)
_ca.DOC_START, _ca.DOC_END = "<|box_start|>", "<|box_end|>"
ex = json.loads(open(DATA).readline())
prompt, output = build_prompt(ex, task="contradiction", query_position="both",
                              use_titles=True, use_alpaca=True, cot_mode="label")
prompt = _ca.wrap_documents(prompt)
p_ids = tok(prompt, add_special_tokens=False).input_ids
o_ids = tok(output, add_special_tokens=False).input_ids + [eos]
tk_content = p_ids + o_ids
tk_fp = content_fingerprint(tk_content)
print(f"\ntokenize row0: content_len={len(tk_content)} "
      f"(p={len(p_ids)} o={len(o_ids)})")
print(f"  first10={tk_content[:10]}")
print(f"  last10={tk_content[-10:]}")
print(f"  tokenize_fp={tk_fp[:16]}...  in_table={tk_fp in gold}")

# --- compare ---
print(f"\nlen_equal={len(tk_content) == len(rt_content)}")
n = min(len(tk_content), len(rt_content))
diffs = [i for i in range(n) if tk_content[i] != rt_content[i]]
print(f"overlap_diffs={len(diffs)}  first_diffs={diffs[:8]}")
if diffs:
    i = diffs[0]
    print(f"  at pos {i}: tokenize={tk_content[i]} runtime={rt_content[i]}")
    print(f"  tokenize[{i}-3:{i}+4]={tk_content[max(0,i-3):i+4]}")
    print(f"  runtime [{i}-3:{i}+4]={rt_content[max(0,i-3):i+4]}")

# --- hit-rate over ALL instances + internal-eos check (replicates what training feeds,
#     since the collator only pads content) ---
hits = miss = internal_eos = 0
miss_examples = []
for i in range(len(ds)):
    row = ds[i]["input_ids"].tolist()
    fe_i = row.index(eos) if eos in row else len(row) - 1
    n_eos_in_content = row[:fe_i + 1].count(eos)  # >1 => internal eos before terminator
    if n_eos_in_content > 1:
        internal_eos += 1
    fp_i = content_fingerprint_from_row(row, eos)
    if fp_i in gold:
        hits += 1
    else:
        miss += 1
        if len(miss_examples) < 3:
            miss_examples.append((i, fe_i, n_eos_in_content, row[:6], row[fe_i - 4:fe_i + 1]))
print(f"\nALL instances: n={len(ds)} hits={hits} miss={miss} "
      f"internal_eos_in_content={internal_eos}")
for (i, fe_i, ne, head, tail) in miss_examples:
    print(f"  MISS ds[{i}]: first_eos={fe_i} eos_in_content={ne} head={head} tail={tail}")
