"""
Compact n=5 view: the ENTIRE token-level attention mask on a small REAL 5-document instance,
so the whole grid fits at once. Units are ACTUAL TOKENS.

The real contradiction instances have a ~105-token FREE instruction prefix and a ~118-token
answer, which would make a full-sequence grid ~400+ wide. To keep the whole mask viewable, we
build a small instance from REAL tokens: a short slice of the real FREE prefix + 5 real
(shortest) documents + a short slice of the real answer. Doc content is real; only the length
of the structurally-uniform FREE prefix/answer is trimmed for display.

Run (CPU is fine; no GPU needed):
    PYTHONPATH=$PWD/src python visualizations/mask_token_viz_n5.py
"""
import numpy as np, torch, os
from dataclasses import replace
from transformers import AutoTokenizer
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import PadToLengthInstanceSourceConfig
from olmo_core.nn.attention.chunked_mask import (
    build_chunk_ids_from_tokens, build_chunked_allowed_mask, AttentionPattern,
    FREE_CHUNK_ID, SINK_CHUNK_ID, PAD_CHUNK_ID)

DS, DE, EOS = 151648, 151649, 151643
D = "/scratch/users/prasann/longctx_sft_qwen/contradiction_n20_docdense"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "visualizations", "mask_n5_full.txt")
SEQ = 8192
N_DOCS = 5          # documents in the compact instance
PREFIX_KEEP = 10    # trailing FREE prefix tokens to keep (structurally uniform)
ANSWER_KEEP = 12    # leading FREE answer tokens to keep
PICK_SHORTEST = True  # pick the N_DOCS shortest real docs to minimize grid width

# --- load one REAL training instance via the exact training loader ---
tok_cfg = TokenizerConfig.qwen3()
doc_tok = replace(tok_cfg, bos_token_id=None)
cfg = PadToLengthInstanceSourceConfig.from_npy(
    f"{D}/token_ids_part_*.npy", tokenizer=doc_tok, sequence_length=SEQ,
    label_mask_paths=[f"{D}/labels_mask_*.npy"], expand_glob=True)
inst = cfg.build("/tmp/mask_viz_wd")[0]
ids_full = np.asarray(inst["input_ids"], dtype=np.int64)
eos_pos = np.where(ids_full == EOS)[0]
end = int(eos_pos[0]) + 1 if len(eos_pos) else len(ids_full)
ids_full = ids_full[:end]
cid_full = build_chunk_ids_from_tokens(torch.tensor(ids_full), DS, DE, EOS, mode="chunked").reshape(-1).numpy()

# --- carve out real segments: FREE prefix, per-doc spans, FREE answer ---
n_all = int(cid_full.max()) + 1
doc_spans = {d: np.where(cid_full == d)[0] for d in range(n_all)}
first_doc_start = doc_spans[0][0]
prefix_idx = [i for i in range(first_doc_start) if cid_full[i] == FREE_CHUNK_ID][-PREFIX_KEEP:]
last_ctx = max(i for i in range(len(cid_full)) if cid_full[i] >= 0)
answer_idx = [i for i in range(last_ctx + 1, len(cid_full)) if cid_full[i] == FREE_CHUNK_ID][:ANSWER_KEEP]

order = sorted(range(n_all), key=lambda d: len(doc_spans[d]))[:N_DOCS] if PICK_SHORTEST else list(range(N_DOCS))
chosen_docs = sorted(order)  # keep ascending index order

# --- stitch a small instance of REAL tokens: prefix + 5 docs + answer ---
new_idx = list(prefix_idx) + [i for d in chosen_docs for i in doc_spans[d]] + list(answer_idx)
ids = torch.tensor(ids_full[new_idx])
S = ids.numel()
chunk_ids = build_chunk_ids_from_tokens(ids, DS, DE, EOS, mode="chunked").reshape(-1)
ndoc = int(chunk_ids.max()) + 1

tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
def role(c):
    c = int(c); return {FREE_CHUNK_ID:"FREE", PAD_CHUNK_ID:"PAD", SINK_CHUNK_ID:"SINK"}.get(c, f"doc{c}")
def rc(c):
    c=int(c); return {FREE_CHUNK_ID:"F", PAD_CHUNK_ID:".", SINK_CHUNK_ID:"S"}.get(c, str(c%10))

DN, DM = 4, 2
masks = {"CHUNKED (each context doc attends ONLY itself + FREE tokens)":
         build_chunked_allowed_mask(AttentionPattern(name="chunked"), chunk_ids)[0]}
for layer in (0, 1, 2):
    stride = DM ** layer
    masks[f"DILATED n={DN} m={DM} @ LAYER {layer} (stride={stride}: doc i also attends i-{stride}, i-{2*stride}, i-{3*stride})"] = \
        build_chunked_allowed_mask(AttentionPattern(name="hierarchical_dilated", dilation_n=DN, dilation_m=DM), chunk_ids, layer_idx=layer)[0]

L = []
L.append("="*100)
L.append("COMPACT n=5 mask on REAL contradiction tokens (from contradiction_n20_docdense).")
L.append("Built from real segments: %d FREE prefix + %d real docs + %d FREE answer = %d tokens." % (len(prefix_idx), N_DOCS, len(answer_idx), S))
L.append("Docs shown (real, %s): %s %d of %d docs, relabeled doc0..doc%d in order. FREE prefix/answer trimmed for width."
         % ("by length" if PICK_SHORTEST else "first", "shortest" if PICK_SHORTEST else "first", N_DOCS, n_all, N_DOCS-1))
L.append("roles: FREE(-1)=instruction/query/answer (attend & attended by everything); docK=context chunk K")
L.append("="*100)

# token stream
L.append("\n### TOKEN STREAM  (idx  id  role  decoded) ###")
for i in range(S):
    L.append("  %3d  %6d  %-5s  %r" % (i, int(ids[i]), role(chunk_ids[i]), tk.decode([int(ids[i])])))

def grid(mask, idxs, title, show_decoded=True):
    L.append("\n"+title)
    L.append("   key-> "+"".join(rc(chunk_ids[j]) for j in idxs))
    for qi in idxs:
        rowdec = ("  %r" % tk.decode([int(ids[qi])])) if show_decoded else ""
        L.append("  q%3d%s %s%s" % (qi, rc(chunk_ids[qi]), "".join("#" if bool(mask[qi,kj]) else "." for kj in idxs), rowdec))

allidx = list(range(S))
L.append("\n"+"="*100)
L.append("FULL TOKEN-LEVEL MASK  (rows/cols = ALL %d actual tokens; '#'=may attend, '.'=blocked)" % S)
L.append("column role codes: F=FREE, 0-9=doc index mod 10")
L.append("="*100)
for title, m in masks.items():
    grid(m, allidx, ">>> "+title+" <<<")

open(OUT, "w").write("\n".join(L))
print("WROTE", OUT, "| tokens", S, "| ndocs", ndoc, "| grid width", S)
