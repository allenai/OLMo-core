"""
Visualize document-chunked / hierarchical-dilated attention masks on a REAL contradiction
training instance, with the mask grid indexed by ACTUAL TOKENS (not one-per-document units).

The full sequence (~5k tokens) is too wide for an ASCII grid, so the token-level views use a
window that spans the prefix->first-documents boundary (where the interesting cross-chunk
structure lives). A compact doc-level overview is still emitted at the end for the big picture.

Run (CPU is fine; no GPU needed):
    PYTHONPATH=$PWD/src python visualizations/mask_token_viz.py
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
OUT = os.path.join(REPO, "visualizations", "mask_token_example_n20.txt")
SEQ = 8192
N_DOCS_IN_WINDOW = 6  # token-level grid spans the prefix tail + this many documents

# --- load one REAL training instance via the exact training loader (custom shard format) ---
tok_cfg = TokenizerConfig.qwen3()
doc_tok = replace(tok_cfg, bos_token_id=None)
cfg = PadToLengthInstanceSourceConfig.from_npy(
    f"{D}/token_ids_part_*.npy", tokenizer=doc_tok, sequence_length=SEQ,
    label_mask_paths=[f"{D}/labels_mask_*.npy"], expand_glob=True)
src = cfg.build("/tmp/mask_viz_wd")
inst = src[0]
ids_full = np.asarray(inst["input_ids"], dtype=np.int64)
lab_full = np.asarray(inst["label_mask"], dtype=bool) if "label_mask" in inst else np.ones_like(ids_full, bool)
# trim right padding (pad == the tokenizer pad; keep up to & incl the first EOS)
eos_pos = np.where(ids_full == EOS)[0]
end = int(eos_pos[0]) + 1 if len(eos_pos) else len(ids_full)
ids = torch.tensor(ids_full[:end])
lab = lab_full[:end]
S = ids.numel()
chunk_ids = build_chunk_ids_from_tokens(ids, DS, DE, EOS, mode="chunked").reshape(-1)

tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
def role(c):
    c = int(c); return {FREE_CHUNK_ID:"FREE", PAD_CHUNK_ID:"PAD", SINK_CHUNK_ID:"SINK"}.get(c, f"doc{c}")
def rc(c):  # 1-char role code for grid headers
    c=int(c); return {FREE_CHUNK_ID:"F", PAD_CHUNK_ID:".", SINK_CHUNK_ID:"S"}.get(c, str(c%10))

ndoc = max([int(c) for c in chunk_ids if int(c) >= 0] + [-1]) + 1
DN, DM = 4, 2  # dilation_n, dilation_m for the "dilated" (hierarchical_dilated) variant
masks = {
  "CHUNKED (each context doc attends ONLY itself + FREE tokens)":
      build_chunked_allowed_mask(AttentionPattern(name="chunked"), chunk_ids)[0],
}
for layer in (0, 1, 2, 3, 4, 5):
    stride = DM ** layer
    masks[f"DILATED n={DN} m={DM} @ LAYER {layer} (stride={stride}: doc i also attends i-{stride}, i-{2*stride}, i-{3*stride})"] = \
        build_chunked_allowed_mask(AttentionPattern(name="hierarchical_dilated", dilation_n=DN, dilation_m=DM), chunk_ids, layer_idx=layer)[0]

L = []
L.append("="*104)
L.append("REAL n=20 document-chunked contradiction training example (contradiction_n20_docdense, loaded via the")
L.append("training PadToLengthInstanceSource; Qwen3 tokenizer). seq_len(real, pre-pad)=%d, #context docs=%d" % (S, ndoc))
L.append("box ids: <|doc_start|>=%d <|doc_end|>=%d eos=%d" % (DS, DE, EOS))
L.append("roles: FREE(-1)=instruction/query/answer (attend everything, attended by everything); docK=context chunk K; PAD")
L.append("="*104)

# ---- token stream (collapse long same-doc runs; always show box markers & role changes) ----
L.append("\n### TOKEN STREAM  (idx  id  role  loss?  decoded) ###")
prev=None; run=0
for i in range(S):
    r=role(chunk_ids[i]); dec=tk.decode([int(ids[i])]); mk=int(ids[i]) in (DS,DE)
    show = mk or r!=prev or i<2 or i>S-2
    if show:
        if run>3: L.append("        ... (%d more %s tokens) ..." % (run-2, prev))
        L.append("  %4d  %6d  %-5s %s  %r" % (i, int(ids[i]), r, ("Y" if lab[i] else "-"), dec))
        run=0
    else: run+=1
    prev=r

# ---- ASCII grid helper: rows/cols are ACTUAL TOKEN indices in `idxs` ----
def grid(mask, idxs, title, show_decoded=False):
    L.append("\n"+title)
    L.append("   key-> "+"".join(rc(chunk_ids[j]) for j in idxs))
    for qi in idxs:
        rowdec = ("  %r" % tk.decode([int(ids[qi])])) if show_decoded else ""
        L.append("  q%4d%s %s%s" % (qi, rc(chunk_ids[qi]), "".join("#" if bool(mask[qi,kj]) else "." for kj in idxs), rowdec))

# ---- TOKEN-LEVEL window: actual tokens spanning the prefix tail + first N documents ----
# find where doc0 starts, back up a few FREE tokens for context, run through N_DOCS_IN_WINDOW docs
doc_starts = [i for i in range(S) if int(chunk_ids[i]) == 0]
first_doc = doc_starts[0] if doc_starts else 0
last_doc_id = min(N_DOCS_IN_WINDOW - 1, ndoc - 1)
after = [i for i in range(S) if int(chunk_ids[i]) > last_doc_id]
win_start = max(0, first_doc - 6)
win_end = after[0] if after else S
sl = list(range(win_start, win_end))
L.append("\n"+"="*104)
L.append("TOKEN-LEVEL MASK  (rows/cols = ACTUAL TOKENS %d..%d: FREE prefix tail + doc0..doc%d, box markers included)" % (win_start, win_end-1, last_doc_id))
L.append("'#'=query(row) may attend key(col); '.'=blocked. Column role codes: F=FREE, 0-9=doc index mod 10.")
L.append("="*104)
grid(masks[list(masks)[0]], sl, ">>> CHUNKED (token level) <<<", show_decoded=True)
grid(masks[[k for k in masks if 'LAYER 0' in k][0]], sl, ">>> DILATED @layer0 stride=1 (token level) <<<")
grid(masks[[k for k in masks if 'LAYER 1' in k][0]], sl, ">>> DILATED @layer1 stride=2 (token level) <<<")

# ---- DOC-LEVEL overview: one representative token per unit (compact big-picture) ----
units=[]; seen=set()
for i in range(S):
    c=int(chunk_ids[i])
    if c not in seen and c!=PAD_CHUNK_ID: seen.add(c); units.append(i)
L.append("\n"+"="*104)
L.append("DOC-LEVEL OVERVIEW  (compact: one representative token per unit -- FREE-prefix, doc0..doc%d, FREE query)" % (ndoc-1))
L.append("rows=query unit, cols=key unit; '#'=may attend, '.'=blocked. (Within a doc it's full-causal.)")
L.append("="*104)
for title, m in masks.items():
    grid(m, units, ">>> "+title+" <<<")

open(OUT,"w").write("\n".join(L))
print("WROTE", OUT, "| seq_len", S, "| ndocs", ndoc, "| token-window", win_start, "..", win_end-1, "| lines", len(L))
