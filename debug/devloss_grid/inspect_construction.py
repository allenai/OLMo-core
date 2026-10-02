"""Decode what the model actually SEES under a scheme, on real rows, on CPU (tiny random model).
    HF_HUB_OFFLINE=1 python debug/devloss_grid/inspect_construction.py ctc_niah 2k gold_fl20,fl20 0
"""
import glob, os, sys, types
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctc_devloss_grid as G  # noqa: E402
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens  # noqa: E402
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder, resolve_keep_docs  # noqa: E402

DATA_ROOT = os.environ.get("SMOKE_DATA_ROOT", "/scratch/users/prasann/devloss_grid_data")
TOK = sorted(glob.glob("/net/cubbins/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*"))[-1]
task, rung, schemes, ri = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4])

from transformers import AutoTokenizer
from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig
tok = AutoTokenizer.from_pretrained(TOK)
ids = G.RESERVED_IDS[G.FAMILY]
row = G.ROSTER[task]
ex = G.load_examples(DATA_ROOT, row, rung, ri + 1)[ri]
r, m, n_spans = G.render_ctc_row(tok, ex, row["seg_task"], ids)
gold = G.gold_docs(row["spec"], ex)
print(f"[inspect] {task}@{rung} row {ri}: {len(r)} toks, {n_spans} spans / {len(ex['documents'])} docs, gold={gold}, answer={tok.decode([t for t, mm in zip(r, m) if mm])!r}")
specials = [ids.doc_start, ids.doc_end, ids.eos, ids.landmark, ids.pad]
uniq = sorted(set(int(t) for t in r) - set(specials)); remap = {t: i for i, t in enumerate(uniq)}; U = len(uniq)
for j, s in enumerate(specials): remap[s] = U + j
V = ((U + 5 + 7) // 8) * 8
small = types.SimpleNamespace(doc_start=U, doc_end=U + 1, eos=U + 2, landmark=U + 3, pad=U + 4)
inv = {v: k for k, v in remap.items()}
pieces = [tok.convert_ids_to_tokens(inv[i]) if i in inv and inv[i] not in specials else f"<sp{i}>" for i in range(V)]
rs = [remap[int(t)] for t in r]
def dec(t):
    t = int(t)
    if t == small.landmark: return " ▮"
    if t == small.doc_start: return " ⟨"
    if t == small.doc_end: return "⟩"
    return tok.decode([inv[t]]) if t in inv else f"<sp{t}>"
stop_ids, shown, tables = G.build_tables(np.asarray(rs), V, pieces, small, dec)
cfg = TransformerConfig.olmo2_190M(vocab_size=V, n_layers=1, fused_ops=False, dtype=DType.float32)
model = cfg.build(init_device="cpu"); G.attach_soft_tokens(model, small, 42, stop_ids)
pst = model._pooled_soft_tokens
x = torch.tensor(np.asarray(rs)[None])
cid0 = build_chunk_ids_from_tokens(x, doc_start_id=small.doc_start, doc_end_id=small.doc_end, eos_id=small.eos, mode="chunked")
n_docs = int(cid0.max()) + 1
for name in schemes:
    sch = G.SCHEMES[name]
    G.configure_scheme(model, pst, sch, tables, stop_ids)
    if sch.get("rule") == "custom":
        ans_pos = torch.tensor(np.nonzero(np.asarray(m))[0]); pred_pos = ans_pos - 1; targets = x[0, ans_pos]
        cap = G.AttnCapture(model) if sch["sel"] == "attnrow" else None
        mask, warn = G.custom_keep_mask(model, x, cid0, n_docs, small, sch["sel"], float(sch["k"]), pred_pos, targets, cap)
        pst["keep_token_mask"] = mask
        print(f"[inspect] {name}: custom mask keeps {int(mask.sum())} body tokens" + (f"; WARNING {warn}" if warn else ""))
    if sch["keep"] == "blind":
        keep = resolve_keep_docs(cid0, n_docs, holder=None, keep_prob=float(sch["keep_prob"]), keep_seed=42).cpu()
    else:
        keep = G.gold_keep_mask(n_docs, gold, float(sch.get("frac", 0.0)), 42, ri)
    model.train(); model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep.clone())
    cb = model._compact_pooled_soft_tokens(x, None, G.IGN)[0]
    seq = cb.input_ids[0].tolist()
    text = "".join(dec(t) for t in seq)
    print(f"\n===== {name}: {len(seq)}/{len(rs)} = {len(seq)/len(rs):.3f}; kept docs {keep.sum().item()}/{n_docs}")
    if len(text) <= 3000:
        print(text)
    else:  # long row: head, a window around every kept (real) document, and the tail (question + answer)
        print(text[:1200]); print("   ...")
        import re as _re
        for k, mm in enumerate(_re.finditer(r" ⟨", text)):
            if k >= 6: break
            a = max(0, mm.start() - 220); print(text[a:mm.start() + 260].replace("\n", " ")); print("   ...")
        print(text[-700:])
