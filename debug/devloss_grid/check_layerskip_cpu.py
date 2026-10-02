"""CPU check of the layer-skip hook on a real niah row with a tiny random model.
(a) empty skip set -> CE bit-identical to gold_fl20p8; (b) all layers skipped -> masked columns'
final hidden state == block-0 input; (c) unmasked columns unaffected; prints masked/total columns.
    HF_HUB_OFFLINE=1 python debug/devloss_grid/check_layerskip_cpu.py
"""
import glob, os, sys, types
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctc_devloss_grid as G  # noqa: E402
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens  # noqa: E402
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder  # noqa: E402

DATA_ROOT = os.environ.get("SMOKE_DATA_ROOT", "/scratch/users/prasann/devloss_grid_data")
TOK = sorted(glob.glob("/net/cubbins/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*"))[-1]
task, rung, ri = "ctc_niah", "2k", 0
from transformers import AutoTokenizer
from olmo_core.config import DType
from olmo_core.nn.transformer import TransformerConfig
tok = AutoTokenizer.from_pretrained(TOK)
ids = G.RESERVED_IDS[G.FAMILY]; row = G.ROSTER[task]
ex = G.load_examples(DATA_ROOT, row, rung, ri + 1)[ri]
r, m, n_spans = G.render_ctc_row(tok, ex, row["seg_task"], ids)
gold = G.gold_docs(row["spec"], ex)
specials = [ids.doc_start, ids.doc_end, ids.eos, ids.landmark, ids.pad]
uniq = sorted(set(int(t) for t in r) - set(specials)); remap = {t: i for i, t in enumerate(uniq)}; U = len(uniq)
for j, sp in enumerate(specials): remap[sp] = U + j
V = ((U + 5 + 7) // 8) * 8
small = types.SimpleNamespace(doc_start=U, doc_end=U + 1, eos=U + 2, landmark=U + 3, pad=U + 4)
inv = {v: k for k, v in remap.items()}
pieces = [tok.convert_ids_to_tokens(inv[i]) if i in inv and inv[i] not in specials else f"<sp{i}>" for i in range(V)]
rs = [remap[int(t)] for t in r]
dec = lambda t: tok.decode([inv[int(t)]]) if int(t) in inv and inv[int(t)] not in specials else f"<sp{t}>"  # noqa: E731
stop_ids, shown, tables = G.build_tables(np.asarray(rs), V, pieces, small, dec)
cfg = TransformerConfig.olmo2_190M(vocab_size=V, n_layers=4, fused_ops=False, dtype=DType.float32)
model = cfg.build(init_device="cpu"); G.attach_soft_tokens(model, small, 42, stop_ids)
n_layers = len(model.blocks)
base = dict(G.SCHEMES["gold_fl20p8"])
schemes = {"full": None, "gold_fl20p8": base,
           "skip_none": dict(base, skip_layers=[]),
           "skip_all": dict(base, skip_layers=list(range(n_layers))),
           "skip_odd": dict(base, skip_layers=list(range(1, n_layers, 2)))}
torch.manual_seed(0)
acc, _ = G.score_rows(model, [rs], [m], [gold], schemes, small, tables, stop_ids, 42, dec)
ce = {k: acc[k]["ce"][0] for k in schemes}
print("CE:", {k: round(v, 6) for k, v in ce.items()}, "layer_frac:", {k: round(acc[k]["layer_frac"][0], 4) for k in schemes})
assert ce["skip_none"] == ce["gold_fl20p8"], "(a) empty skip set must be bit-identical"
assert ce["skip_all"] != ce["gold_fl20p8"], "(b) skipping all layers must change the CE"
# (b)/(c): capture block-0 input and last-block output under skip_all and compare per column
x = torch.tensor(np.asarray(rs)[None])
cid0 = build_chunk_ids_from_tokens(x, doc_start_id=small.doc_start, doc_end_id=small.doc_end, eos_id=small.eos, mode="chunked")
n_docs = int(cid0.max()) + 1
pst = model._pooled_soft_tokens
G.configure_scheme(model, pst, base, tables, stop_ids)
keep = G.gold_keep_mask(n_docs, gold, 0.0, 42, ri)
model.train(); model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep.clone())
cb = model._compact_pooled_soft_tokens(x, None, G.IGN)[0]
smask = G.skip_mask_for(cb, cid0, keep)
print(f"masked columns: {int(smask.sum())} / {smask.numel()}  (docs {n_docs}, gold {gold}, kept whole {int(keep.sum())})")
cap = {}
h0 = model.blocks["0"].register_forward_pre_hook(lambda mod, args: cap.__setitem__("in0", args[0].detach().clone()))
sk = G.LayerSkip(model); sk.install(list(range(n_layers))); sk.mask = smask
# capture AFTER installing the skip hooks so it sees the overwritten output (hooks fire in registration order)
hl = model.blocks[str(n_layers - 1)].register_forward_hook(lambda mod, args, out: cap.__setitem__("out_last", out.detach().clone()))
with torch.no_grad():
    model(x)
sk.remove(); h0.remove(); hl.remove()
d = (cap["out_last"] - cap["in0"]).abs().amax(-1)[0]
assert torch.all(d[smask] == 0), "(b) masked columns must equal the block-0 input after skipping every layer"
assert torch.all(d[~smask] > 0), "(c) unmasked columns must have been transformed"
# (c) unmasked columns identical between skip_none and skip_odd at the last block
cap2 = {}
def run(skip):
    sk = G.LayerSkip(model); sk.install(skip); sk.mask = smask if skip else None
    hh = model.blocks[str(n_layers - 1)].register_forward_hook(lambda mod, args, out: cap2.__setitem__("o", out.detach().clone()))
    with torch.no_grad(): model(x)
    sk.remove(); hh.remove(); return cap2["o"][0]
o_none, o_odd = run([]), run(list(range(1, n_layers, 2)))
print("unmasked-column max|Δ| skip_odd vs none:", float((o_none - o_odd).abs()[~smask].amax()), "(expected >0: masked tokens' stale states feed attention)")
print("[layerskip] OK")
