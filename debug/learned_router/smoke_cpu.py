"""CPU smoke test for the learned token router (router_lib / train_router / the driver's
``sel="router"`` schemes): a tiny random model over a remapped vocabulary, real nq + oolong rows.

    HF_HUB_OFFLINE=1 python debug/learned_router/smoke_cpu.py

Checks
1. feature shapes: N_POS columns, exactly one start- and one end-offset bucket per token, relative
   features in [0, 1], gold flag == membership of the token's document in the gold set;
2. drop semantics through ``Transformer._compact_pooled_soft_tokens`` (custom mask, keep none,
   drop_slots): kept body tokens + markers + non-document tokens survive at original positions,
   nothing else; a document whose body is dropped ENTIRELY keeps its empty marker pair with
   ``keep_markers=True`` and vanishes completely with ``keep_markers=False`` (the historical
   ``mark_positions_free`` rule, which also strips the markers of partially kept documents);
3. an all-kept mask reproduces the full row's CE; batched K-sample CE == one-at-a-time CE;
4. the driver's router schemes (det / sampled / nomark) run end to end from a weights file;
5. the router_train / router_val rows are disjoint from every staged test row of every rung.
"""
import glob
import json
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "debug", "devloss_grid"))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import ctc_devloss_grid as G  # noqa: E402
import router_lib as RL  # noqa: E402
import train_router as TR  # noqa: E402

DATA = os.environ.get("SMOKE_DATA_ROOT", "/net/sneetches/data/prasann/devloss_grid/data")
TOK = os.environ.get("SMOKE_TOKENIZER") or sorted(
    glob.glob("/net/sneetches/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*"))[-1]
SCRATCH = os.environ.get("SMOKE_SCRATCH", "/tmp/claude-3018/-accounts-projects-berkeleynlp-prasann-projects-OLMo-core/"
                         "b25b0e58-1d93-45d2-a076-a44e278feae6/scratchpad/router_smoke")


def main():
    from transformers import AutoTokenizer

    from olmo_core.config import DType
    from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
    from olmo_core.nn.transformer import TransformerConfig

    tok = AutoTokenizer.from_pretrained(TOK)
    ids = G.RESERVED_IDS[G.FAMILY]
    rows, masks, golds = [], [], []
    for task, n in (("ctc_nq", 2), ("ctc_oolong", 1)):
        row = G.ROSTER[task]
        for ex in G.load_examples(DATA, row, "2k", n):
            r, m, _ = G.render_ctc_row(tok, ex, row["seg_task"], ids)
            rows.append(r); masks.append(m); golds.append(G.gold_docs(row["spec"], ex))
    print(f"[smoke] rows {[len(r) for r in rows]} gold {golds}")

    specials = [ids.doc_start, ids.doc_end, ids.eos, ids.landmark, ids.pad]
    uniq = sorted(set(int(t) for r in rows for t in r) - set(specials))
    remap = {t: i for i, t in enumerate(uniq)}
    U = len(uniq)
    for j, s in enumerate(specials):
        remap[s] = U + j
    V = ((U + 5 + 7) // 8) * 8
    sid = types.SimpleNamespace(doc_start=U, doc_end=U + 1, eos=U + 2, landmark=U + 3, pad=U + 4)
    inv = {v: k for k, v in remap.items()}
    pieces = [tok.convert_ids_to_tokens(inv[i]) if i in inv and inv[i] not in specials else f"<sp{i}>" for i in range(V)]
    rows_s = [[remap[int(t)] for t in r] for r in rows]
    decode = lambda t: tok.decode([inv[int(t)]]) if int(t) in inv and inv[int(t)] not in specials else f"<sp{t}>"  # noqa: E731
    stop_ids, _, tables = G.build_tables(np.concatenate([np.asarray(r) for r in rows_s]), V, pieces, sid, decode)

    torch.manual_seed(0)
    cfg = TransformerConfig.olmo2_190M(vocab_size=V, n_layers=2, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    G.attach_soft_tokens(model, sid, 42, stop_ids)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    TR.configure(model, stop_ids, tables)
    pst = model._pooled_soft_tokens

    # ---- 1. features ----
    for r, g in zip(rows_s, golds):
        x = torch.tensor(r)
        cid = build_chunk_ids_from_tokens(x[None], doc_start_id=sid.doc_start, doc_end_id=sid.doc_end, eos_id=sid.eos, mode="chunked")[0]
        n_docs = int(cid.max()) + 1
        f = RL.routed_features(x, cid, sid.doc_start, sid.doc_end, n_docs, sorted(g) if g else None)
        N = int(f["idx"].numel())
        assert f["pos"].shape == (N, RL.N_POS), f["pos"].shape
        assert torch.all(f["pos"][:, : RL.N_OFF].sum(1) == 1) and torch.all(f["pos"][:, RL.N_OFF : 2 * RL.N_OFF].sum(1) == 1)
        assert float(f["pos"][:, -2:].min()) >= 0 and float(f["pos"][:, -2:].max()) <= 1
        gold_t = torch.tensor([int(d) in (g or set()) for d in f["doc"].tolist()], dtype=torch.float32)
        assert torch.equal(f["gold"], gold_t)
        body = (cid >= 0) & (x != sid.doc_start) & (x != sid.doc_end)
        assert N == int(body.sum())
        assert list(RL.offset_bucket(torch.tensor([0, 15, 16, 31, 32, 63, 64, 10 ** 6])).tolist()) == [0, 15, 16, 16, 17, 17, 18, RL.N_OFF - 1]
        print(f"[smoke] features OK: N={N} docs={n_docs} gold tokens={int(f['gold'].sum())} max doc body={int(f['n'].max())}")

    # ---- 2. drop semantics incl. the all-dropped document ----
    r = rows_s[0]
    x = torch.tensor(r)
    cid = build_chunk_ids_from_tokens(x[None], doc_start_id=sid.doc_start, doc_end_id=sid.doc_end, eos_id=sid.eos, mode="chunked")[0]
    n_docs = int(cid.max()) + 1
    f = RL.routed_features(x, cid, sid.doc_start, sid.doc_end, n_docs, None)
    gen = torch.Generator().manual_seed(1)
    keep = torch.rand(f["idx"].shape, generator=gen) < 0.3
    keep[f["doc"] == 0] = False  # doc 0: whole body dropped
    keep[f["doc"] == 1] = True  # doc 1: whole body kept
    m = RL.keep_mask_from(f, keep, len(r))
    from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder

    def compact(keep_markers):
        pst["keep_token_mask_markers"] = keep_markers
        pst["keep_token_mask"] = m[None]
        model.train()
        model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=torch.zeros(1, n_docs, dtype=torch.bool))
        cb = model._compact_pooled_soft_tokens(x[None], None, -100)[0]
        model.eval()
        model._pooled_keep_holder = None
        return cb.position_ids[0].tolist(), cb.input_ids[0].tolist()

    is_marker = (x == sid.doc_start) | (x == sid.doc_end)
    outside = cid < 0
    exp = set(torch.nonzero(m | outside | (is_marker & (cid >= 0))).flatten().tolist())
    pos, ids_c = compact(True)
    assert set(pos) == exp and len(pos) == len(exp), (len(pos), len(exp))
    assert all(int(x[p]) == t for p, t in zip(pos, ids_c)), "compacted ids != original ids at kept positions"
    assert pos == sorted(pos) and sid.landmark not in ids_c, "slot emitted or order broken"
    d0 = torch.nonzero(cid == 0).flatten().tolist()
    d0_kept = [p for p in pos if p in set(d0)]
    assert d0_kept == [d0[0], d0[-1]] and all(is_marker[p] for p in d0_kept), d0_kept
    print(f"[smoke] keep_markers=True: doc 0 (body all dropped) -> empty marker pair {d0_kept}; T2/T={len(pos) / len(r):.3f}")
    pos_nm, _ = compact(False)
    whole = {int(d) for d in range(n_docs) if bool(keep[f['doc'] == d].all()) and bool((f['doc'] == d).any())}
    exp_nm = set(torch.nonzero(m | outside).flatten().tolist()) | {p for p in torch.nonzero(is_marker & (cid >= 0)).flatten().tolist() if int(cid[p]) in whole}
    assert set(pos_nm) == exp_nm, (len(pos_nm), len(exp_nm))
    assert not (set(d0) & set(pos_nm)), "doc 0 should vanish entirely without forced markers"
    print(f"[smoke] keep_markers=False: doc 0 vanishes entirely; partially kept docs lose their markers "
          f"(only {len(whole)} fully kept doc(s) keep theirs); T2/T={len(pos_nm) / len(r):.3f}")
    pst["keep_token_mask_markers"] = True

    # ---- 3. CE: all-kept == full; batched == single ----
    for r, mk, g in zip(rows_s, masks, golds):
        p = TR.prep_row(model, {"ids": r, "mask": mk, "gold": g, "rung": "2k"}, sid, "cpu")
        c = TR.ce_full(model, p)
        gm = torch.Generator().manual_seed(0)
        dec = torch.rand((4,) + tuple(p["feats"]["idx"].shape), generator=gm) < 0.4
        dec[0] = True
        dec[1] = False  # every body token dropped
        mm = torch.stack([RL.keep_mask_from(p["feats"], dec[k], p["T"]) for k in range(4)])
        ce_b, comp_b = TR.ce_masked(model, p, mm, max_tokens=10 ** 9)
        ce_s = torch.cat([TR.ce_masked(model, p, mm[k : k + 1])[0] for k in range(4)])
        ce_split, _ = TR.ce_masked(model, p, mm, max_tokens=1)  # forces one sample per sub-batch
        assert torch.allclose(ce_b, ce_s, atol=1e-4) and torch.allclose(ce_b, ce_split, atol=1e-4), (ce_b, ce_s)
        assert abs(float(ce_b[0]) - c) < 1e-4, (float(ce_b[0]), c)
        assert abs(float(comp_b[0]) - 1.0) < 1e-9 and float(comp_b[1]) < float(comp_b[2]) < 1.0, comp_b
        print(f"[smoke] CE full {c:.4f} | all-kept {float(ce_b[0]):.4f}  batched==single OK  T2/T {[round(v, 3) for v in comp_b.tolist()]}")

    # ---- 3b. one REINFORCE step moves the parameters, gradients finite ----
    class A:  # minimal args
        K, rows_per_step, epochs, min_epochs, patience, lr, emb_lr, val_samples = 4, 2, 1, 1, 1, 0.05, 0.01, 1
    prs = [TR.prep_row(model, {"ids": r, "mask": mk, "gold": g, "rung": "2k"}, sid, "cpu") for r, mk, g in zip(rows_s, masks, golds)]
    cef = [TR.ce_full(model, p) for p in prs]
    router, res, best, final = TR.train_one(model, "l0.2", prs[:2], prs[2:], cef[:2], cef[2:], A, tok, stop_ids, tables)
    assert len(res["steps"]) == 1 and np.isfinite(res["steps"][0]["grad_norm"]) and res["steps"][0]["grad_norm"] > 0
    assert float(final["w_pos"].abs().sum()) > 0
    print(f"[smoke] REINFORCE step OK: grad_norm={res['steps'][0]['grad_norm']:.3g} w_gold={float(final['w_gold']):+.3f}")

    # ---- 4. driver router schemes end to end ----
    os.makedirs(os.path.join(SCRATCH, "smoke"), exist_ok=True)
    st = RL.LinearRouter(int(model.embeddings.weight.shape[1]), "full").state()
    st["b"] = torch.tensor([0.4])
    st["w_gold"] = torch.tensor([3.0])
    st["w_emb"] = torch.randn(st["d_emb"]) * 0.5
    torch.save(st, os.path.join(SCRATCH, "smoke", "l0.2.pt"))
    torch.save(dict(st, variant="nogold"), os.path.join(SCRATCH, "smoke", "nogold_l0.2.pt"))
    G.ROUTER_WEIGHTS = os.path.join(SCRATCH, "{task}", "{name}.pt")
    schemes = {s: G.SCHEMES[s] for s in ("full", "router_l0.2", "router_l0.2_samp", "router_nogold_l0.2", "router_l0.2_nomark", "gold_rand20p8_noslot")}
    acc, _ = G.score_rows(model, rows_s, masks, golds, schemes, sid, tables, stop_ids, 42, decode, task_key="smoke")
    c = {s: float(np.mean(acc[s]["compaction"])) for s in schemes}
    rk = {s: float(np.nanmean(acc[s]["route_keep"])) if s.startswith("router") else None for s in schemes}
    print(f"[smoke] driver compaction {({k: round(v, 3) for k, v in c.items()})} route_keep {rk}")
    for s in schemes:
        assert all(np.isfinite(acc[s]["ce"])), s
    assert c["router_l0.2_nomark"] < c["router_l0.2"], c
    assert acc["router_l0.2"]["kept_docs"] == [0] * len(rows_s)

    # ---- 5. disjointness of the train/val rows vs every staged test row ----
    rep = json.load(open(os.path.join(HERE, "split_report.json")))
    import fetch_train_rows as FT

    for key in rep:
        row = G.ROSTER[f"ctc_{key}"]
        tq, tg, tqa = set(), set(), set()
        staged = sorted(glob.glob(os.path.join(DATA, row["subset"], "rung_*.jsonl")))  # incl. 16k stand-ins
        for path in staged:
            for ex in FT._read_jsonl(path, 64):
                q, qa, g, _, _ = FT.row_keys(ex, row["spec"])
                tq.add(q); tg |= g; tqa.add(qa)
        n = 0
        for split in ("router_train", "router_val"):
            for rung in rep[key]["rungs"]:
                for ex in G.load_examples(os.path.join(DATA, split), row, rung, 10 ** 6):
                    q, qa, g, _, hg = FT.row_keys(ex, row["spec"])
                    # (query, answer) is the key only without a gold set: templated queries + short
                    # id answers ("7; 10; 12") collide by chance on outlier
                    assert not (g & tg) and (hg or qa not in tqa), (key, split, rung)

                    assert ex["_router_src"]["index"] >= 64
                    n += 1
        print(f"[smoke] disjoint OK: {key} {n} train+val rows vs {64 * len(staged)} staged test-side rows")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
