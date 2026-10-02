"""CPU smoke test for ctc_devloss_grid.py: a tiny random model over a REMAPPED vocabulary, two
real nq rows rendered through the real tokenizer + marker scaffold, every scheme end to end.

    HF_HUB_OFFLINE=1 python debug/devloss_grid/smoke_cpu.py

Checks: shapes; ``full`` compaction == 1.0; k0 < first16 < first64 < 1.0; K-matched rules compact
identically; gold-forced masks keep every gold doc; and the marker-scaffold prompt text equals the
vendored olmo-eval ``spec.build_prompt`` (non-alpaca form) once the markers are stripped.
"""

import glob
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctc_devloss_grid as G  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
DATA_ROOT = os.environ.get("SMOKE_DATA_ROOT", "/net/cubbins/data/prasann/ctc_suite_staged/eval_rungs")
TOK = os.environ.get("SMOKE_TOKENIZER") or sorted(
    glob.glob("/net/cubbins/data/prasann/hf_cache/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*")
)[-1]
TASK, RUNG, ROWS, SEED = "ctc_nq", "2k", 2, 42


def main():
    from transformers import AutoTokenizer

    from olmo_core.config import DType
    from olmo_core.nn.transformer import TransformerConfig

    G.check_task_cfg()
    G.check_gold_conventions()
    tok = AutoTokenizer.from_pretrained(TOK)
    ids = G.RESERVED_IDS[G.FAMILY]
    row = G.ROSTER[TASK]
    examples = G.load_examples(DATA_ROOT, row, RUNG, ROWS)
    rows, masks, golds = [], [], []
    for ex in examples:
        r, m, n_spans = G.render_ctc_row(tok, ex, row["seg_task"], ids)
        assert n_spans == len(ex["documents"]), (n_spans, len(ex["documents"]))
        rows.append(r); masks.append(m); golds.append(G.gold_docs(row["spec"], ex))
    print(f"[smoke] rows {[len(r) for r in rows]} answer toks {[sum(m) for m in masks]} gold {golds}")

    # --- prompt text == vendored spec.build_prompt (non-alpaca), markers stripped ---
    if os.path.isdir(G.OLMO_EVAL_VENDOR):
        sys.path.insert(0, G.OLMO_EVAL_VENDOR)
        from ctc.format import registry
        from ctc.tasks import load_all

        load_all()
        spec = registry.get(row["spec"])
        text = tok.decode(rows[0])
        user = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        user = user.replace("<|box_start|>", "").replace("<|box_end|>", "")
        ref = spec.build_prompt(examples[0], query_position="both", use_alpaca=False, use_titles=False)
        assert user.strip() == ref.strip(), "marker-scaffold prompt != vendored spec.build_prompt\n--ours--\n" + user[:600] + "\n--ref--\n" + ref[:600]
        print("[smoke] prompt text matches vendored spec.build_prompt (non-alpaca form) after stripping markers")

    # --- remap the vocabulary so a tiny model fits on CPU ---
    specials = [ids.doc_start, ids.doc_end, ids.eos, ids.landmark, ids.pad]
    uniq = sorted(set(int(t) for r in rows for t in r) - set(specials))
    remap = {t: i for i, t in enumerate(uniq)}
    U = len(uniq)
    for j, s in enumerate(specials):
        remap[s] = U + j
    V = ((U + 5 + 7) // 8) * 8
    small_ids = types.SimpleNamespace(doc_start=U, doc_end=U + 1, eos=U + 2, landmark=U + 3, pad=U + 4)
    inv = {v: k for k, v in remap.items()}
    pieces = [tok.convert_ids_to_tokens(inv[i]) if i in inv and inv[i] not in specials else f"<sp{i}>" for i in range(V)]
    rows_s = [[remap[int(t)] for t in r] for r in rows]
    decode = lambda t: tok.decode([inv[int(t)]]) if int(t) in inv and inv[int(t)] not in specials else f"<sp{t}>"  # noqa: E731
    all_ids = np.concatenate([np.asarray(r) for r in rows_s])
    stop_ids, shown, tables = G.build_tables(all_ids, V, pieces, small_ids, decode)
    print(f"[smoke] vocab {V}, stop set {len(stop_ids)} ids, top dropped {shown[:8]}")

    cfg = TransformerConfig.olmo2_190M(vocab_size=V, n_layers=2, fused_ops=False, dtype=DType.float32)
    model = cfg.build(init_device="cpu")
    G.attach_soft_tokens(model, small_ids, SEED, stop_ids)
    torch.manual_seed(0)
    acc, degenerate = G.score_rows(model, rows_s, masks, golds, G.SCHEMES, small_ids, tables, stop_ids, SEED, decode)

    c = {s: float(np.mean(acc[s]["compaction"])) for s in G.SCHEMES}
    print("[smoke] compaction:", {k: round(v, 3) for k, v in c.items()})
    for s in G.SCHEMES:
        assert len(acc[s]["ce"]) == ROWS and all(np.isfinite(acc[s]["ce"])), s
    assert c["full"] == 1.0
    assert c["k0"] < c["first16"] < c["first64"] < 1.0, c
    assert abs(c["first16"] - c["fl16"]) < 1e-9 and abs(c["first16"] - c["idf16"]) < 1e-9, c
    assert abs(c["first64"] - c["fl64"]) < 1e-9 and abs(c["first64"] - c["idf64"]) < 1e-9, c
    assert c["rand33"] > c["k0"] and c["gold_rand33"] >= c["rand33"] - 1e-9, c
    assert c["gold_fl64"] >= c["fl64"], c
    for i, g in enumerate(golds):
        assert acc["gold_fl64"]["kept_docs"][i] >= len(g) and acc["gold_rand33"]["kept_docs"][i] >= len(g), (i, g)
    assert not degenerate["gold_fl64"], degenerate
    # random-position twin: same per-document budget as first_last on top of the same 8-token prefix
    assert abs(c["gold_rand20p8_noslot"] - c["gold_fl20p8_noslot"]) < 1e-9, c
    # full == k0 CE only by accident; but top1 vs full must be 1.0 for 'full' itself
    assert all(t == 1.0 for t in acc["full"]["top1"])
    print("[smoke] OK")


if __name__ == "__main__":
    main()
