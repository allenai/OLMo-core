"""Interpret the learned router weights of one or more tasks: gold weight, bias, position profile,
and the vocabulary score s[v] = w_emb . e_rms(v) / sqrt(d) over the token types that occur in the
task's router TRAIN rows.

    python debug/learned_router/analyze_weights.py --tasks nq,outlier --data-root /data/prasann/devloss_grid/data

Needs each task's checkpoint embedding table (node-local /data, read directly from the safetensors,
no model build) -- run on the node that holds the checkpoints. Beyond-noise checks for the embedding
term: Pearson / Spearman correlation of s across the two seeds of l0.2 (``l0.2`` vs ``l0.2_s2``) and
across lambdas, over types occurring >= 3 times; and s against token IDF / digit / capitalised /
punctuation / top-100-frequency categories. Writes ``runs/<task>/weights_analysis.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "debug", "devloss_grid"))
sys.path.insert(0, HERE)

import ctc_devloss_grid as G  # noqa: E402
from router_lib import POS_NAMES  # noqa: E402


def load_embedding(ckpt: str) -> torch.Tensor:
    from safetensors import safe_open

    if not glob.glob(os.path.join(ckpt, "*.safetensors")):
        # olmo-core distcp: read ONLY the embedding tensor (olmo-core's own reader; torch's generic
        # FileSystemReader cannot parse these shards' storage metadata)
        from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

        path = G.find_distcp(ckpt)
        keys = [k for k in get_checkpoint_metadata(path).state_dict_metadata if k.endswith("embeddings.weight")]
        if len(keys) != 1:
            raise SystemExit(f"{path}: embedding keys {keys}")
        return next(load_keys(path, keys))
    for s in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
        with safe_open(s, "pt") as f:
            for k in f.keys():
                if k.endswith("embed_tokens.weight"):
                    return f.get_tensor(k)
    raise SystemExit(f"no embed_tokens.weight in {ckpt}")


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--data-root", default="/data/prasann/devloss_grid/data")
    ap.add_argument("--tokenizer", default=os.environ.get("DEVLOSS_TOKENIZER", G.TOKENIZER_BY_FAMILY[G.FAMILY]))
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = G.RESERVED_IDS[G.FAMILY]
    man = json.load(open(os.path.join(REPO, "debug", "devloss_grid", "manifest.json")))
    for task in a.tasks.split(","):
        wdir = os.path.join(HERE, "weights", task)
        states = {os.path.basename(p)[:-3]: torch.load(p, map_location="cpu") for p in sorted(glob.glob(os.path.join(wdir, "*.pt")))}
        if not states:
            print(f"{task}: no weights")
            continue
        row = G.ROSTER[f"ctc_{task}"]
        cnt = Counter()
        for rung in ("2k", "8k"):
            for ex in G.load_examples(os.path.join(a.data_root, "router_train"), row, rung, 10 ** 6):
                r_ids, _, _ = G.render_ctc_row(tok, ex, row["seg_task"], ids)
                x = torch.tensor(r_ids)
                from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens

                cid = build_chunk_ids_from_tokens(x[None], doc_start_id=ids.doc_start, doc_end_id=ids.doc_end, eos_id=ids.eos, mode="chunked")[0]
                body = (cid >= 0) & (x != ids.doc_start) & (x != ids.doc_end)
                cnt.update(x[body].tolist())
        types = sorted(cnt)
        E = load_embedding(man["tasks"][task]["ckpt"])
        e = E[torch.tensor(types)].float()
        e = e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-6)
        pieces = [tok.decode([t]) for t in types]
        freq = np.array([cnt[t] for t in types])
        n_tot = freq.sum()
        idf = -np.log(freq / n_tot)
        is_dig = np.array([any(ch.isdigit() for ch in p) for p in pieces])
        is_cap = np.array([p.strip()[:1].isupper() for p in pieces])
        is_punct = np.array([bool(p.strip()) and not any(ch.isalnum() for ch in p) for p in pieces])
        top100 = set(np.argsort(-freq)[:100].tolist())
        is_top = np.array([i in top100 for i in range(len(types))])
        ge3 = freq >= 3
        res = {"task": task, "n_types": len(types), "n_types_ge3": int(ge3.sum()), "configs": {}, "pairs": {}}
        svals = {}
        for name, st in states.items():
            summ = {"variant": st["variant"], "b": float(st["b"]), "w_gold": float(st["w_gold"]),
                    "w_pos": {n: round(float(v), 3) for n, v in zip(POS_NAMES, st["w_pos"].tolist())},
                    "w_emb_norm": float(st["w_emb"].norm())}
            if st["variant"] != "noemb":
                s = ((e @ st["w_emb"].float()) / math.sqrt(st["d_emb"])).numpy()
                svals[name] = s
                order = np.argsort(s)
                pick = [i for i in order if ge3[i]]
                fmt = lambda lst: [{"piece": pieces[i], "s": round(float(s[i]), 3), "count": int(freq[i])} for i in lst]  # noqa: E731
                summ.update({
                    "s_std_ge3": float(s[ge3].std()),
                    "s_weighted_std": float(np.sqrt(np.average((s - np.average(s, weights=freq)) ** 2, weights=freq))),
                    "top": fmt(pick[::-1][: a.top]), "bottom": fmt(pick[: a.top]),
                    "corr_idf_ge3": float(np.corrcoef(s[ge3], idf[ge3])[0, 1]),
                    "mean_s": {"digit": float(s[is_dig & ge3].mean()) if (is_dig & ge3).any() else None,
                               "capitalised": float(s[is_cap & ge3].mean()) if (is_cap & ge3).any() else None,
                               "punct": float(s[is_punct & ge3].mean()) if (is_punct & ge3).any() else None,
                               "top100_freq": float(s[is_top].mean()),
                               "all_ge3": float(s[ge3].mean())},
                })
            res["configs"][name] = summ
        names = sorted(svals)
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                res["pairs"][f"{n1}|{n2}"] = {"pearson_ge3": float(np.corrcoef(svals[n1][ge3], svals[n2][ge3])[0, 1]),
                                             "spearman_ge3": spearman(svals[n1][ge3], svals[n2][ge3])}
        out = os.path.join(HERE, "runs", task, "weights_analysis.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
        print(f"== {task}: {len(types)} types ({int(ge3.sum())} with count>=3) -> {out}")
        for name, sm in res["configs"].items():
            print(f"  {name:14} b={sm['b']:+.2f} w_gold={sm['w_gold']:+.2f} |w_emb|={sm['w_emb_norm']:.2f}"
                  + (f" s_std={sm['s_std_ge3']:.2f} corr(s,idf)={sm['corr_idf_ge3']:+.2f} mean_s={ {k: (round(v, 2) if v is not None else None) for k, v in sm['mean_s'].items()} }" if "s_std_ge3" in sm else ""))
            if "top" in sm:
                print("     top:", " ".join(repr(t["piece"]) for t in sm["top"][:15]))
                print("     bot:", " ".join(repr(t["piece"]) for t in sm["bottom"][:15]))
        for k, v in res["pairs"].items():
            print(f"  corr {k:30} pearson {v['pearson_ge3']:+.2f} spearman {v['spearman_ge3']:+.2f}")


if __name__ == "__main__":
    main()
