"""Feasibility probe for the learned KV summarizer (pooled-doc-KV "B1/B2" variants).

Question: given only a document's MEAN INPUT EMBEDDING (+ length/position scalars), how well can a
predictor recover the document's per-layer MEAN post-RoPE K, mean V, and mean hidden state as
produced by a full forward pass — and how does that change with context length (2k -> 32k)?

Method:
  1. Build marker-wrapped contexts (preamble + <|box_start|>doc<|box_end|> * N) from the real
     contradiction PubMed claim pool at rungs 2k/8k/32k. Every PROBE doc is planted in several
     contexts per rung at random slots (same content, different context/position).
  2. Forward Qwen3-4B (dense) with use_cache + output_hidden_states; per doc span compute the mean
     input embed (feature) and per-layer mean K (post-RoPE raw, and a doc-center "derotated"
     frame), mean V, mean hidden state (targets).
  3. Fit ridge probes per (rung, layer, target) with doc-identity-disjoint train/val/test splits.
     Report centered R^2 and cosine against two references: the predict-train-mean baseline, and
     the same-doc-across-contexts consistency ceiling (no content-only predictor can beat it).

Run (needs internet the first time to fetch Qwen/Qwen3-4B into the node-local HF cache):

    srun -p berkeleynlp -w horton --gres=gpu:1 --qos=preemptive -t 120 \
      env HF_HOME=/data/prasann/hf_cache \
      /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python \
      research/token_routing/pooled_kv/kv_predictability_probe.py

Outputs: metrics JSON in research/token_routing/pooled_kv/probe_results/ (repo), raw feature/target tensors for
richer predictors later in /data/prasann/pooledkv_probe/ (node-local).
"""

import argparse
import hashlib
import json
import os
import random
import time

import torch
import torch.nn.functional as F

DOC_START, DOC_END = 151648, 151649  # qwen3 <|box_start|> / <|box_end|>
RUNGS = {2048: 40, 8192: 20, 32768: 10}  # target_len -> n_contexts
N_PROBE_DOCS = 40
PROBE_PLACEMENTS = 5  # contexts per rung each probe doc is planted into
LAYER_STRIDE = 3  # analyze layers 0, 3, 6, ..., plus the last
PREAMBLE = (
    "You will be given a list of scientific claims. Read them carefully; you will be asked "
    "questions about them afterwards.\n\n"
)
DATA = "/data/prasann/corpus-reasoning/data/contradiction_train_pubmed_both_n100_k3.jsonl"
OUT_TENSORS = "/data/prasann/pooledkv_probe"
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_results")
TARGET_NAMES = ("hid", "k_raw", "k_ctr", "v")


def log(msg):
    print(f"[probe] {msg}", flush=True)


def load_doc_pool(path, tokenizer, max_docs=9000):
    """Unique claim texts with token ids (unwrapped), 8..400 tokens."""
    pool, seen = [], set()
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            for d in ex["documents"]:
                t = d["text"].strip()
                h = hashlib.sha1(t.encode()).hexdigest()
                if not t or h in seen:
                    continue
                seen.add(h)
                ids = tokenizer(t, add_special_tokens=False)["input_ids"]
                if 8 <= len(ids) <= 400:
                    pool.append({"hash": h, "ids": ids})
            if len(pool) >= max_docs:
                break
    return pool


def build_context(pool, planted, target_len, preamble_ids, rng):
    """One context row: (ids, spans); spans = [(doc_hash, start, end_exclusive)]. ``planted`` docs
    are guaranteed present; the rest are random fillers up to the token budget."""
    ids = list(preamble_ids)
    docs = list(planted)
    est = sum(len(d["ids"]) + 2 for d in docs)
    budget = target_len - len(ids)
    while True:
        d = pool[rng.randrange(len(pool))]
        if est + len(d["ids"]) + 2 > budget:
            break
        docs.append(d)
        est += len(d["ids"]) + 2
    rng.shuffle(docs)
    spans = []
    for d in docs:
        s = len(ids)
        ids.extend([DOC_START, *d["ids"], DOC_END])
        spans.append((d["hash"], s, len(ids)))
    return ids, spans


def derotate(x, pos, inv_freq):
    """Inverse-RoPE a (n, H, D) tensor by per-row positions (n,): the doc-center frame."""
    freqs = pos[:, None].float() * inv_freq[None, :]  # (n, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    half = x.shape[-1] // 2
    rot_half = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos - rot_half * sin


def get_layer_kv(pkv, layer):
    """Post-RoPE (K, V) for one layer, across transformers cache API versions."""
    if hasattr(pkv, "key_cache"):
        return pkv.key_cache[layer], pkv.value_cache[layer]
    if hasattr(pkv, "layers"):
        return pkv.layers[layer].keys, pkv.layers[layer].values
    return pkv[layer][0], pkv[layer][1]


@torch.no_grad()
def collect_rung(model, pool, probe_docs, target_len, n_ctx, layers, inv_freq, preamble_ids, seed):
    """Forward n_ctx contexts; return (hashes, feats, targets) over doc occurrences (CPU fp32)."""
    rng = random.Random(seed)
    embed = model.get_input_embeddings()
    # Deterministic probe planting: each probe doc goes into PROBE_PLACEMENTS distinct contexts.
    plant = [[] for _ in range(n_ctx)]
    for d in probe_docs:
        for ci in rng.sample(range(n_ctx), min(PROBE_PLACEMENTS, n_ctx)):
            plant[ci].append(d)

    hashes, feats, doc_lens, centers = [], [], [], []
    targets = {name: [[] for _ in layers] for name in TARGET_NAMES}
    t0 = time.time()
    for ci in range(n_ctx):
        ids, spans = build_context(pool, plant[ci], target_len, preamble_ids, rng)
        x = torch.tensor([ids], device="cuda")
        out = model(x, use_cache=True, output_hidden_states=True)
        emb_row = embed(x)[0].float()
        hs = out.hidden_states
        starts = torch.tensor([s for _, s, _ in spans], device="cuda")
        ends = torch.tensor([e for _, _, e in spans], device="cuda")
        ctr = (starts + ends - 1) / 2.0
        for h, s, e in spans:
            hashes.append(h)
            feats.append(emb_row[s:e].mean(0).cpu())
            doc_lens.append(e - s)
            centers.append((s + e - 1) / 2.0)
        for si, li in enumerate(layers):
            k, v = get_layer_kv(out.past_key_values, li)
            k = k[0].transpose(0, 1).float()  # (T, H_kv, hd)
            v = v[0].transpose(0, 1).float()
            h_l = hs[li + 1][0].float()  # (T, d)
            mk = torch.stack([k[s:e].mean(0) for _, s, e in spans])
            mv = torch.stack([v[s:e].mean(0) for _, s, e in spans])
            mh = torch.stack([h_l[s:e].mean(0) for _, s, e in spans])
            mkc = derotate(mk, ctr, inv_freq)
            targets["hid"][si].append(mh.cpu())
            targets["k_raw"][si].append(mk.flatten(1).cpu())
            targets["k_ctr"][si].append(mkc.flatten(1).cpu())
            targets["v"][si].append(mv.flatten(1).cpu())
        del out
        if ci + 1 in (1, 2, 5, 10) or (ci + 1) % 10 == 0 or ci + 1 == n_ctx:
            per = (time.time() - t0) / (ci + 1)
            log(
                f"rung {target_len}: ctx {ci + 1}/{n_ctx} ({len(spans)} docs, T={len(ids)}) "
                f"{per:.1f}s/ctx ETA {(n_ctx - ci - 1) * per:.0f}s"
            )
    X = torch.stack(feats)
    extra = torch.tensor(
        [[dl / 100.0, c / target_len] for dl, c in zip(doc_lens, centers)], dtype=torch.float32
    )
    X = torch.cat([X, extra], dim=1)
    targets = {n: [torch.cat(per_l) for per_l in targets[n]] for n in TARGET_NAMES}
    return hashes, X, targets


def ridge_metrics(X, Y, split):
    """Ridge on train, lambda chosen on val, metrics on test: centered R^2 + raw cosine."""
    tr, va, te = (split == 0), (split == 1), (split == 2)
    Xm, Ym = X[tr].mean(0), Y[tr].mean(0)
    Xc, Yc = X - Xm, Y - Ym
    XtX = Xc[tr].T @ Xc[tr]
    XtY = Xc[tr].T @ Yc[tr]
    scale = XtX.diagonal().mean()
    eye = torch.eye(X.shape[1], device=X.device)
    best = None
    for rel in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        W = torch.linalg.solve(XtX + rel * scale * eye, XtY)
        err = ((Xc[va] @ W - Yc[va]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, W)
    W = best[1]
    pred_c = Xc[te] @ W
    r2 = float(1 - ((pred_c - Yc[te]) ** 2).sum() / (Yc[te] ** 2).sum().clamp(min=1e-9))
    pred = pred_c + Ym
    cos = float(F.cosine_similarity(pred, Y[te], dim=1).mean())
    cos_base = float(F.cosine_similarity(Ym.expand_as(Y[te]), Y[te], dim=1).mean())
    return r2, cos, cos_base


def consistency_ceiling(hashes, Y, probe_hashes, train_mask):
    """Same-doc-across-contexts agreement: centered pairwise cosine, and the between/(between+
    within) variance ratio — the R^2 any content-only predictor is capped by."""
    Ym = Y[train_mask].mean(0)
    Yc = Y - Ym
    by_doc = {}
    for i, h in enumerate(hashes):
        if h in probe_hashes:
            by_doc.setdefault(h, []).append(i)
    coss, w_var, b_var = [], [], []
    for idx in by_doc.values():
        if len(idx) < 2:
            continue
        Z = Yc[idx]
        mu = Z.mean(0, keepdim=True)
        w_var.append(((Z - mu) ** 2).sum(1).mean())
        b_var.append((mu**2).sum())
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                coss.append(float(F.cosine_similarity(Z[a], Z[b], dim=0)))
    if not coss:
        return None, None
    w, b = torch.stack(w_var).mean(), torch.stack(b_var).mean()
    return float(sum(coss) / len(coss)), float(b / (b + w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--seed", type=int, default=0)
    opts = ap.parse_args()
    os.makedirs(OUT_TENSORS, exist_ok=True)
    os.makedirs(OUT_JSON, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"loading {opts.model} ...")
    tok = AutoTokenizer.from_pretrained(opts.model)
    model = AutoModelForCausalLM.from_pretrained(
        opts.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.cuda().eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    layers = list(range(0, n_layers, LAYER_STRIDE))
    if n_layers - 1 not in layers:
        layers.append(n_layers - 1)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    inv_freq = (
        1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    ).cuda()
    log(f"{n_layers} layers, probing {layers}; head_dim={head_dim} rope_theta={cfg.rope_theta}")

    log("building doc pool ...")
    pool = load_doc_pool(DATA, tok)
    rng = random.Random(opts.seed)
    probe_docs = rng.sample(pool, N_PROBE_DOCS)
    probe_hashes = {d["hash"] for d in probe_docs}
    base_pool = [d for d in pool if d["hash"] not in probe_hashes]
    log(f"pool: {len(base_pool)} filler docs + {N_PROBE_DOCS} probe docs")
    preamble_ids = tok(PREAMBLE, add_special_tokens=False)["input_ids"]

    def split_of(h):
        if h in probe_hashes:
            return 2
        r = int(h[:8], 16) % 10
        return 0 if r < 8 else (1 if r == 8 else 2)

    results = {"model": opts.model, "layers": layers, "rungs": {}}
    for target_len, n_ctx in RUNGS.items():
        log(f"=== rung {target_len} ({n_ctx} contexts) ===")
        hashes, X, targets = collect_rung(
            model,
            base_pool,
            probe_docs,
            target_len,
            n_ctx,
            layers,
            inv_freq,
            preamble_ids,
            seed=opts.seed + target_len,
        )
        torch.save(
            {"hashes": hashes, "X": X, "targets": targets, "layers": layers},
            f"{OUT_TENSORS}/rung{target_len}.pt",
        )
        split = torch.tensor([split_of(h) for h in hashes])
        Xg = X.cuda()
        rung_res = {"n_doc_occurrences": len(hashes), "layers": {}}
        for si, li in enumerate(layers):
            entry = {}
            for name in TARGET_NAMES:
                Y = targets[name][si]
                r2, cos, cos_base = ridge_metrics(Xg, Y.cuda(), split.cuda())
                ceil_cos, ceil_r2 = consistency_ceiling(hashes, Y, probe_hashes, split == 0)
                entry[name] = {
                    "r2": round(r2, 4),
                    "cos": round(cos, 4),
                    "cos_meanbase": round(cos_base, 4),
                    "ceiling_r2": None if ceil_r2 is None else round(ceil_r2, 4),
                    "ceiling_cos": None if ceil_cos is None else round(ceil_cos, 4),
                }
            rung_res["layers"][li] = entry
            log(
                f"rung {target_len} L{li:2d}: "
                + " ".join(
                    f"{n}: R2 {entry[n]['r2']:+.3f} (ceil {entry[n]['ceiling_r2']})"
                    for n in ("hid", "k_ctr", "v")
                )
            )
        results["rungs"][target_len] = rung_res
        with open(f"{OUT_JSON}/probe_metrics.json", "w") as f:
            json.dump(results, f, indent=1)
    log(f"done -> {OUT_JSON}/probe_metrics.json")


if __name__ == "__main__":
    main()
