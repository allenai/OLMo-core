"""Fit the MLP KV summarizer g_theta and measure ATTENTION-OPERATION fidelity.

Phase 1 — fit: g_theta(mean input embed [+len, pos-frac]) -> per-layer (center-frame mean K, mean V)
on the tensors saved by kv_predictability_probe.py (train-split docs, all rungs pooled). Report
test R^2 / cosine next to the ridge baseline.

Phase 2 — fidelity (the metric that matters): for real queries at the end of held-out-doc contexts,
compare the attention operation itself between FULL attention and each compression variant, where
every document's per-token KV columns are replaced by ONE slot (K_hat rotated to the doc center,
logit + log(doc_len)):
  * meanpool  — slot = exact per-layer mean K/V (what PooledDocKVAttention trains with today)
  * mlp       — slot = g_theta prediction
  * ridge     — slot = linear-probe prediction
Per (rung, layer), aggregated over queries x heads:
  * out_cos / out_relerr — cosine and relative L2 between the softmax-weighted value outputs
  * docmass_tv           — total-variation distance between the per-doc softmax mass allocations
  * top_doc_agree        — does the argmax-mass document stay the same
The hand-computed attention is validated against HF output_attentions before anything is trusted.

Run:
    srun -p berkeleynlp -w horton --gres=gpu:1 --qos=preemptive -t 60 \
      env HF_HOME=/data/prasann/hf_cache \
      /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python \
      research/token_routing/pooled_kv/fit_summarizer_and_fidelity.py
"""

import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_predictability_probe as probe  # noqa: E402

OUT_TENSORS = probe.OUT_TENSORS
OUT_JSON = probe.OUT_JSON
FIDELITY_CTX_PER_RUNG = 4
N_QUERY_TOK = 24
QUERY_SUFFIX = (
    "\n\nQuestion: Based on the claims above, which pairs of claims directly contradict each "
    "other? Think carefully about every claim before answering. Answer:"
)


def log(msg):
    print(f"[fit] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: fit g_theta
# ---------------------------------------------------------------------------


class Summarizer(torch.nn.Module):
    """Trunk MLP + per-(layer, target) linear heads: the g_theta prototype."""

    def __init__(self, d_in, d_kv, layers):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(d_in, 2048),
            torch.nn.GELU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.GELU(),
        )
        self.heads = torch.nn.ModuleDict(
            {f"{n}_{li}": torch.nn.Linear(2048, d_kv) for li in layers for n in ("k", "v")}
        )

    def forward(self, x, layer, name):
        return self.heads[f"{name}_{layer}"](self.trunk(x))


def fit_summarizer(layers, split_of):
    """Train g_theta on saved tensors (all rungs pooled, train docs only). Returns the model plus
    per-layer standardization stats and the ridge weight dicts (baseline + fidelity reuse)."""
    Xs, hashes_all, tgt = [], [], {n: [[] for _ in layers] for n in ("k_ctr", "v")}
    for rung in probe.RUNGS:
        blob = torch.load(f"{OUT_TENSORS}/rung{rung}.pt", weights_only=False)
        assert blob["layers"] == layers
        Xs.append(blob["X"])
        hashes_all.extend(blob["hashes"])
        for si in range(len(layers)):
            tgt["k_ctr"][si].append(blob["targets"]["k_ctr"][si])
            tgt["v"][si].append(blob["targets"]["v"][si])
    X = torch.cat(Xs).cuda()
    split = torch.tensor([split_of(h) for h in hashes_all]).cuda()
    tr, va, te = split == 0, split == 1, split == 2
    log(
        f"fit data: {X.shape[0]} occurrences ({int(tr.sum())} train / {int(va.sum())} val / "
        f"{int(te.sum())} test)"
    )
    d_kv = tgt["k_ctr"][0][0].shape[1]
    Xm, Xs_ = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
    Xn = (X - Xm) / Xs_

    Y = {}
    stats = {}
    for n in ("k_ctr", "v"):
        for si, li in enumerate(layers):
            y = torch.cat(tgt[n][si]).cuda()
            ym, ys = y[tr].mean(0), y[tr].std(0).clamp(min=1e-6)
            Y[(n, li)] = (y - ym) / ys
            stats[(n, li)] = (ym, ys)

    model = Summarizer(X.shape[1], d_kv, layers).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    idx_tr = tr.nonzero(as_tuple=True)[0]
    best_val, best_state, patience = None, None, 0
    t0 = time.time()
    for epoch in range(200):
        model.train()
        perm = idx_tr[torch.randperm(len(idx_tr), device="cuda")]
        for b in range(0, len(perm), 1024):
            bi = perm[b : b + 1024]
            z = model.trunk(Xn[bi])
            loss = sum(
                F.mse_loss(model.heads[f"{n[0]}_{li}"](z), Y[(n, li)][bi])
                for n in ("k_ctr", "v")
                for li in layers
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            z = model.trunk(Xn[va])
            vloss = float(
                sum(
                    F.mse_loss(model.heads[f"{n[0]}_{li}"](z), Y[(n, li)][va])
                    for n in ("k_ctr", "v")
                    for li in layers
                )
            )
        if best_val is None or vloss < best_val - 1e-4:
            best_val, patience = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 12:
                break
        if epoch in (0, 1, 4, 9) or epoch % 20 == 19:
            log(f"epoch {epoch + 1}: val {vloss:.4f} ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    log(f"g_theta trained ({epoch + 1} epochs, best val {best_val:.4f})")

    # Ridge baselines on the SAME pooled data (for fidelity reuse).
    ridge_W = {}
    fit_report = {}
    XtX = None
    with torch.no_grad():
        Xc = Xn  # already standardized; ridge on the same features
        XtX = Xc[tr].T @ Xc[tr]
        eye = torch.eye(Xc.shape[1], device="cuda") * XtX.diagonal().mean()
        for n in ("k_ctr", "v"):
            for si, li in enumerate(layers):
                y = Y[(n, li)]
                best_r = None
                for rel in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
                    W = torch.linalg.solve(XtX + rel * eye, Xc[tr].T @ y[tr])
                    err = ((Xc[va] @ W - y[va]) ** 2).sum()
                    if best_r is None or err < best_r[0]:
                        best_r = (err, W)
                ridge_W[(n, li)] = best_r[1]
                ym, ys = stats[(n, li)]
                yt = y[te] * ys + ym
                for tag, pred_n in (
                    ("mlp", model(Xn[te], li, n[0])),
                    ("ridge", Xc[te] @ best_r[1]),
                ):
                    pred = pred_n * ys + ym
                    r2 = float(1 - ((pred - yt) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum())
                    cos = float(F.cosine_similarity(pred, yt, dim=1).mean())
                    fit_report.setdefault(f"L{li}", {}).setdefault(n, {})[tag] = {
                        "r2": round(r2, 4),
                        "cos": round(cos, 4),
                    }
    for li in layers:
        e = fit_report[f"L{li}"]
        log(
            f"L{li:2d} test: k_ctr mlp R2 {e['k_ctr']['mlp']['r2']:+.3f} "
            f"(ridge {e['k_ctr']['ridge']['r2']:+.3f})  "
            f"v mlp R2 {e['v']['mlp']['r2']:+.3f} (ridge {e['v']['ridge']['r2']:+.3f})"
        )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "Xm": Xm,
            "Xs": Xs_,
            "stats": stats,
            "ridge_W": ridge_W,
            "layers": layers,
        },
        f"{OUT_TENSORS}/summarizer.pt",
    )
    return model, (Xm, Xs_), stats, ridge_W, fit_report


# ---------------------------------------------------------------------------
# Phase 2: attention-operation fidelity
# ---------------------------------------------------------------------------


def rotate(x, pos, inv_freq):
    """Forward-RoPE a (n, H, D) tensor by per-row positions (n,)."""
    freqs = pos[:, None].float() * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    half = x.shape[-1] // 2
    rot_half = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rot_half * sin


@torch.no_grad()
def compute_queries(hf_model, attn_inputs, layer, qpos, inv_freq):
    """Post-RoPE queries (n_q, n_heads, hd) for one layer from the stashed self_attn input."""
    sa = hf_model.model.layers[layer].self_attn
    x = attn_inputs[layer][0, qpos]  # (n_q, d), already input-layernormed
    q = sa.q_proj(x)
    n_q = q.shape[0]
    hd = sa.head_dim
    q = q.view(n_q, -1, hd)
    q = sa.q_norm(q)
    return rotate(q.float(), qpos.float(), inv_freq)


@torch.no_grad()
def validate_attention_math(hf_model, tok, pool, inv_freq, preamble_ids, seed):
    """Sanity: hand-computed softmax probs == HF eager output_attentions on a short context."""
    rng = random.Random(seed)
    ids, spans = probe.build_context(pool, [], 1024, preamble_ids, rng)
    ids = ids + tok(QUERY_SUFFIX, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], device="cuda")
    attn_inputs = {}
    hooks = [
        hf_model.model.layers[li].self_attn.register_forward_pre_hook(
            lambda m, a, kw, li=li: attn_inputs.__setitem__(
                li, (kw.get("hidden_states") if kw else a[0]).detach()
            ),
            with_kwargs=True,
        )
        for li in (0, 17, 35)
    ]
    # NB the eager flip is NOT reliably reversible across transformers versions (a restored
    # "sdpa" was observed to keep dispatching eager, OOMing the next 32k forward with 137GiB of
    # materialized T x T weights). This function must therefore run AFTER all other forwards.
    prev_impl = hf_model.config._attn_implementation
    hf_model.config._attn_implementation = "eager"  # sdpa cannot return attention probs
    hf_model.model.config._attn_implementation = "eager"
    try:
        out = hf_model(x, use_cache=True, output_attentions=True)
    finally:
        hf_model.config._attn_implementation = prev_impl
        hf_model.model.config._attn_implementation = prev_impl
        for h in hooks:
            h.remove()
    assert out.attentions is not None and out.attentions[0] is not None
    T = x.shape[1]
    qpos = torch.arange(T - 4, T, device="cuda")
    worst, mean_d = 0.0, 0.0
    for li in (0, 17, 35):
        k, _ = probe.get_layer_kv(out.past_key_values, li)
        k = k[0].transpose(0, 1).float()  # (T, H_kv, hd)
        q = compute_queries(hf_model, attn_inputs, li, qpos, inv_freq)  # (4, 32, 128)
        n_rep = q.shape[1] // k.shape[1]
        kx = k.repeat_interleave(n_rep, dim=1)  # (T, 32, 128)
        logits = torch.einsum("qhd,thd->qht", q, kx) / (k.shape[-1] ** 0.5)
        mask = torch.arange(T, device="cuda")[None, :] > qpos[:, None]
        logits.masked_fill_(mask[:, None, :], float("-inf"))
        mine = logits.softmax(-1)
        hf_probs = out.attentions[li][0].float()  # (32, T, T)
        ref = hf_probs[:, qpos, :].transpose(0, 1)  # (4, 32, T)
        worst = max(worst, float((mine - ref).abs().max()))
        mean_d = max(mean_d, float((mine - ref).abs().mean()))
    log(f"attention-math validation: max|mine - HF| = {worst:.2e}, mean = {mean_d:.2e}")
    # HF returns bf16-rounded probs (+-4e-3 quantization), so the max can reach a few e-2 from
    # numerics alone; a real indexing/RoPE bug would blow up the MEAN, which must stay tiny.
    assert worst < 6e-2 and mean_d < 5e-4, "hand-computed attention disagrees with HF"


def make_grouped_pool(pool, group_size, rng, n_groups=4000):
    """A pool of larger chunks: each = ``group_size`` random claims' token ids concatenated."""
    if group_size == 1:
        return pool
    out = []
    for _ in range(n_groups):
        docs = rng.sample(pool, group_size)
        out.append(
            {
                "hash": "+".join(d["hash"][:8] for d in docs),
                "ids": [t for d in docs for t in d["ids"]],
            }
        )
    return out


@torch.no_grad()
def fidelity_for_rung(
    hf_model,
    tok,
    pool,
    target_len,
    layers,
    inv_freq,
    preamble_ids,
    g_theta,
    xstats,
    ystats,
    ridge_W,
    seed,
    variants=("meanpool", "mlp", "ridge"),
):
    """Per-layer fidelity metrics for the requested compression variants at one rung."""
    rng = random.Random(seed)
    Xm, Xs_ = xstats
    embed = hf_model.get_input_embeddings()
    acc = {v: {li: {"cos": [], "rel": [], "tv": [], "top": []} for li in layers} for v in variants}
    chunk_lens = []
    suffix_ids = tok(QUERY_SUFFIX, add_special_tokens=False)["input_ids"]
    for ci in range(FIDELITY_CTX_PER_RUNG):
        ids, spans = probe.build_context(pool, [], target_len - len(suffix_ids), preamble_ids, rng)
        ids = ids + suffix_ids
        x = torch.tensor([ids], device="cuda")
        T = x.shape[1]
        attn_inputs = {}
        hooks = [
            hf_model.model.layers[li].self_attn.register_forward_pre_hook(
                lambda m, a, kw, li=li: attn_inputs.__setitem__(
                    li, (a[0] if a else kw["hidden_states"]).detach()
                ),
                with_kwargs=True,
            )
            for li in layers
        ]
        if target_len >= 32768:
            log(f"pre-forward GPU mem: {torch.cuda.memory_allocated() / 2**30:.1f}GiB")
        try:
            out = hf_model(x, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = hf_model(x, use_cache=True)
        if target_len >= 32768:
            log(f"post-forward GPU mem: {torch.cuda.memory_allocated() / 2**30:.1f}GiB")
        for h in hooks:
            h.remove()
        out.logits = None  # full-vocab logits are (T, 152k) — never needed here
        emb_row = embed(x)[0].float()
        starts = torch.tensor([s for _, s, _ in spans], device="cuda")
        ends = torch.tensor([e for _, _, e in spans], device="cuda")
        centers = (starts + ends - 1) / 2.0
        lens = (ends - starts).float()
        feats = torch.stack([emb_row[s:e].mean(0) for _, s, e in spans] + [])
        extra = torch.stack([lens / 100.0, centers / target_len], dim=1)
        Xn = (torch.cat([feats, extra], dim=1) - Xm) / Xs_
        qpos = torch.arange(T - N_QUERY_TOK, T, device="cuda")
        doc_col = torch.full((T,), -1, dtype=torch.long, device="cuda")
        for di, (_, s, e) in enumerate(spans):
            doc_col[s:e] = di
        is_doc = doc_col >= 0
        n_docs = len(spans)

        for li in layers:
            k, v = probe.get_layer_kv(out.past_key_values, li)
            k = k[0].transpose(0, 1).float()
            v = v[0].transpose(0, 1).float()
            H_kv, hd = k.shape[1], k.shape[2]
            q = compute_queries(hf_model, attn_inputs, li, qpos, inv_freq)  # (nq, Hq, hd)
            n_rep = q.shape[1] // H_kv
            scale = hd**-0.5
            kx = k.repeat_interleave(n_rep, dim=1)
            vx = v.repeat_interleave(n_rep, dim=1)
            logits_full = torch.einsum("qhd,thd->qht", q, kx) * scale  # (nq, Hq, T)
            causal = torch.arange(T, device="cuda")[None, :] > qpos[:, None]  # (nq, T)
            logits_full.masked_fill_(causal[:, None, :], float("-inf"))
            p_full = logits_full.softmax(-1)
            o_full = torch.einsum("qht,thd->qhd", p_full, vx)
            onehot = F.one_hot(doc_col.clamp(min=0), n_docs).float() * is_doc[:, None]
            mass_full = torch.einsum("qht,tD->qhD", p_full, onehot)  # (nq, Hq, n_docs)

            # Slot K/V per variant.
            mk = torch.stack([k[s:e].mean(0) for _, s, e in spans])  # (D, H_kv, hd)
            mv = torch.stack([v[s:e].mean(0) for _, s, e in spans])
            slots = {"meanpool": (mk, mv)} if "meanpool" in variants else {}
            for tag in (t for t in ("mlp", "ridge") if t in variants):
                pk, pv = [], []
                for n, store in (("k_ctr", pk), ("v", pv)):
                    ym, ys = ystats[(n, li)]
                    if tag == "mlp":
                        pred = g_theta(Xn, li, n[0]) * ys + ym
                    else:
                        pred = Xn @ ridge_W[(n, li)] * ys + ym
                    store.append(pred.view(n_docs, H_kv, hd))
                k_ctr_pred = pk[0]
                k_pred = rotate(k_ctr_pred, centers, inv_freq)  # to doc-center positions
                slots[tag] = (k_pred, pv[0])

            base_logits = logits_full.masked_fill(is_doc[None, None, :], float("-inf"))
            for tag, (sk, sv) in slots.items():
                skx = sk.repeat_interleave(n_rep, dim=1)
                svx = sv.repeat_interleave(n_rep, dim=1)
                slot_logits = (
                    torch.einsum("qhd,Dhd->qhD", q, skx) * scale + lens.log()[None, None, :]
                )
                cat = torch.cat([base_logits, slot_logits], dim=-1)
                p = cat.softmax(-1)
                o = torch.einsum("qht,thd->qhd", p[..., :T], vx) + torch.einsum(
                    "qhD,Dhd->qhd", p[..., T:], svx
                )
                mass = p[..., T:]
                a = acc[tag][li]
                a["cos"].append(
                    float(F.cosine_similarity(o.flatten(0, 1), o_full.flatten(0, 1), dim=1).mean())
                )
                a["rel"].append(
                    float(((o - o_full).norm(dim=-1) / o_full.norm(dim=-1).clamp(min=1e-9)).mean())
                )
                a["tv"].append(float(0.5 * (mass - mass_full).abs().sum(-1).mean()))
                a["top"].append(float((mass.argmax(-1) == mass_full.argmax(-1)).float().mean()))
            del k, v, kx, vx, logits_full, p_full, o_full, mass_full, base_logits, onehot
            if target_len >= 32768:
                log(f"    L{li} mem {torch.cuda.memory_allocated() / 2**30:.1f}GiB")
        del out, attn_inputs
        torch.cuda.empty_cache()
        chunk_lens.extend((ends - starts).tolist())
        log(
            f"rung {target_len}: fidelity ctx {ci + 1}/{FIDELITY_CTX_PER_RUNG} (docs={n_docs}, T={T})"
        )
    res = {
        tag: {
            li: {m: round(sum(vals[m]) / len(vals[m]), 4) for m in vals}
            for li, vals in per_layer.items()
        }
        for tag, per_layer in acc.items()
    }
    res["mean_chunk_tokens"] = round(sum(chunk_lens) / len(chunk_lens), 1)
    return res


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("loading model ...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    hf_model.cuda().eval()
    cfg = hf_model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    inv_freq = (
        1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    ).cuda()

    log("rebuilding doc pool + splits (must match the probe run) ...")
    pool = probe.load_doc_pool(probe.DATA, tok)
    rng = random.Random(0)
    probe_docs = rng.sample(pool, probe.N_PROBE_DOCS)
    probe_hashes = {d["hash"] for d in probe_docs}

    def split_of(h):
        if h in probe_hashes:
            return 2
        r = int(h[:8], 16) % 10
        return 0 if r < 8 else (1 if r == 8 else 2)

    heldout_pool = [d for d in pool if split_of(d["hash"]) != 0]
    lens = sorted(len(d["ids"]) + 2 for d in heldout_pool)
    log(
        f"held-out pool for fidelity contexts: {len(heldout_pool)} docs; chunk tokens "
        f"p25/p50/p75 = {lens[len(lens)//4]}/{lens[len(lens)//2]}/{lens[3*len(lens)//4]}"
    )
    preamble_ids = tok(probe.PREAMBLE, add_special_tokens=False)["input_ids"]

    layers = list(range(0, cfg.num_hidden_layers, probe.LAYER_STRIDE))
    if cfg.num_hidden_layers - 1 not in layers:
        layers.append(cfg.num_hidden_layers - 1)

    g_theta, xstats, ystats, ridge_W, fit_report = fit_summarizer(layers, split_of)
    g_theta.eval()
    torch.cuda.empty_cache()
    log(f"post-fit GPU mem: {torch.cuda.memory_allocated() / 2**30:.1f}GiB")

    results = {"fit": fit_report, "fidelity": {}}
    fidelity_rungs = [int(r) for r in os.environ.get("FIDELITY_RUNGS", "").split(",") if r] or list(
        probe.RUNGS
    )
    for rung in fidelity_rungs:
        log(f"=== fidelity rung {rung} ===")
        res = fidelity_for_rung(
            hf_model,
            tok,
            heldout_pool,
            rung,
            layers,
            inv_freq,
            preamble_ids,
            g_theta,
            xstats,
            ystats,
            ridge_W,
            seed=777 + rung,
        )
        results["fidelity"][rung] = res
        for li in layers:
            log(
                f"rung {rung} L{li:2d}: "
                + "  ".join(
                    f"{tag}: cos {res[tag][li]['cos']:.3f} tv {res[tag][li]['tv']:.3f} "
                    f"top {res[tag][li]['top']:.2f}"
                    for tag in ("meanpool", "mlp", "ridge")
                )
            )
        with open(f"{OUT_JSON}/fidelity_metrics.json", "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=1, default=str)

    # Chunk-size scaling: exact-meanpool fidelity with G claims concatenated per chunk. (The
    # learned variants stay at G=1 — g_theta was trained on single-claim chunks.)
    results["fidelity_by_group_size"] = {}
    for gsize in () if os.environ.get("FIDELITY_RUNGS") else (4, 16):
        gpool = make_grouped_pool(heldout_pool, gsize, random.Random(31 + gsize))
        for rung in (8192, 32768):
            log(f"=== fidelity group_size={gsize} rung {rung} (meanpool only) ===")
            res = fidelity_for_rung(
                hf_model,
                tok,
                gpool,
                rung,
                layers,
                inv_freq,
                preamble_ids,
                g_theta,
                xstats,
                ystats,
                ridge_W,
                seed=777 + rung + gsize,
                variants=("meanpool",),
            )
            results["fidelity_by_group_size"][f"g{gsize}_rung{rung}"] = res
            mid = layers[len(layers) // 2]
            log(
                f"g{gsize} rung {rung} (mean chunk {res['mean_chunk_tokens']} tok) L{mid}: "
                f"cos {res['meanpool'][mid]['cos']:.3f} tv {res['meanpool'][mid]['tv']:.3f} "
                f"top {res['meanpool'][mid]['top']:.2f}"
            )
            with open(f"{OUT_JSON}/fidelity_metrics.json", "w") as f:
                json.dump({str(k): v for k, v in results.items()}, f, indent=1, default=str)

    # LAST (see the note inside): the eager flip can permanently poison sdpa dispatch.
    validate_attention_math(hf_model, tok, heldout_pool, inv_freq, preamble_ids, seed=123)
    log(f"done -> {OUT_JSON}/fidelity_metrics.json")


if __name__ == "__main__":
    main()
