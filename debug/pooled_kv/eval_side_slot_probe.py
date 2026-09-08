"""
Eval-side probe for the soft-token slot construction (Prasann, 2026-09-08): hold a model that was
trained to high accuracy with DENSE attention fixed, and ask which slot construction reproduces its
full-attention answer loss on held-out examples. Whatever is closest at eval is the most
in-distribution thing to train with.

For each held-out row (marker-tokenized on the fly from the eval JSONL with the same converter the
training shards used) and each configuration:
  * FULL: plain full attention over the real tokens (reference).
  * SOFT(keep, bias): pooled-doc compaction -- gold docs + a fraction ``keep`` of the others real,
    every other doc -> one soft token (identity projector = mean input embedding) at its centre
    position; ``bias`` in {none, +log L, +log L + c, constant}.
Metrics at the answer positions (aligned by original position): mean CE of the true answer
tokens, top-1 agreement with FULL's argmax, KL(FULL || SOFT), and exact-match of the greedy
teacher-forced answer against the true answer ("correct").

    python debug/pooled_kv/eval_side_slot_probe.py --task contradiction --rung 32k --rows 24 --out /results/probe.json
"""

import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from olmo_core.data.document_chunk_landmark import RESERVED_IDS
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder, make_fingerprint_keep_docs_fn
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig

W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
FAMILY = os.environ.get("PROBE_FAMILY", "qwen3_5")  # qwen3_5 (GDN hybrid) | qwen3 (pure attention)
IDS = RESERVED_IDS[FAMILY]

_GDN_SLOT_KEEP = {"mask": None}  # (B, T2) bool, True = real token; set per config when the GDN no-write variant runs


def install_gdn_nowrite_hook():
    """Make soft-token slots attention-only: GatedDeltaNet layers receive block_keep=~slot so a
    slot neither writes the recurrent state nor leaks through the causal conv (see
    GatedDeltaNet.forward). Attention layers are untouched (slots stay keys/values)."""
    from olmo_core.nn.attention.recurrent import GatedDeltaNet

    orig = GatedDeltaNet.forward

    def fwd(self, x, *args, **kw):
        m = _GDN_SLOT_KEEP["mask"]
        if m is not None and m.shape[1] == x.shape[1] and kw.get("block_keep") is None:
            kw["block_keep"] = m.to(x.device)
        return orig(self, x, *args, **kw)

    GatedDeltaNet.forward = fwd
    return orig
VOCAB = 248320 if FAMILY == "qwen3_5" else 151936
TOKENIZER = "Qwen/Qwen3.5-0.8B-Base" if FAMILY == "qwen3_5" else "Qwen/Qwen3-4B"
CKPT_Q3 = {  # dense Qwen3-4B ladder runs (lr 5e-5), pure attention: every layer takes K/V slots
    "contradiction": f"{W}/ctc_suite/ckpts/fs35q3s4bdense2-contradiction-dense-s56M",
    "oolong": f"{W}/ctc_suite/ckpts/fs35q3s4bdense2-oolong-dense-s80M",
}
CKPT = {  # dense-trained Qwen3.5-4B ladder runs (largest budget); globbed with * so any save-root suffix matches
    "contradiction": f"{W}/ctc_suite/ckpts/tsl-full-contradiction-s56M-4b-20260831T235942-0700",
    "oolong": f"{W}/ctc_suite/ckpts/tsl-full-oolong-s80M-4b-20260901T085004-0700",
    "nq": f"{W}/*/ckpts/lmx-full-nmixs48M-nq-4b",
    "outlier": f"{W}/*/ckpts/lmx-full-mixs160M-4b-2026",
}
if FAMILY == "qwen3":
    CKPT = CKPT_Q3
EVAL_JSONL = {
    "contradiction": {"2k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl",
                      "8k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
                      "16k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n385_k3.jsonl",
                      "32k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n765_k3.jsonl"},
    "nq": {r: f"{W}/outlier_lengthmix/eval_rungs/nq/rung_{n}.jsonl" for r, n in (("2k", 2048), ("8k", 8192), ("16k", 16384), ("32k", 32768))},
    "outlier": {r: f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_{n}.jsonl" for r, n in (("8k", 8192), ("16k", 16384), ("32k", 32768))},
    "oolong": {"2k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl",
               "8k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl",
               "16k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx16384_spliteval.jsonl",
               "32k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx32768_spliteval.jsonl"},
}
CONV = {"contradiction": ("contradiction", "document"), "oolong": ("oolong", "line"), "nq": ("retrieval", "document"), "outlier": ("outlier", "document")}
GOLD_TASKS = ("contradiction", "nq", "outlier")  # tasks with a gold sidecar (keep = gold + random fraction); oolong is gold-blind


def log(m):
    print(f"[probe] {m}", flush=True)


def build_cfg():
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig

    fac = TransformerConfig.qwen3_5_4B if FAMILY == "qwen3_5" else TransformerConfig.qwen3_4B
    return fac(vocab_size=VOCAB, attn_backend=AttentionBackendName.torch)


def find_ckpt(root):
    if root.endswith("model_and_optim") and os.path.isdir(root):
        return root
    cands = sorted(glob.glob(f"{root}*/model_and_optim") + glob.glob(f"{root}*/step*/model_and_optim"))
    if not cands:
        raise SystemExit(f"no model_and_optim under {root}*")
    return cands[-1]


def convert(task, jsonl, rows, out_dir):
    if os.path.exists(f"{out_dir}/metadata.json"):
        return
    os.makedirs(out_dir, exist_ok=True)
    head = f"{out_dir}/head.jsonl"
    with open(jsonl) as f, open(head, "w") as g:
        for i, line in enumerate(f):
            if i >= rows:
                break
            g.write(line)
    conv, chunk = CONV[task]
    cmd = ["python", "src/scripts/data/convert_unified_to_document_landmark.py", "--input-jsonl", head, "--task", conv,
           "--out-dir", out_dir, "--emit", "dense", "--marker-set", FAMILY, "--tokenizer", TOKENIZER,
           "--seq-len", "65536", "--query-position", "after", "--cot-mode", "none", "--chunk-by", chunk, "--num-proc", "4"]
    if task in GOLD_TASKS:
        cmd.append("--emit-gold-sidecar")
    log("convert: " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONPATH="src", TOKENIZERS_PARALLELISM="false"))


def policy_keep_mask(x, cid, gold, frac, policy, seed=0, hard_negs=None):
    """(1, n_docs) bool: gold docs plus a fraction ``frac`` of the others chosen by ``policy``."""
    n_docs = int(cid.max()) + 1
    non_gold = [d for d in range(n_docs) if not bool(gold[d]) and int((cid == d).sum()) > 0]
    k = int(round(frac * len(non_gold)))
    if policy in ("hardneg", "hardneg+rand"):
        hn = [d for d in (hard_negs or []) if d in non_gold]
        chosen = list(hn)
        if policy == "hardneg+rand":
            g = torch.Generator().manual_seed(seed)
            rest = [d for d in non_gold if d not in hn]
            order = torch.randperm(len(rest), generator=g).tolist()
            chosen += [rest[i] for i in order[:k]]
    elif policy == "random":
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(non_gold), generator=g).tolist()
        chosen = [non_gold[i] for i in order[:k]]
    else:
        last_doc_end = int((cid >= 0).nonzero(as_tuple=True)[0].max()) if int((cid >= 0).sum()) else -1
        q_tokens = set(x[0, last_doc_end + 1:].tolist())  # question + answer region (FREE tokens after the docs)
        scores = []
        for d in non_gold:
            toks = x[0, cid == d].tolist()
            if policy == "overlap":
                st = set(toks); scores.append(len(st & q_tokens) / max(1, len(st)))
            elif policy == "length":
                scores.append(float(len(toks)))
            elif policy == "short":
                scores.append(-float(len(toks)))
            elif policy == "first":
                scores.append(-float(int((cid == d).nonzero(as_tuple=True)[0][0])))
            elif policy == "last":
                scores.append(float(int((cid == d).nonzero(as_tuple=True)[0][0])))
            else:
                raise ValueError(policy)
        order = sorted(range(len(non_gold)), key=lambda i: -scores[i])
        chosen = [non_gold[i] for i in order[:k]]
    mask = gold.clone().bool()
    for d in chosen:
        mask[d] = True
    return mask[None]


def load_rows(shard, n_rows):
    ids_f = sorted(glob.glob(f"{shard}/token_ids_part_*.npy"))[0]
    mask_f = sorted(glob.glob(f"{shard}/labels_mask_*.npy"))[0]
    meta = json.load(open(f"{shard}/metadata.json"))
    ids = np.memmap(ids_f, dtype=np.dtype(meta["dtype"]), mode="r")
    mask = np.memmap(mask_f, dtype=np.dtype(meta.get("mask_dtype", "bool")), mode="r")
    eos = meta["eos_token_id"]
    rows, masks, start, i = [], [], 0, 0
    while len(rows) < n_rows and i < len(ids):
        if ids[i] == eos:
            row = np.asarray(ids[start:i + 1], dtype=np.int64)
            m = np.asarray(mask[start:i + 1], dtype=bool)
            if m.any():
                rows.append(row); masks.append(m)
            start = i + 1
        i += 1
    return rows, masks


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="contradiction", choices=["contradiction", "oolong", "nq", "outlier"])
    ap.add_argument("--rung", default="32k")
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--keeps", default="0.3333,0.1667,0.0833,0")
    ap.add_argument("--extras", default="-2,-1,1,2", help="constant offsets c tried on top of +log L")
    ap.add_argument("--consts", default="-4,-2,-1,1", help="pure constant slot biases c (no log L term): is the optimum below zero?")
    ap.add_argument("--policies", default="random", help="comma list of keep policies: random | overlap (top-k non-gold docs by token "
                    "overlap with the question) | length (longest) | short (shortest) | first (earliest) | last (latest)")
    ap.add_argument("--biases", default="all", help="'all' = the full bias sweep; 'none' = only the unbiased soft token per policy/keep")
    ap.add_argument("--gdn-nowrite", action="store_true", help="also run each keep with slots made attention-only (no GDN state write / conv leak)")
    ap.add_argument("--work", default="/results/probe_work")
    ap.add_argument("--out", default="/results/probe.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None, help="override: run dir (or model_and_optim) of the dense-trained checkpoint")
    ap.add_argument("--jsonl", default=None, help="override: held-out eval JSONL to tokenize (local cluster)")
    ap.add_argument("--shard", default=None, help="override: already-tokenized shard dir (skips conversion)")
    a = ap.parse_args()

    shard = a.shard or f"{a.work}/{FAMILY}_{a.task}_{a.rung}"
    if a.shard is None:
        convert(a.task, a.jsonl or EVAL_JSONL[a.task][a.rung], a.rows, shard)
    rows, masks = load_rows(shard, a.rows)
    log(f"{len(rows)} rows, lengths {[len(r) for r in rows[:6]]}...")
    hard_negs_rows = None
    head = f"{shard}/head.jsonl"
    if os.path.exists(head):
        hard_negs_rows = []
        for line in open(head):
            ex = json.loads(line)
            hn = ex.get("hard_neg_indices") or []
            hn = hn[0] if hn and isinstance(hn[0], list) else hn
            hard_negs_rows.append([int(d) for d in hn])
        n_hn = sum(len(h) for h in hard_negs_rows)
        log(f"hard negatives from {head}: {n_hn} across {len(hard_negs_rows)} rows")
        if n_hn == 0:
            hard_negs_rows = None

    cfg = build_cfg()
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    ck = find_ckpt(a.ckpt) if a.ckpt else find_ckpt(CKPT[a.task])
    t0 = time.time(); load_model_and_optim_state(ck, model); log(f"loaded {ck} in {time.time() - t0:.0f}s")
    model.enable_pooled_soft_tokens(IDS.doc_start, IDS.doc_end, IDS.eos, placeholder_id=IDS.landmark, keep_prob=0.0,
                                    keep_seed=a.seed, detach_soft_kv=True)
    model.pooled_projector.reset_parameters()  # identity: soft token == mean input embedding
    model = model.cuda().to(torch.bfloat16)
    pst = model._pooled_soft_tokens
    if a.gdn_nowrite:
        install_gdn_nowrite_hook()

    gold_table = json.load(open(f"{shard}/gold_fingerprints.json")) if a.task in GOLD_TASKS else None
    keeps = [float(k) for k in a.keeps.split(",")]
    extras = [float(c) for c in a.extras.split(",")]
    policies = a.policies.split(",")
    configs = [("full", None, None)]
    for pol in policies:
        tag = "" if pol == "random" else f" policy={pol}"
        for k in keeps:
            configs.append((f"soft k={k:.3f}{tag} no-bias", (k, pol), (False, 1.0, 0.0)))
            if a.gdn_nowrite:
                configs.append((f"soft k={k:.3f}{tag} gdn-nowrite", (k, pol), (False, 1.0, 0.0, "gdn-nowrite")))
            if a.biases == "none":
                continue
            configs.append((f"soft k={k:.3f}{tag} +logL", (k, pol), (True, 1.0, 0.0)))
            for c in extras:
                configs.append((f"soft k={k:.3f}{tag} +logL{c:+.0f}", (k, pol), (True, 1.0, c)))
            configs.append((f"soft k={k:.3f}{tag} const=mean logL", (k, pol), (True, 0.0, float(np.log(45.0)))))
            for c in [float(c) for c in a.consts.split(",") if c]:
                configs.append((f"soft k={k:.3f}{tag} const{c:+.0f}", (k, pol), (True, 0.0, c)))

    res = {name: {"ce": [], "top1": [], "kl": [], "correct": [], "compaction": [], "sec": []} for name, _, _ in configs}
    full_cache = {}
    for ri, (row, rmask) in enumerate(zip(rows, masks)):
        x = torch.tensor(row[None], device="cuda")
        ans_pos = torch.tensor(np.nonzero(rmask)[0], device="cuda")  # positions of answer tokens
        pred_pos = ans_pos - 1  # logits predicting them
        targets = x[0, ans_pos]
        keep_cache = {}
        cid_row = build_chunk_ids_from_tokens(x.cpu(), doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos, mode="chunked")[0]
        gold_row = None
        if a.task in GOLD_TASKS:
            gold_fn = make_fingerprint_keep_docs_fn(gold_table, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end, eos_id=IDS.eos,
                                                    n_random_frac=0.0, mode="gold_plus_random", seed=a.seed)
            gold_row = gold_fn(x.cpu())[0].bool()
        else:
            gold_row = torch.zeros(int(cid_row.max()) + 1, dtype=torch.bool)
        for name, keep, bias in configs:
            t_cfg = time.time()
            if keep is not None and isinstance(keep, tuple):
                keep, pol = keep
            else:
                pol = "random"
            if keep is None:
                model.eval()
                # logits only at the answer-predicting positions (a full 34k x 248k logit tensor
                # was 34 GB and ~40 s per forward)
                lg = model(x, logits_to_keep=pred_pos[None])[0].float()
                full_cache[ri] = lg
                comp = 1.0
            else:
                model.train()
                ck_ = (keep, pol)
                if ck_ not in keep_cache:  # keep set once per (row, keep, policy); reused across bias variants
                    if pol == "random" and a.task in GOLD_TASKS:
                        keep_fn = make_fingerprint_keep_docs_fn(gold_table, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end,
                                                                eos_id=IDS.eos, n_random_frac=keep, mode="gold_plus_random", seed=a.seed)
                        keep_cache[ck_] = PooledDocKeepHolder(keep_docs=keep_fn(x.cpu()))
                    elif pol == "random":
                        keep_cache[ck_] = None
                    else:
                        keep_cache[ck_] = PooledDocKeepHolder(keep_docs=policy_keep_mask(x.cpu(), cid_row, gold_row, keep, pol, a.seed, hard_negs=hard_negs_rows[ri] if hard_negs_rows else None))
                    if ri == 0 and keep_cache[ck_] is not None:
                        kd = keep_cache[ck_].keep_docs[0]
                        log(f"keep set ({pol}, {keep:.3f}): {int(kd.sum())}/{kd.numel()} docs real, first kept: {kd.nonzero(as_tuple=True)[0][:8].tolist()}")
                model._pooled_keep_holder = keep_cache[ck_]
                if pol == "random" and a.task not in GOLD_TASKS:
                    pst["keep_prob"] = keep
                pst["len_bias"], pst["len_bias_scale"], pst["len_bias_extra"] = bias[:3]
                cb = model._compact_pooled_soft_tokens(x, None, -100)[0]
                if len(bias) > 3 and bias[3] == "gdn-nowrite":
                    m_keep = torch.ones_like(cb.input_ids, dtype=torch.bool)
                    m_keep[cb.soft_rows, cb.soft_cols] = False
                    _GDN_SLOT_KEEP["mask"] = m_keep
                else:
                    _GDN_SLOT_KEEP["mask"] = None
                posmap = {int(p): c for c, p in enumerate(cb.position_ids[0].tolist())}
                cols = torch.tensor([posmap[int(p)] for p in pred_pos.tolist()], device="cuda")
                lg = model(x, logits_to_keep=cols[None])[0].float()
                comp = cb.input_ids.shape[1] / x.shape[1]
                model.eval()
            lf = full_cache[ri]
            ce = float(F.cross_entropy(lg, targets))
            top1 = float((lg.argmax(-1) == lf.argmax(-1)).float().mean())
            kl = float(F.kl_div(F.log_softmax(lg, -1), F.log_softmax(lf, -1), log_target=True, reduction="batchmean"))
            correct = float((lg.argmax(-1) == targets).all())
            r = res[name]; r["ce"].append(ce); r["top1"].append(top1); r["kl"].append(kl); r["correct"].append(correct); r["compaction"].append(comp)
            r.setdefault("sec", []).append(time.time() - t_cfg)
        if ri + 1 in (1, 2, 5) or (ri + 1) % 8 == 0:
            log(f"row {ri + 1}/{len(rows)} done; full CE {res['full']['ce'][-1]:.3f} correct {res['full']['correct'][-1]:.0f}")
            summary(res, configs)
    summary(res, configs)
    out = {"task": a.task, "rung": a.rung, "rows": len(rows), "ckpt": ck,
           "configs": {n: {k: float(np.mean(v)) for k, v in res[n].items()} for n, _, _ in configs}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"wrote {a.out}")


def summary(res, configs):
    print(f"{'config':32} {'answer CE':>9} {'top1=full':>9} {'KL':>7} {'correct':>8} {'compact':>8} {'s/row':>6}", flush=True)
    for name, _, _ in configs:
        r = res[name]
        if not r["ce"]:
            continue
        print(f"{name:32} {np.mean(r['ce']):9.3f} {np.mean(r['top1']):9.3f} {np.mean(r['kl']):7.3f} {np.mean(r['correct']):8.2f} {np.mean(r['compaction']):8.3f} {np.mean(r['sec']):6.1f}", flush=True)


if __name__ == "__main__":
    main()
