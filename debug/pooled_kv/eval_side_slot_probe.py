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
from olmo_core.nn.attention.pooled_doc_kv import PooledDocKeepHolder, make_fingerprint_keep_docs_fn
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig

W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
IDS = RESERVED_IDS["qwen3_5"]
VOCAB = 248320
CKPT = {
    "contradiction": f"{W}/ctc_suite/ckpts/tsl-full-contradiction-s56M-4b-20260831T235942-0700",
    "oolong": f"{W}/ctc_suite/ckpts/tsl-full-oolong-s80M-4b-20260901T085004-0700",
}
EVAL_JSONL = {
    "contradiction": {"2k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl",
                      "8k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
                      "16k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n385_k3.jsonl",
                      "32k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n765_k3.jsonl"},
    "oolong": {"2k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl",
               "8k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl",
               "16k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx16384_spliteval.jsonl",
               "32k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx32768_spliteval.jsonl"},
}
CONV = {"contradiction": ("contradiction", "document"), "oolong": ("oolong", "line")}


def log(m):
    print(f"[probe] {m}", flush=True)


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
           "--out-dir", out_dir, "--emit", "dense", "--marker-set", "qwen3_5", "--tokenizer", "Qwen/Qwen3.5-0.8B-Base",
           "--seq-len", "65536", "--query-position", "after", "--cot-mode", "none", "--chunk-by", chunk, "--num-proc", "4"]
    if task == "contradiction":
        cmd.append("--emit-gold-sidecar")
    log("convert: " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONPATH="src", TOKENIZERS_PARALLELISM="false"))


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
    ap.add_argument("--task", default="contradiction", choices=["contradiction", "oolong"])
    ap.add_argument("--rung", default="32k")
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--keeps", default="0.3333,0.1667,0.0833,0")
    ap.add_argument("--extras", default="-2,-1,1,2", help="constant offsets c tried on top of +log L")
    ap.add_argument("--consts", default="-4,-2,-1,1", help="pure constant slot biases c (no log L term): is the optimum below zero?")
    ap.add_argument("--work", default="/results/probe_work")
    ap.add_argument("--out", default="/results/probe.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None, help="override: run dir (or model_and_optim) of the dense-trained checkpoint")
    ap.add_argument("--jsonl", default=None, help="override: held-out eval JSONL to tokenize (local cluster)")
    ap.add_argument("--shard", default=None, help="override: already-tokenized shard dir (skips conversion)")
    a = ap.parse_args()

    shard = a.shard or f"{a.work}/{a.task}_{a.rung}"
    if a.shard is None:
        convert(a.task, a.jsonl or EVAL_JSONL[a.task][a.rung], a.rows, shard)
    rows, masks = load_rows(shard, a.rows)
    log(f"{len(rows)} rows, lengths {[len(r) for r in rows[:6]]}...")

    cfg = TransformerConfig.qwen3_5_4B(vocab_size=VOCAB, attn_backend=AttentionBackendName.torch)
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    ck = find_ckpt(a.ckpt) if a.ckpt else find_ckpt(CKPT[a.task])
    t0 = time.time(); load_model_and_optim_state(ck, model); log(f"loaded {ck} in {time.time() - t0:.0f}s")
    model.enable_pooled_soft_tokens(IDS.doc_start, IDS.doc_end, IDS.eos, placeholder_id=IDS.landmark, keep_prob=0.0,
                                    keep_seed=a.seed, detach_soft_kv=True)
    model.pooled_projector.reset_parameters()  # identity: soft token == mean input embedding
    model = model.cuda().to(torch.bfloat16)
    pst = model._pooled_soft_tokens

    gold_table = json.load(open(f"{shard}/gold_fingerprints.json")) if a.task == "contradiction" else None
    keeps = [float(k) for k in a.keeps.split(",")]
    extras = [float(c) for c in a.extras.split(",")]
    configs = [("full", None, None)]
    for k in keeps:
        configs.append((f"soft k={k:.3f} no-bias", k, (False, 1.0, 0.0)))
        configs.append((f"soft k={k:.3f} +logL", k, (True, 1.0, 0.0)))
        for c in extras:
            configs.append((f"soft k={k:.3f} +logL{c:+.0f}", k, (True, 1.0, c)))
        configs.append((f"soft k={k:.3f} const=mean logL", k, (True, 0.0, float(np.log(45.0)))))
        for c in [float(c) for c in a.consts.split(",") if c]:
            configs.append((f"soft k={k:.3f} const{c:+.0f}", k, (True, 0.0, c)))

    res = {name: {"ce": [], "top1": [], "kl": [], "correct": [], "compaction": [], "sec": []} for name, _, _ in configs}
    full_cache = {}
    for ri, (row, rmask) in enumerate(zip(rows, masks)):
        x = torch.tensor(row[None], device="cuda")
        ans_pos = torch.tensor(np.nonzero(rmask)[0], device="cuda")  # positions of answer tokens
        pred_pos = ans_pos - 1  # logits predicting them
        targets = x[0, ans_pos]
        keep_cache = {}
        for name, keep, bias in configs:
            t_cfg = time.time()
            if keep is None:
                model.eval()
                # logits only at the answer-predicting positions (a full 34k x 248k logit tensor
                # was 34 GB and ~40 s per forward)
                lg = model(x, logits_to_keep=pred_pos[None])[0].float()
                full_cache[ri] = lg
                comp = 1.0
            else:
                model.train()
                if keep not in keep_cache:  # keep set once per (row, keep); reused across bias variants
                    if a.task == "contradiction":
                        keep_fn = make_fingerprint_keep_docs_fn(gold_table, doc_start_id=IDS.doc_start, doc_end_id=IDS.doc_end,
                                                                eos_id=IDS.eos, n_random_frac=keep, mode="gold_plus_random", seed=a.seed)
                        keep_cache[keep] = PooledDocKeepHolder(keep_docs=keep_fn(x.cpu()))
                    else:
                        keep_cache[keep] = None
                model._pooled_keep_holder = keep_cache[keep]
                if a.task != "contradiction":
                    pst["keep_prob"] = keep
                pst["len_bias"], pst["len_bias_scale"], pst["len_bias_extra"] = bias
                cb = model._compact_pooled_soft_tokens(x, None, -100)[0]
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
