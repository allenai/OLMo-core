"""Output-level equivalence probe: how far is the COMPRESSED forward from the REAL forward,
through a TRAINED checkpoint, on the same rows?

For each row: forward once with real full attention, once with soft-token compression (same keep
set the trainer would draw), and compare AT ANSWER POSITIONS (aligned by original position):
  * per-layer hidden-state cosine (where does divergence start / explode),
  * final answer-token logit KL(full || compressed) and top-1 agreement.

This defines the gap a consistency-distillation loss must close, and which layers/positions carry
it. Run on the node holding the checkpoint, e.g.::

    srun -p jsteinhardt -w mooney --gres=gpu:H200:1 --qos=preemptive_high -t 40 \
      env PYTHONPATH=$PWD/src /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python \
      research/token_routing/pooled_kv/output_equivalence_probe.py \
      --ckpt /data/prasann/pooledkv_exp/runs/q4b-pkv32-full-e2 \
      --data /data/prasann/pooledkv_exp/contra32k_train --rows 16
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from olmo_core.data import TokenizerConfig
from olmo_core.data.document_chunk_landmark import (
    DOC_END_ID,
    DOC_START_ID,
    EOS_TOKEN_ID,
    LANDMARK_TOKEN_ID,
)
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.attention.pooled_doc_kv import (
    PooledDocKeepHolder,
    make_fingerprint_keep_docs_fn,
)
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig


def log(m):
    print(f"[oeq] {m}", flush=True)


def load_rows(shard, n_rows, seq_len):
    import glob

    ids_f = sorted(glob.glob(f"{shard}/token_ids_part_*.npy"))[0]
    mask_f = sorted(glob.glob(f"{shard}/labels_mask_*.npy"))[0]
    meta = json.load(open(f"{shard}/metadata.json"))
    # olmo-core shards are RAW binary memmaps (dtype in metadata), not numpy-format files. The
    # stream is EOS-terminated variable-length examples; split by scanning for EOS.
    ids = np.memmap(ids_f, dtype=np.dtype(meta["dtype"]), mode="r")
    mask = np.memmap(mask_f, dtype=np.dtype(meta.get("mask_dtype", "bool")), mode="r")
    rows, row_masks, start = [], [], 0
    eos = meta["eos_token_id"]
    i = 0
    while len(rows) < n_rows and i < len(ids):
        if ids[i] == eos:
            row = np.asarray(ids[start : i + 1], dtype=np.int64)
            if len(row) <= seq_len and mask[start : i + 1].any():
                rows.append(row)
                row_masks.append(np.asarray(mask[start : i + 1], dtype=bool))
            start = i + 1
        i += 1
    return rows, row_masks


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run dir with model_and_optim/")
    ap.add_argument("--data", required=True, help="tokenized shard dir")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--ckpt-has-projector", action="store_true")
    ap.add_argument("--n-gold", type=int, default=2)
    ap.add_argument("--n-random", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = TokenizerConfig.qwen3()
    cfg = TransformerConfig.qwen3_4B(vocab_size=tok.padded_vocab_size())
    cfg.lm_head.loss_implementation = LMLossImplementation.default
    model = cfg.build(init_device="cpu")
    if args.ckpt_has_projector:
        model.enable_pooled_soft_tokens(
            DOC_START_ID, DOC_END_ID, EOS_TOKEN_ID, placeholder_id=LANDMARK_TOKEN_ID
        )
        load_model_and_optim_state(f"{args.ckpt}/model_and_optim", model)
    else:
        load_model_and_optim_state(f"{args.ckpt}/model_and_optim", model)
        model.enable_pooled_soft_tokens(  # identity projector (P(x) == x)
            DOC_START_ID, DOC_END_ID, EOS_TOKEN_ID, placeholder_id=LANDMARK_TOKEN_ID
        )
    model = model.cuda().to(torch.bfloat16)
    log(f"loaded {args.ckpt} (projector={'ckpt' if args.ckpt_has_projector else 'identity'})")

    gold_table = json.load(open(f"{args.data}/gold_fingerprints.json"))
    keep_fn = make_fingerprint_keep_docs_fn(
        gold_table,
        doc_start_id=DOC_START_ID,
        doc_end_id=DOC_END_ID,
        eos_id=EOS_TOKEN_ID,
        n_random=args.n_random,
        mode="gold_subsample",
        n_gold=args.n_gold,
        seed=args.seed,
        debug_calls=0,
    )
    rows, row_masks = load_rows(args.data, args.rows, args.seq_len)
    log(f"{len(rows)} rows loaded")

    n_layers = model.n_layers
    hid_cos = {li: [] for li in range(n_layers)}
    kls, top1s, agree_docids = [], [], []

    # Hook every block to capture its output hidden state.
    captured = {}

    def mk_hook(li):
        def h(_m, _i, out):
            captured[li] = out.detach()

        return h

    hooks = [model.blocks[str(li)].register_forward_hook(mk_hook(li)) for li in range(n_layers)]

    for ri, (row, rmask) in enumerate(zip(rows, row_masks)):
        x = torch.tensor(row[None], device="cuda")
        # Positions whose LOGITS predict an answer token (answer token at p is predicted at p-1;
        # the first predictor is the last query token, which compaction always keeps).
        ans_pos = torch.tensor(np.nonzero(rmask)[0], device="cuda") - 1

        # ---- real forward (eval mode: compaction off) ----
        model.eval()
        captured.clear()
        logits_full = model(x)[0]  # (T, V)
        h_full = {li: captured[li][0, ans_pos].float() for li in range(n_layers)}
        lf = logits_full[ans_pos].float()
        del logits_full

        # ---- compressed forward (train-mode compaction, no grad) ----
        model.train()
        model._pooled_keep_holder = PooledDocKeepHolder(keep_docs=keep_fn(x.cpu()))
        cb, _ = model._compact_pooled_soft_tokens(x, None, -100)
        # map original answer positions -> compacted columns
        posmap = {int(p): c for c, p in enumerate(cb.position_ids[0].tolist())}
        cols = torch.tensor([posmap[int(p)] for p in ans_pos.tolist()], device="cuda")
        captured.clear()
        logits_comp = model(x)[0]  # forward recomputes the same compaction (same holder)
        h_comp = {li: captured[li][0, cols].float() for li in range(n_layers)}
        lc = logits_comp[cols].float()
        del logits_comp
        model.eval()

        for li in range(n_layers):
            hid_cos[li].append(float(F.cosine_similarity(h_full[li], h_comp[li], dim=-1).mean()))
        kls.append(
            float(
                F.kl_div(
                    F.log_softmax(lc, dim=-1),
                    F.log_softmax(lf, dim=-1),
                    log_target=True,
                    reduction="batchmean",
                )
            )
        )
        top1s.append(float((lf.argmax(-1) == lc.argmax(-1)).float().mean()))
        if ri + 1 in (1, 2, 5) or (ri + 1) % 8 == 0:
            log(
                f"row {ri + 1}/{len(rows)}: KL={kls[-1]:.3f} top1={top1s[-1]:.2f} "
                f"L0cos={hid_cos[0][-1]:.4f} Lmidcos={hid_cos[n_layers // 2][-1]:.4f} "
                f"Llastcos={hid_cos[n_layers - 1][-1]:.4f}"
            )

    for h in hooks:
        h.remove()
    res = {
        "ckpt": args.ckpt,
        "projector": "ckpt" if args.ckpt_has_projector else "identity",
        "rows": len(rows),
        "answer_logit_KL_mean": sum(kls) / len(kls),
        "answer_top1_agree_mean": sum(top1s) / len(top1s),
        "hidden_cos_by_layer": {li: sum(v) / len(v) for li, v in hid_cos.items()},
    }
    log("=== SUMMARY ===")
    log(f"answer logit KL(full||comp): {res['answer_logit_KL_mean']:.4f}")
    log(f"answer top-1 agreement:      {res['answer_top1_agree_mean']:.3f}")
    for li in range(0, n_layers, 4):
        log(f"  L{li:2d} hidden cos: {res['hidden_cos_by_layer'][li]:.4f}")
    log(f"  L{n_layers-1:2d} hidden cos: {res['hidden_cos_by_layer'][n_layers-1]:.4f}")
    out = args.out or f"research/token_routing/pooled_kv/probe_results/oeq_{os.path.basename(args.ckpt)}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
