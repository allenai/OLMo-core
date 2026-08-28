"""Teacher-forced held-out CE scorer: cheap per-rung proxy for the length-mix scaling laws.

Scores a trained checkpoint on the per-rung heldout shards (token_ids/labels_mask npy from
convert_unified_to_sft) WITHOUT generation: mean CE over answer tokens, per rung. Uses the model's
own fused-linear loss (loss_reduction='sum') so the 248k-vocab logits are never materialized.

For sparselandmark checkpoints the input stream is landmark-ified to match training: mem_id every
MEM_FREQ content tokens, tail block-padded with pad_id, labels False on inserted tokens.

    python score_heldout_ce.py --ckpt /data/.../ckpts/<run> --variant sparselandmark \
        --shards n14:/data/.../tokenized/n14_heldout n57:/data/.../tokenized/n57_heldout \
        --out ce.json
"""

import argparse
import json
import time

import numpy as np
import torch

from olmo_core.data.document_chunk_landmark import RESERVED_IDS
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer import TransformerConfig

MEM_FREQ = 63
BLOCK = MEM_FREQ + 1
IGNORE = -100


def split_examples(ids: np.ndarray, mask: np.ndarray, eos: int):
    ends = np.where(ids == eos)[0]
    out, start = [], 0
    for e in ends:
        out.append((ids[start : e + 1], mask[start : e + 1]))
        start = e + 1
    return out


def landmarkify(ids: np.ndarray, mask: np.ndarray, mem_id: int, pad_id: int):
    out_i, out_m = [], []
    for s in range(0, len(ids), MEM_FREQ):
        seg_i, seg_m = ids[s : s + MEM_FREQ], mask[s : s + MEM_FREQ]
        out_i.append(seg_i)
        out_m.append(seg_m)
        if len(seg_i) < MEM_FREQ:  # tail: pad content to a full block
            padn = MEM_FREQ - len(seg_i)
            out_i.append(np.full(padn, pad_id, dtype=ids.dtype))
            out_m.append(np.zeros(padn, dtype=bool))
        out_i.append(np.array([mem_id], dtype=ids.dtype))
        out_m.append(np.array([False]))
    return np.concatenate(out_i), np.concatenate(out_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run ROOT (config.json + model_and_optim)")
    ap.add_argument("--variant", choices=["full", "sparselandmark"], required=True)
    ap.add_argument("--shards", nargs="+", required=True, help="label:path pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-examples", type=int, default=300)
    ap.add_argument("--family", default="qwen3_5")
    args = ap.parse_args()

    ids_r = RESERVED_IDS[args.family]
    t0 = time.time()
    cfg = TransformerConfig.from_dict(json.load(open(f"{args.ckpt}/config.json"))["model"])
    model = cfg.build(init_device="cuda")
    load_model_and_optim_state(f"{args.ckpt}/model_and_optim", model)
    model = model.to(torch.bfloat16).eval()
    print(f"[ce] model loaded in {time.time()-t0:.1f}s", flush=True)

    results = {}
    for spec in args.shards:
        label, path = spec.split(":", 1)
        ids = np.fromfile(f"{path}/token_ids_part_000000.npy", dtype=np.uint32)
        mask = np.fromfile(f"{path}/labels_mask_000000.npy", dtype=bool)
        meta = json.load(open(f"{path}/metadata.json"))
        exs = split_examples(ids, mask, meta["eos"])[: args.max_examples]
        tot_loss, tot_tok, t1 = 0.0, 0, time.time()
        for i, (ei, em) in enumerate(exs):
            if args.variant == "sparselandmark":
                ei, em = landmarkify(ei, em, ids_r.landmark, ids_r.pad)
            x = torch.from_numpy(ei.astype(np.int64))[None].cuda()
            labels = torch.where(
                torch.from_numpy(em.copy())[None].cuda(), x, torch.full_like(x, IGNORE)
            )
            # match training: labels are input ids LEFT-SHIFTED (predict t+1 from prefix t);
            # see olmo_core.data.utils.get_labels.
            labels = torch.nn.functional.pad(labels[..., 1:], (0, 1), value=IGNORE)
            with torch.no_grad():
                out = model(x, labels=labels, loss_reduction="sum")
            loss = out.ce_loss if hasattr(out, "ce_loss") else out.loss
            n = int(em.sum())
            tot_loss += float(loss)
            tot_tok += n
            if i in (0, 1, 4, 9) or (i + 1) % 50 == 0:
                print(f"[ce:{label}] {i+1}/{len(exs)} ce_so_far={tot_loss/max(tot_tok,1):.4f} "
                      f"({time.time()-t1:.0f}s)", flush=True)
        results[label] = {"heldout_ce": tot_loss / max(tot_tok, 1), "eval_size": len(exs),
                          "answer_tokens": tot_tok}
        print(f"[ce:{label}] DONE ce={results[label]['heldout_ce']:.4f} "
              f"n={len(exs)} ({time.time()-t1:.0f}s)", flush=True)
    json.dump({"ckpt": args.ckpt, "variant": args.variant, "rungs": results},
              open(args.out, "w"), indent=2)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
