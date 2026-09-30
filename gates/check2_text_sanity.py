"""Check 2: text-only CE of the 8T LM on its own pretraining data, forward-only, eval mode.

Usage:
    PYTHONPATH=<olmo-core>/src:<gates dir> python check2_text_sanity.py <out.json> [--files 0,100,...] [--seq 8192] [--windows 2]

Prints per-file CE and the token-weighted mean. Reference: the pretraining run logged train CE
1.711 (range 1.664 to 1.742) around step 476.9k on the same mixture.
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--files", default="0,150,300,450,600,750,900,1050")
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--windows", type=int, default=2)
    p.add_argument("--offset", type=int, default=1_000_000)
    p.add_argument("--emo", action="store_true", help="keep EMO routing (eval pool) instead of clearing it")
    args = p.parse_args()

    import olmo35_weights as W
    import te_shim

    te_shim.install()
    cfg = W.load_config()
    _, model = W.build_model(cfg, device="cuda", dtype=torch.bfloat16, emo=args.emo)
    W.set_router_groups(model, W.init_single_process_group())
    rep = W.load_main_weights(model)
    assert not rep["missing"] and not rep["mismatch"], rep
    model.eval()
    with open(W.CKPT + "/data_paths.txt") as f:
        paths = [line.strip() for line in f if line.strip()]
    results = []
    total_ce, total_tok = 0.0, 0
    for fi in [int(x) for x in args.files.split(",")]:
        t0 = time.time()
        ids = W.text_batch(cfg, batch_size=args.windows, seq_len=args.seq, file_index=fi, offset_tokens=args.offset)
        labels = torch.full_like(ids, -100)
        labels[:, :-1] = ids[:, 1:]
        n = int((labels != -100).sum())
        with torch.no_grad():
            out = model(ids, labels=labels, ignore_index=-100, loss_reduction="sum", loss_div_factor=n)
        ce = float(out.ce_loss)
        model.compute_auxiliary_metrics(reset=True)
        src = "/".join(paths[fi].split("/")[7:10])
        results.append({"file": fi, "source": src, "ce": ce, "tokens": n, "seconds": round(time.time() - t0, 1)})
        total_ce += ce * n
        total_tok += n
        print(f"file {fi:4d} {src:60s} CE {ce:.4f}")
    mean = total_ce / total_tok
    print(f"token-weighted mean CE over {len(results)} files: {mean:.4f}  (run logged 1.711 at this step)")
    json.dump({"emo": args.emo, "seq": args.seq, "windows": args.windows, "mean_ce": mean, "results": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
