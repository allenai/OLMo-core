"""Check 4: production kernels on a B300 (the text team's image: TE, grouped_gemm, flash_4, KDA kernels).

Runs the text team's exact model config (from the resolved mid-training config, EMO off, flash_4
on the two attention layers, experimental kernels on) with the 8T weights and checks:

(a) text CE on the same 8 pretraining windows as check 2 (seq 8192) -> compared with the local
    A100 torch-path numbers in check2_pin_seq8192.json; criterion |diff| <= 2e-3 per file.
(b) document isolation through the production path with ``doc_lens``/``max_doc_lens`` (flash_4
    varlen + KDA cu_seqlens), compile off and on: per-token CE of doc B packed after doc A vs
    doc B standalone; criterion mean |diff| < 0.01 and the no-boundary negative control leaks.

Usage (inside the job): PYTHONPATH=src:gates python gates/check4_production.py --out /results/check4.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch


def per_token_ce(model, ids, **kw):
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]
    with torch.no_grad():
        out = model(ids, labels=labels, ignore_index=-100, loss_reduction="none", **kw)
    return out.ce_loss.float()


def isolation(model, cfg, W, tag):
    seg = 1024
    a = W.text_batch(cfg, batch_size=1, seq_len=seg, file_index=150, offset_tokens=2_000_000)
    b = W.text_batch(cfg, batch_size=1, seq_len=seg, file_index=750, offset_tokens=2_000_000)
    packed = torch.cat([a, b], dim=1)
    ref_b = per_token_ce(model, b)[0]
    doc_lens = torch.tensor([[seg, seg]], device=packed.device)
    packed_iso = per_token_ce(model, packed, doc_lens=doc_lens, max_doc_lens=[seg])[0, seg:]
    packed_leak = per_token_ce(model, packed)[0, seg:]
    valid = slice(0, seg - 1)  # last token has no label
    iso = float((packed_iso[valid] - ref_b[valid]).abs().mean())
    leak = float((packed_leak[valid] - ref_b[valid]).abs().mean())
    res = {"tag": tag, "iso_mean_abs_diff": iso, "leak_mean_abs_diff": leak,
           "ref_b_ce": float(ref_b[valid].mean()), "packed_iso_ce": float(packed_iso[valid].mean()),
           "packed_leak_ce": float(packed_leak[valid].mean()), "pass": iso < 0.01 and leak > 10 * max(iso, 1e-4)}
    print(f"[isolation/{tag}] iso diff {iso:.5f} | leak diff {leak:.5f} | ref CE {res['ref_b_ce']:.4f} -> {'PASS' if res['pass'] else 'FAIL'}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/results/check4.json")
    p.add_argument("--text-config", default=os.path.join(os.path.dirname(__file__), "config_pin.json"))
    p.add_argument("--reference", default=os.path.join(os.path.dirname(__file__), "check2_pin_seq8192.json"))
    p.add_argument("--files", default="0,150,300,450,600,750,900,1050")
    p.add_argument("--seq", type=int, default=8192)
    p.add_argument("--windows", type=int, default=2)
    args = p.parse_args()

    import olmo35_weights as W
    from olmo_core.nn.transformer.config import OLMoDDPModelConfig

    text_cfg = json.load(open(args.text_config))["config"]
    model_cfg = OLMoDDPModelConfig.from_dict(text_cfg["model"])
    print("model config from text MT recipe: block overrides", sorted((model_cfg.block_overrides or {}).keys()))
    model = model_cfg.build(init_device="meta").to_empty(device="cuda").to(torch.bfloat16)
    W.set_router_groups(model, W.init_single_process_group())
    ckpt_cfg = W.load_config()
    rep = W.load_main_weights(model)
    assert not rep["missing"] and not rep["mismatch"], rep
    print("weights loaded:", {k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items()})
    model.eval()
    report = {"image": os.environ.get("BEAKER_IMAGE_ID"), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}

    # (a) text CE vs the local reference
    ref = {r["file"]: r["ce"] for r in json.load(open(args.reference))["results"]}
    rows, ok_a = [], True
    for fi in [int(x) for x in args.files.split(",")]:
        t0 = time.time()
        ids = W.text_batch(ckpt_cfg, batch_size=args.windows, seq_len=args.seq, file_index=fi)
        labels = torch.full_like(ids, -100)
        labels[:, :-1] = ids[:, 1:]
        n = int((labels != -100).sum())
        with torch.no_grad():
            out = model(ids, labels=labels, ignore_index=-100, loss_reduction="sum", loss_div_factor=n)
        ce = float(out.ce_loss)
        model.compute_auxiliary_metrics(reset=True)
        d = ce - ref[fi]
        ok_a &= abs(d) <= 2e-3
        rows.append({"file": fi, "ce": ce, "ref_ce_a100_torch": ref[fi], "diff": d, "seconds": round(time.time() - t0, 1)})
        print(f"[text CE] file {fi:4d}: {ce:.5f} vs A100/torch {ref[fi]:.5f} (diff {d:+.5f})")
    report["text_ce"] = {"rows": rows, "pass": ok_a, "criterion": "|diff| <= 2e-3"}
    print("[text CE]", "PASS" if ok_a else "FAIL")

    # (b) isolation, compile off then on
    report["isolation_eager"] = isolation(model, ckpt_cfg, W, "eager")
    model.apply_compile()
    report["isolation_compiled"] = isolation(model, ckpt_cfg, W, "compiled")
    report["pass"] = bool(ok_a and report["isolation_eager"]["pass"] and report["isolation_compiled"]["pass"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print("CHECK 4:", "PASSED" if report["pass"] else "FAILED", "->", args.out)


if __name__ == "__main__":
    main()
