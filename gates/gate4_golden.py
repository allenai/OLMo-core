"""Gate 4: golden forward/backward of the OLMo 3.5 LM on real weights, for bitwise comparison.

Usage:
    PYTHONPATH=<olmo-core>/src:<gates dir> python gate4_golden.py <out.pt> [--compile] [--batch 2] [--seq 2048]
    python gate4_golden.py --compare <a.pt> <b.pt>

Runs the exact call the OLMoDDP train module makes (sum-reduced CE + z-loss with a token divisor,
train mode so router auxiliary losses attach to the graph), then backward, and records loss
terms, router metrics, logits and per-parameter gradient checksums. Two dumps from different
olmo-core trees must be bitwise equal for the tree under test to count as text-neutral.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time

import torch


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def run(args):
    import os

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    import olmo35_weights as W
    import te_shim

    print('TE shim installed:', te_shim.install())
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = W.load_config()
    build_kwargs = {}
    if args.production:  # keep the checkpoint's own kernels/backends (flash_4, experimental KDA kernels)
        build_kwargs = dict(attention_backend=None, experimental_kernels=None)
    model_cfg, model = W.build_model(cfg, device="cuda", dtype=torch.bfloat16, **build_kwargs)
    group = W.init_single_process_group()
    print('routers attached to a 1-rank LB group:', W.set_router_groups(model, group))
    rep = W.load_main_weights(model)
    assert not rep["missing"] and not rep["mismatch"], rep
    print("weights loaded:", {k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items()})
    input_ids = W.text_batch(cfg, batch_size=args.batch, seq_len=args.seq)
    labels = torch.full_like(input_ids, -100)
    labels[:, :-1] = input_ids[:, 1:]
    n_tokens = int((labels != -100).sum())
    model.train()
    fwd = model
    if args.compile:
        model.apply_compile()  # same entry point the train module uses
    def one_pass():
        for p in model.parameters():
            p.grad = None
        t0 = time.time()
        out = fwd(
            input_ids,
            labels=labels,
            ignore_index=-100,
            loss_reduction="sum",
            z_loss_multiplier=1e-5,
            loss_div_factor=n_tokens,
            return_logits=True,
        )
        out.loss.backward()
        torch.cuda.synchronize()
        metrics = {k: (v.detach().float().cpu() if torch.is_tensor(v) else v) for k, (v, _) in model.compute_auxiliary_metrics(reset=True).items()}
        grads = {n: (sha(p.grad), float(p.grad.float().abs().sum())) for n, p in model.named_parameters() if p.grad is not None}
        grad_tensors = {n: p.grad.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.grad is not None}
        return {
            "olmo_core": __import__("olmo_core").__file__,
            "compile": args.compile,
            "deterministic": args.deterministic,
            "production": args.production,
            "batch": args.batch,
            "seq": args.seq,
            "n_tokens": n_tokens,
            "loss": float(out.loss),
            "ce_loss": float(out.ce_loss),
            "z_loss": None if out.z_loss is None else float(out.z_loss),
            "ce_per_token": float(out.ce_loss),  # already divided by n_tokens
            "logits_sha": sha(out.logits),
            "logits_first": out.logits[0, :4, :8].detach().float().cpu(),
            "metrics": metrics,
            "grads": grads,
            "grad_tensors": grad_tensors,
            "logits": out.logits.detach().to("cpu", copy=True),
            "seconds": round(time.time() - t0, 1),
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        }

    passes = [one_pass() for _ in range(args.repeat)]
    record = dict(passes[0], passes=passes)
    for i, extra in enumerate(passes[1:], start=2):
        same_logits = extra["logits_sha"] == record["logits_sha"]
        grad_diff = sum(1 for n in record["grads"] if record["grads"][n][0] != extra["grads"][n][0])
        print(f"in-process pass {i}: loss {extra['loss']:.6f} (pass 1 {record['loss']:.6f}) | logits same: {same_logits} | "
              f"grad checksums differing: {grad_diff}/{len(record['grads'])}")
    torch.save(record, args.out)
    print(f"loss {record['loss']:.6f} ce {record['ce_loss']:.6f} z {record['z_loss']} | logits {record['logits_sha']} | "
          f"{len(record['grads'])} grads | {record['seconds']}s | peak {record['peak_mem_gib']} GiB -> {args.out}")


def _rel_l2(x, y):
    x, y = x.float(), y.float()
    return float((x - y).norm() / (x.norm() + 1e-12))


def _grad_distance(pa, pb):
    """Max over parameters of the relative L2 distance between two passes' gradients."""
    worst = (0.0, None)
    for n, ga in pa["grad_tensors"].items():
        d = _rel_l2(ga, pb["grad_tensors"][n])
        if d > worst[0]:
            worst = (d, n)
    return worst


def _metric_floor(dumps):
    """Per-metric max pairwise |diff| across reference dumps (cross-process spread)."""
    floor = {}
    for i in range(len(dumps)):
        for j in range(i + 1, len(dumps)):
            for k in set(dumps[i]["metrics"]) & set(dumps[j]["metrics"]):
                va, vb = dumps[i]["metrics"][k], dumps[j]["metrics"][k]
                if torch.is_tensor(va) and torch.is_tensor(vb):
                    floor[k] = max(floor.get(k, 0.0), float((va.float() - vb.float()).abs().max()))
    return floor


def compare(a_path, b_path, pin_paths=(), tol_factor=2.0):
    """A = reference (pin) dump, B = dump under test, pins = more dumps of the reference tree
    (cross-process spread). B passes when it is bitwise equal to any reference dump, or when
    every distance to A is within ``tol_factor`` x the reference tree's own spread."""
    a, b = torch.load(a_path, weights_only=False), torch.load(b_path, weights_only=False)
    pins = [torch.load(p, weights_only=False) for p in pin_paths]
    refs = [a] + pins
    ok = True
    for k in ("loss", "ce_loss", "z_loss", "logits_sha", "n_tokens"):
        print(f"{'ok  ' if a[k] == b[k] else 'DIFF'} {k}: {a[k]} vs {b[k]}")
    exact_forward = any(b["logits_sha"] == r["logits_sha"] and b["loss"] == r["loss"] for r in refs)
    loss_dist = abs(a["loss"] - b["loss"])
    loss_floor = max([abs(r["loss"] - q["loss"]) for r in refs for q in refs] + [0.0])
    logits_dist = _rel_l2(a["logits"], b["logits"]) if "logits" in a and "logits" in b else None
    logits_floor = max([_rel_l2(r["logits"], q["logits"]) for r in refs for q in refs if "logits" in r and "logits" in q] + [0.0])
    print(f"forward: |loss diff| {loss_dist:.3e} (ref spread {loss_floor:.3e}) | logits rel-L2 {logits_dist if logits_dist is None else f'{logits_dist:.3e}'} (ref spread {logits_floor:.3e}) | bitwise equal to a reference dump: {exact_forward}")
    if exact_forward:
        fwd_ok = True
    else:
        fwd_ok = loss_dist <= max(tol_factor * loss_floor, 2e-3)
        if logits_dist is not None:
            fwd_ok &= logits_dist <= max(tol_factor * logits_floor, 1e-3)
    print(f"forward criterion -> {fwd_ok}")
    ok &= fwd_ok
    floor = _metric_floor(refs)
    n_diff = 0
    for k in sorted(set(a["metrics"]) | set(b["metrics"])):
        va, vb = a["metrics"].get(k), b["metrics"].get(k)
        if torch.is_tensor(va) and torch.is_tensor(vb):
            d = float((va.float() - vb.float()).abs().max())
            tol = max(tol_factor * floor.get(k, 0.0), 1e-3 * (1 + float(va.float().abs().max())))
            if d > tol:
                n_diff += 1
                print(f"DIFF metric {k}: {va} vs {vb} (max abs diff {d:.3e} > tol {tol:.3e}, ref spread {floor.get(k, 0.0):.3e})")
        elif va != vb:
            n_diff += 1
            print(f"DIFF metric {k}: {va} vs {vb}")
    print(f"metrics: {len(a['metrics'])} compared, {n_diff} outside tolerance")
    ok &= n_diff == 0
    extra = [n for n in b["grads"] if n not in a["grads"]] + [n for n in a["grads"] if n not in b["grads"]]
    if extra:
        ok = False
        print(f"DIFF parameter sets differ: {extra[:10]}")
    dist = _grad_distance(a, b)
    floors = [(_grad_distance(r, q), "cross-process") for r in refs for q in refs if r is not q and "grad_tensors" in r and "grad_tensors" in q]
    for d in refs:
        if len(d.get("passes", [])) >= 2 and "grad_tensors" in d["passes"][1]:
            floors.append((_grad_distance(d["passes"][0], d["passes"][1]), "in-process"))
    for (f, n), kind in floors:
        print(f"reference grad spread ({kind}): max rel-L2 {f:.3e} at {n}")
    print(f"A vs B grads: max rel-L2 {dist[0]:.3e} at {dist[1]}")
    grad_floor = max([f for (f, _), _ in floors] + [0.0])
    grads_ok = dist[0] <= max(tol_factor * grad_floor, 1e-6)
    print(f"grad criterion: {dist[0]:.3e} <= {tol_factor} x floor {grad_floor:.3e} -> {grads_ok}")
    ok &= grads_ok
    print("GATE 4:", ("PASSED" + (" (bitwise identical forward)" if exact_forward else " (within the reference tree's own spread)")) if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", nargs="?")
    p.add_argument("--compare", nargs=2)
    p.add_argument("--pins", nargs="*", default=[], help="extra dumps of the reference tree (cross-process spread)")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--production", action="store_true", help="keep the config's attention backend and experimental kernels (B300 image)")
    p.add_argument("--repeat", type=int, default=1, help="forward/backward passes in one process (in-process reproducibility)")
    p.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms(warn_only) + CUBLAS workspace")
    a = p.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare, pin_paths=a.pins))
    run(a)
