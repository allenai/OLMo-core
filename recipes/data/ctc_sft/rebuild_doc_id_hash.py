"""
Rebuild a CTC SFT set's shards with hashed document ids (``--doc-id-style hash``), from the SAME
unified rows with the SAME converter arguments as the shipped build, into a new set root that holds
all of the set's tasks -- so ``launch_setA_sft.py --data-root <new root>`` trains on it unchanged.

Every task whose prompt shows document ids is re-tokenized in hash mode; every other task (oolong:
its items are unnumbered) is copied byte-for-byte. Nothing is symlinked.

The converter arguments are read from the source set's ``manifest.json`` (``convert``) and the
builder roster (``build_ctc_sft.SETS``: spec + chunk_by per task), exactly as ``_convert_one`` ran
them, and are proven to be the shipped ones before anything is written: the first ``--check-rows``
rows of every task are re-tokenized in ``seq`` mode and must reproduce the head of the shipped
shard byte for byte. The unified rows' sha256 must match the manifest too.

Verified per task after the rebuild (all printed, and written to ``<dst>/manifest.json``):

* same ``num_instances`` / ``num_dropped`` as the shipped shard, and the same number of document
  marker tokens per example (relabelling never moves a chunk boundary);
* the token delta per example (5-digit ids are longer than ``1..N``) -- never negative;
* the 32,768 cap of the 32k arm (``LongDocStrategy.exclude``: kept iff ``len <= 32768``): how many
  examples it keeps before and after, and how many cross it only because of that delta.

    python recipes/data/ctc_sft/rebuild_doc_id_hash.py \\
        --src-root /weka/.../ctc_sft_sets/setA_max20_evaliid \\
        --dst-root /weka/.../ctc_sft_sets/setA_max20_evaliid_hashid
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from glob import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[os.pardir] * 3))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

import build_ctc_sft as B  # noqa: E402

from olmo_core.data.corpus_reasoning_prompts.doc_id_hash import (  # noqa: E402
    SEED_VERSION,
    shows_doc_ids,
)
from olmo_core.data.document_chunk_landmark import reserved_ids  # noqa: E402

CONVERTER = os.path.join(REPO, "recipes", "data", "convert_unified_to_document_landmark.py")
CAP_32K = 32_768


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def task_shows_ids(spec: str, rows_path: str) -> bool:
    """Decided from the task's own first row, by the same function the converter uses."""
    with open(rows_path) as f:
        ex = json.loads(f.readline())
    if "ex" in ex and "documents" not in ex:
        ex = ex["ex"]
    return shows_doc_ids(ex, spec)


def converter_argv(task, conv: dict, rows: str, out: str, style: str, extra=()) -> list:
    """Exactly build_ctc_sft._convert_one's command, plus --doc-id-style."""
    argv = [sys.executable, CONVERTER, "--emit", "dense", "--task", task.spec,
            "--chunk-by", task.chunk_by, "--marker-set", conv["marker_set"],
            "--tokenizer", conv["tokenizer"], "--query-position", conv["query_position"],
            "--cot-mode", conv["cot_mode"], "--seq-len", str(conv["seq_len"]),
            "--input-jsonl", rows, "--out-dir", out, "--doc-id-style", style, *extra]
    if not conv.get("doc_markers", True):
        argv.append("--no-doc-markers")
    return argv


def run(argv: list, log_path: str) -> None:
    with open(log_path, "w") as f:
        p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        tail = open(log_path).read().strip().splitlines()[-5:]
        raise RuntimeError(f"converter failed (rc={p.returncode}): {' | '.join(tail)}")


def read_shard(d: str, eos: int, markers: tuple) -> dict:
    """Per-example lengths, loss tokens and marker counts of an EOS-terminated shard dir."""
    lens, loss, mark = [], [], []
    for tp in sorted(glob(os.path.join(d, "token_ids_part_*.npy"))):
        ids = np.fromfile(tp, dtype=np.uint32)
        mask = np.fromfile(tp.replace("token_ids_part_", "labels_mask_part_"), dtype=np.bool_)
        assert mask.size == ids.size, tp
        ends = np.flatnonzero(ids == eos)
        assert ends.size and ends[-1] == ids.size - 1, f"{tp}: not EOS-terminated"
        starts = np.concatenate([[0], ends[:-1] + 1])
        lens.append(ends - starts + 1)
        loss.append(np.add.reduceat(mask.astype(np.int64), starts))
        mark.append(np.add.reduceat(np.isin(ids, markers).astype(np.int64), starts))
    if not lens:
        z = np.zeros(0, dtype=np.int64)
        return {"len": z, "loss": z, "markers": z}
    return {"len": np.concatenate(lens), "loss": np.concatenate(loss),
            "markers": np.concatenate(mark)}


def seq_prefix_check(task, conv, rows, shipped_dir, work, n_rows) -> dict:
    """Re-tokenize the first n_rows in seq mode; must equal the head of the shipped shard."""
    out = os.path.join(work, f"seqcheck_{task.name}")
    os.makedirs(out, exist_ok=True)
    run(converter_argv(task, conv, rows, out, "seq", ["--limit", str(n_rows), "--num-proc", "4"]),
        os.path.join(work, f"seqcheck_{task.name}.log"))
    ok = True
    n_bytes = 0
    for name in ("token_ids_part_000000.npy", "labels_mask_part_000000.npy"):
        mine = open(os.path.join(out, name), "rb").read()
        with open(os.path.join(shipped_dir, name), "rb") as f:
            head = f.read(len(mine))
        ok &= mine == head
        n_bytes += len(mine)
    meta = json.load(open(os.path.join(out, "metadata.json")))
    return {"ok": bool(ok), "rows": n_rows, "instances": meta["num_instances"], "bytes": n_bytes}


def build_one(job) -> tuple:
    task, conv, rows, dst_dir, work, num_proc = job
    if os.path.exists(os.path.join(dst_dir, "metadata.json")):
        meta = json.load(open(os.path.join(dst_dir, "metadata.json")))
        if meta.get("doc_id_style") == "hash":
            return task.name, "cached"
        raise RuntimeError(f"{dst_dir} exists and is not a hash build")
    os.makedirs(dst_dir, exist_ok=True)
    t0 = time.time()
    run(converter_argv(task, conv, rows, dst_dir, "hash", ["--num-proc", str(num_proc)]),
        os.path.join(work, f"build_{task.name}.log"))
    return task.name, f"built in {time.time() - t0:.0f}s"


def compare(task: str, src_dir: str, dst_dir: str, ids) -> dict:
    sm = json.load(open(os.path.join(src_dir, "metadata.json")))
    dm = json.load(open(os.path.join(dst_dir, "metadata.json")))
    markers = (ids.doc_start, ids.doc_end)
    a = read_shard(src_dir, ids.eos, markers)
    b = read_shard(dst_dir, ids.eos, markers)
    errs = []
    if dm["num_instances"] != sm["num_instances"] or dm["num_dropped"] != sm["num_dropped"]:
        errs.append(f"instances {sm['num_instances']}/{sm['num_dropped']} dropped -> "
                    f"{dm['num_instances']}/{dm['num_dropped']}")
    if a["len"].size != b["len"].size:
        errs.append(f"EOS-split example count {a['len'].size} -> {b['len'].size}")
        return {"task": task, "errors": errs}
    if a["len"].size != sm["num_instances"]:
        errs.append(f"shipped shard holds {a['len'].size} EOS-split examples, metadata says "
                    f"{sm['num_instances']}")
    if not np.array_equal(a["markers"], b["markers"]):
        errs.append(f"marker tokens differ in {int((a['markers'] != b['markers']).sum())} examples")
    d = b["len"] - a["len"]
    dl = b["loss"] - a["loss"]
    if (d < 0).any():
        errs.append(f"{int((d < 0).sum())} examples got SHORTER")
    if dm.get("doc_id_style") != "hash":
        errs.append("metadata lacks doc_id_style=hash")
    old_keep, new_keep = a["len"] <= CAP_32K, b["len"] <= CAP_32K
    return {
        "task": task,
        "errors": errs,
        "num_instances": int(dm["num_instances"]),
        "num_dropped": int(dm["num_dropped"]),
        "tokens_old": int(a["len"].sum()),
        "tokens_new": int(b["len"].sum()),
        "token_delta": int(d.sum()),
        "token_delta_pct": round(100.0 * d.sum() / max(1, a["len"].sum()), 3),
        "delta_per_example": {"min": int(d.min()), "mean": round(float(d.mean()), 2),
                              "max": int(d.max())} if d.size else None,
        "loss_tokens_old": int(a["loss"].sum()),
        "loss_tokens_new": int(b["loss"].sum()),
        "loss_delta_per_example_mean": round(float(dl.mean()), 2) if dl.size else None,
        "max_example_len": [int(sm["max_example_len"]), int(dm["max_example_len"])],
        "kept_at_32k": [int(old_keep.sum()), int(new_keep.sum())],
        "kept_tokens_at_32k": [int(a["len"][old_keep].sum()), int(b["len"][new_keep].sum())],
        "newly_over_32k": int((old_keep & ~new_keep).sum()),
        "newly_under_32k": int((~old_keep & new_keep).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", required=True, help="the shipped set root (has manifest.json)")
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--shards-dir", default="shards_qwen35_256k")
    ap.add_argument("--set", dest="set_name", default="set-a", choices=sorted(B.SETS))
    ap.add_argument("--check-rows", type=int, default=300,
                    help="rows per task re-tokenized in seq mode against the shipped shard head")
    ap.add_argument("--jobs", type=int, default=0, help="tasks converted at once (0 = all)")
    ap.add_argument("--cpus", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.src_root, "manifest.json")))
    conv = manifest["convert"]
    log(f"source {args.src_root}: convert args {json.dumps(conv)}")
    tasks = {t.name: t for t in B.SETS[args.set_name][0]}
    src_shards = os.path.join(args.src_root, args.shards_dir)
    dst_shards = os.path.join(args.dst_root, args.shards_dir)
    work = os.path.join(args.dst_root, "_rebuild_logs")
    os.makedirs(work, exist_ok=True)
    ids = reserved_ids(conv["marker_set"])
    present = sorted(t for t in tasks if os.path.isdir(os.path.join(src_shards, t)))
    log(f"tasks in {src_shards}: {present}")

    # 1. inputs are the build's: rows hash to the manifest, and every shard's metadata agrees
    #    with the convert args (seq_len is not in metadata; the seq prefix check covers the rest)
    plan, bad = {}, []
    for name in present:
        t = tasks[name]
        rows = os.path.join(args.src_root, "per_task", name, "train.jsonl")
        want = manifest["per_task"][name]["sha256"]
        got = sha256(rows)
        meta = json.load(open(os.path.join(src_shards, name, "metadata.json")))
        for k, v in (("task", t.spec), ("chunk_by", t.chunk_by), ("tokenizer", conv["tokenizer"]),
                     ("marker_set", conv["marker_set"]), ("cot_mode", conv["cot_mode"]),
                     ("query_position", conv["query_position"]), ("emit", "dense"),
                     ("doc_markers", conv.get("doc_markers", True)), ("use_titles", False),
                     ("free_pad_repeat", 0)):
            if meta.get(k) != v:
                bad.append(f"{name}: shipped metadata {k}={meta.get(k)!r}, expected {v!r}")
        if got != want:
            bad.append(f"{name}: rows sha256 {got[:12]} != manifest {want[:12]}")
        plan[name] = "hash" if task_shows_ids(t.spec, rows) else "copy"
        log(f"  {name:<14} spec={t.spec:<13} rows ok={got == want}  -> {plan[name]}")
    if bad:
        raise SystemExit("INPUT CHECK FAILED:\n  " + "\n  ".join(bad))

    # 2. the convert args reproduce the shipped shards (seq mode, first rows, byte-exact)
    log(f"=== seq-mode prefix check ({args.check_rows} rows/task) ===")
    checks = {}
    with cf.ProcessPoolExecutor(max_workers=min(len(present), max(1, args.cpus // 4))) as pool:
        futs = {pool.submit(seq_prefix_check, tasks[n], conv,
                            os.path.join(args.src_root, "per_task", n, "train.jsonl"),
                            os.path.join(src_shards, n), work, args.check_rows): n
                for n in present}
        for f in cf.as_completed(futs):
            n = futs[f]
            checks[n] = f.result()
            log(f"  {n:<14} {checks[n]}")
    if not all(c["ok"] for c in checks.values()):
        raise SystemExit(f"SEQ PREFIX CHECK FAILED: {[n for n, c in checks.items() if not c['ok']]}")

    # 3. rebuild the id-bearing tasks in hash mode; copy the rest
    hash_tasks = [n for n in present if plan[n] == "hash"]
    for n in present:
        if plan[n] == "copy":
            d = os.path.join(dst_shards, n)
            if not os.path.exists(os.path.join(d, "metadata.json")):
                shutil.copytree(os.path.join(src_shards, n), d, dirs_exist_ok=True)
            log(f"  copied {n}")
    jobs = args.jobs or len(hash_tasks)
    per = max(1, args.cpus // jobs)
    log(f"=== hash rebuild: {hash_tasks}, {jobs} at a time, {per} procs each ===")
    with cf.ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(build_one, (tasks[n], conv,
                                        os.path.join(args.src_root, "per_task", n, "train.jsonl"),
                                        os.path.join(dst_shards, n), work, per)): n
                for n in hash_tasks}
        for f in cf.as_completed(futs):
            n, how = f.result()
            log(f"  {n}: {how}")

    # 4. verify against the shipped shards
    log("=== verify ===")
    report, errors = {}, []
    for n in present:
        src_dir, dst_dir = os.path.join(src_shards, n), os.path.join(dst_shards, n)
        if plan[n] == "copy":
            same = all(sha256(os.path.join(src_dir, f)) == sha256(os.path.join(dst_dir, f))
                       for f in os.listdir(src_dir))
            report[n] = {"task": n, "copied": True, "identical": same, "errors": [] if same else
                         ["copy differs"]}
        else:
            report[n] = compare(n, src_dir, dst_dir, ids)
        errors += [f"{n}: {e}" for e in report[n]["errors"]]
        log(f"REPORT {json.dumps(report[n])}")

    hdr = (f"{'task':<14}{'inst':>9}{'tok old':>15}{'tok new':>15}{'delta':>12}{'%':>7}"
           f"{'d/ex':>8}{'kept@32k old':>14}{'new':>9}{'crossed':>9}")
    print("\n" + hdr, flush=True)
    for n in present:
        r = report[n]
        if r.get("copied"):
            print(f"{n:<14}  (copied unchanged; identical={r['identical']})", flush=True)
            continue
        if "tokens_old" not in r:
            print(f"{n:<14}  {r['errors']}", flush=True)
            continue
        print(f"{n:<14}{r['num_instances']:>9,}{r['tokens_old']:>15,}{r['tokens_new']:>15,}"
              f"{r['token_delta']:>12,}{r['token_delta_pct']:>7.2f}"
              f"{r['delta_per_example']['mean']:>8.1f}{r['kept_at_32k'][0]:>14,}"
              f"{r['kept_at_32k'][1]:>9,}{r['newly_over_32k']:>9,}", flush=True)
    rebuilt = [report[n] for n in hash_tasks if "tokens_old" in report[n]]
    tot = {k: sum(r[k] for r in rebuilt) for k in ("tokens_old", "tokens_new", "token_delta",
                                                   "newly_over_32k", "newly_under_32k")}
    print(f"TOTAL rebuilt: {json.dumps(tot)}", flush=True)

    out = dict(manifest)
    out.update(
        tag=os.path.basename(os.path.normpath(args.dst_root)), root=args.dst_root,
        doc_id_hash={
            "src_root": args.src_root, "shards_dir": args.shards_dir, "seed_version": SEED_VERSION,
            "rebuilt": hash_tasks, "copied": [n for n in present if plan[n] == "copy"],
            "olmo_core_commit": B._git_commit(REPO), "seq_prefix_check": checks,
            "report": report, "totals": tot, "errors": errors,
        },
    )
    with open(os.path.join(args.dst_root, "manifest.json"), "w") as f:
        json.dump(out, f, indent=2)
    log(f"manifest -> {os.path.join(args.dst_root, 'manifest.json')}")
    if errors:
        raise SystemExit("VERIFY FAILED:\n  " + "\n  ".join(errors))
    log("HASHID_REBUILD_OK")


if __name__ == "__main__":
    main()
