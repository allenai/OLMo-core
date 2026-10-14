"""
Re-tokenize setA with the Qwen3 tokenizer, then cut it to the baseline 32k arm's example set.

Runs on Beaker (CPU, weka mounted): ``launch_setA_sft.py shards --model qwen3-4b``.

Steps:

1. **Full Qwen3 build.** Reads the set root's ``manifest.json`` and runs the SAME converter
   (``recipes/data/convert_unified_to_document_landmark.py``) with the SAME arguments as the shipped
   ``shards_qwen35_256k`` build (emit dense, spec / chunk_by per task from the manifest roster, query
   position, CoT mode, seq-len, document markers) on the SAME unified rows
   (``<root>/per_task/<task>/train.jsonl``). Only ``--tokenizer`` and ``--marker-set`` change (Qwen3:
   ``<|box_start|>``/``<|box_end|>`` = 151648/151649, EOS 151643). Output: ``<root>/shards_qwen3_256k/``.

2. **Row alignment and filter.** Every unified row is rendered again, in-process, with the converter's
   own :func:`tokenize_example` under both tokenizers, and each render is hashed. The Qwen3.5 renders
   must reproduce the SHIPPED ``shards_qwen35_256k`` shard example-for-example (token ids and label
   mask, sha1), and the Qwen3 renders the shard written in step 1. That proves the row order, the
   arguments and the code path, and it gives every row both lengths. A row is kept iff its Qwen3.5
   length (with EOS, as the packer counts it) is <= 32,768: exactly the examples the baseline 32k arm
   kept under ``LongDocStrategy.exclude``.

3. **Label-mask alignment** (kept rows only). The converter labels everything after the chat
   template's generation prompt. Qwen3.5-4B's generation prompt ends ``<think>\\n``, so the baseline
   labels start at ``</think>`` (``</think> \\n\\n <answer> <|im_end|> \\n``). Qwen3's generation
   prompt ends ``assistant\\n``, so the raw Qwen3 labels would start two tokens earlier, at
   ``<think> \\n\\n``. Those two tokens are unlabeled here, so the labeled span is the baseline's
   exactly. The token ids are unchanged. Asserted on every row of both builds.

Output: ``<root>/shards_qwen3_q35le32k/<task>/`` (``token_ids_part_*.npy``, ``labels_mask_part_*.npy``,
``metadata.json``, ``kept_rows.npy`` = row indices into ``per_task/<task>/train.jsonl``), and
``<root>/shards_qwen3_q35le32k/build_report.json``.
"""

import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
CONVERTER = os.path.join(REPO, "recipes", "data", "convert_unified_to_document_landmark.py")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from _setA_defaults import DEFAULT_SET_ROOT, DEFAULT_TASKS, parse_tasks  # noqa: E402

#: The baseline 32k arm's window, in Qwen3.5 tokens (example length incl. EOS).
CAP = 32_768
SHARD_TOKENS = 20_000_000  # the converter's default --shard-tokens


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_unified", CONVERTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------- workers
_W: Dict[str, Any] = {}


def _init(tok35: str, tok3: str, kwargs: Dict[str, Any]) -> None:
    from transformers import AutoTokenizer

    from olmo_core.data.document_chunk_landmark import reserved_ids

    _W["conv"] = _load_converter()
    _W["t35"] = AutoTokenizer.from_pretrained(tok35)
    _W["t3"] = AutoTokenizer.from_pretrained(tok3)
    _W["ids35"] = reserved_ids("qwen3_5")
    _W["ids3"] = reserved_ids("qwen3")
    _W["kw"] = kwargs


def _digest(ids: np.ndarray, mask: np.ndarray) -> str:
    h = hashlib.sha1(np.ascontiguousarray(ids, dtype=np.uint32).tobytes())
    h.update(np.ascontiguousarray(mask, dtype=np.bool_).tobytes())
    return h.hexdigest()


def _render(ex: dict) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
    conv, kw = _W["conv"], _W["kw"]
    out = []
    for tok, ids_set in ((_W["t35"], _W["ids35"]), (_W["t3"], _W["ids3"])):
        res = conv.tokenize_example(tok, ex, ids_set=ids_set, **kw)
        out += [None, None] if res is None else [len(res[0]), _digest(*res)]
    return tuple(out)  # type: ignore[return-value]


# ------------------------------------------------------------------------------------- shards
def read_shard(d: str, eos: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (ids, mask, ends) of a converter shard dir; ``ends`` = exclusive example ends."""
    ids, mask = [], []
    for p in sorted(os.listdir(d)):
        if p.startswith("token_ids_part_"):
            ids.append(np.fromfile(os.path.join(d, p), dtype=np.uint32))
            mp_ = os.path.join(d, p.replace("token_ids_part_", "labels_mask_part_"))
            mask.append(np.fromfile(mp_, dtype=np.bool_))
            assert ids[-1].size == mask[-1].size, f"{p}: ids/mask size mismatch"
    all_ids = np.concatenate(ids) if ids else np.zeros(0, np.uint32)
    all_mask = np.concatenate(mask) if mask else np.zeros(0, np.bool_)
    ends = np.flatnonzero(all_ids == eos) + 1
    assert ends.size == 0 or ends[-1] == all_ids.size, f"{d}: trailing tokens after the last EOS"
    return all_ids, all_mask, ends


def first_labels(mask: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """:returns: index of the first labeled token of every example (-1 if none)."""
    starts = np.concatenate([[0], ends[:-1]])
    out = np.full(ends.size, -1, dtype=np.int64)
    lab = np.flatnonzero(mask)
    if lab.size == 0:
        return out
    pos = np.searchsorted(lab, starts)
    ok = pos < lab.size
    cand = np.where(ok, lab[np.minimum(pos, lab.size - 1)], -1)
    inside = ok & (cand < ends)
    out[inside] = cand[inside]
    return out


def scaffold_check(ids: np.ndarray, starts_lab: np.ndarray, name: str,
                   expect: List[Tuple[int, int]]) -> None:
    """Assert ``ids[label_start + off] == want`` for every example and every ``(off, want)``."""
    bad = int((starts_lab < 0).sum())
    assert bad == 0, f"{name}: {bad} examples without any label"
    for off, want in expect:
        got = ids[starts_lab + off]
        n_bad = int((got != want).sum())
        assert n_bad == 0, f"{name}: {n_bad}/{got.size} examples differ at label-start{off:+d}"


def write_parts(out: str, ids: np.ndarray, mask: np.ndarray, ends: np.ndarray) -> int:
    """Write like the converter: a part is flushed once it holds >= SHARD_TOKENS tokens."""
    os.makedirs(out, exist_ok=True)
    part, buffered_from = 0, 0
    for e in ends:
        if e - buffered_from >= SHARD_TOKENS:
            ids[buffered_from:e].tofile(os.path.join(out, f"token_ids_part_{part:06d}.npy"))
            mask[buffered_from:e].tofile(os.path.join(out, f"labels_mask_part_{part:06d}.npy"))
            part += 1
            buffered_from = e
    if buffered_from < ids.size:
        ids[buffered_from:].tofile(os.path.join(out, f"token_ids_part_{part:06d}.npy"))
        mask[buffered_from:].tofile(os.path.join(out, f"labels_mask_part_{part:06d}.npy"))
        part += 1
    return part


# ------------------------------------------------------------------------------------- main
def run_converter(job) -> Tuple[str, int, str]:
    task, cmd, log = job
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    tail = open(log).read().strip().splitlines()[-3:]
    return task, rc, f"{time.time() - t0:.0f}s | " + " | ".join(tail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-root", default=DEFAULT_SET_ROOT)
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--full-dir", default="shards_qwen3_256k")
    ap.add_argument("--out-dir", default="shards_qwen3_q35le32k")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--convert-parallel", type=int, default=4,
                    help="converter processes at once (each uses --jobs // this workers)")
    ap.add_argument("--skip-convert", action="store_true",
                    help="reuse an existing full build (every task needs metadata.json)")
    args = ap.parse_args()
    tasks = parse_tasks(args.tasks)
    root = args.set_root.rstrip("/")
    t_start = time.time()

    from transformers import AutoTokenizer

    from olmo_core.data.document_chunk_landmark import reserved_ids

    man = json.load(open(f"{root}/manifest.json"))
    conv_args = man["convert"]
    print("MANIFEST convert", json.dumps(conv_args), "olmo_core_commit", man.get("olmo_core_commit"),
          flush=True)
    assert conv_args["marker_set"] == "qwen3_5", conv_args
    # The manifest predates the ``shards_dir`` / ``doc_markers`` fields: the build wrote
    # ``shards_qwen3_5`` (since renamed ``shards_qwen35_256k``, the dir the 32k/256k runs read) with
    # the converter's default document markers. Both are checked below: the Qwen3.5 re-render with
    # these arguments must reproduce that shard token for token.
    conv_args.setdefault("doc_markers", True)
    shipped = f"{root}/shards_qwen35_256k"
    for t in tasks:
        meta35 = json.load(open(f"{shipped}/{t}/metadata.json"))
        for k in ("tokenizer", "marker_set", "query_position", "cot_mode"):
            assert meta35.get(k) == conv_args[k], (t, k, meta35.get(k), conv_args[k])
        assert meta35.get("doc_markers", True) == conv_args["doc_markers"], (t, meta35)
    roster = {r["name"]: r for r in man["roster"]}
    ids35, ids3 = reserved_ids("qwen3_5"), reserved_ids("qwen3")
    kwargs = dict(
        task=None, emit="dense", n_summary_tokens=5, query_position=conv_args["query_position"],
        cot_mode=conv_args["cot_mode"], mem_freq=63, seq_len=int(conv_args["seq_len"]),
        chunk_by=None, item_regex=r"\|\|", use_titles=False, free_pad_repeat=0,
        repeat_doc_text=1, summary_every_k=0, doc_markers=bool(conv_args["doc_markers"]),
    )

    # ---- 1. full Qwen3 build with the shipped converter args ----
    full_root = f"{root}/{args.full_dir}"
    os.makedirs(full_root, exist_ok=True)
    jobs = []
    for task in tasks:
        r = roster[task]
        out = f"{full_root}/{task}"
        if args.skip_convert:
            assert os.path.exists(f"{out}/metadata.json"), f"--skip-convert: no build at {out}"
            continue
        if os.path.exists(f"{out}/metadata.json"):  # the converter writes it last: a finished build
            print(f"[1] reusing the finished build at {out}", flush=True)
            continue
        if os.path.exists(out) and os.listdir(out):
            raise SystemExit(f"{out} holds a partial build (no metadata.json); remove it first")
        cmd = [sys.executable, CONVERTER, "--emit", "dense", "--task", r["spec"],
               "--chunk-by", r["chunk_by"], "--marker-set", "qwen3", "--tokenizer", args.tokenizer,
               "--query-position", conv_args["query_position"], "--cot-mode",
               conv_args["cot_mode"], "--seq-len", str(conv_args["seq_len"]),
               "--num-proc", str(max(1, args.jobs // args.convert_parallel)),
               "--input-jsonl", f"{root}/per_task/{task}/train.jsonl", "--out-dir", out]
        if not conv_args["doc_markers"]:
            cmd.append("--no-doc-markers")
        jobs.append((task, cmd, f"/tmp/convert_{task}.log"))
    if jobs:
        print(f"=== [1/3] converter x{len(jobs)} -> {full_root} ===", flush=True)
        print("  e.g.", " ".join(jobs[0][1]), flush=True)
        with cf.ThreadPoolExecutor(args.convert_parallel) as pool:
            for task, rc, tail in pool.map(run_converter, jobs):
                print(f"  {task}: rc={rc} {tail}", flush=True)
                if rc:
                    raise SystemExit(f"converter failed on {task}; log /tmp/convert_{task}.log")

    # ---- 2+3. render both tokenizers per row, verify against both builds, filter, align ----
    t3 = AutoTokenizer.from_pretrained(args.tokenizer)
    t35 = AutoTokenizer.from_pretrained(conv_args["tokenizer"])
    sc3 = dict(think=t3.convert_tokens_to_ids("<think>"), end=t3.convert_tokens_to_ids("</think>"),
               nn=t3.encode("\n\n", add_special_tokens=False))
    sc35 = dict(think=t35.convert_tokens_to_ids("<think>"),
                end=t35.convert_tokens_to_ids("</think>"),
                nn=t35.encode("\n\n", add_special_tokens=False))
    assert len(sc3["nn"]) == 1 and len(sc35["nn"]) == 1, (sc3, sc35)
    sc3["nn"], sc35["nn"] = sc3["nn"][0], sc35["nn"][0]
    print(f"scaffold ids qwen3={sc3} qwen3.5={sc35}", flush=True)
    # Workers load the tokenizers from local copies: one pool per task x --jobs workers each
    # calling the Hub (11 x 32 x 2 requests) gets rate-limited (HTTP 429) part-way through.
    tok_dir = tempfile.mkdtemp(prefix="setA_tok_")
    tok3_local, tok35_local = f"{tok_dir}/qwen3", f"{tok_dir}/qwen35"
    t3.save_pretrained(tok3_local)
    t35.save_pretrained(tok35_local)

    conv = _load_converter()
    out_root = f"{root}/{args.out_dir}"
    report: Dict[str, Any] = {"cap_q35_tokens": CAP, "tokenizer": args.tokenizer,
                              "manifest_convert": conv_args, "tasks": {}}
    failures: List[str] = []
    for task in tasks:
        r = roster[task]
        kw = dict(kwargs, task=r["spec"], chunk_by=r["chunk_by"])
        rows = conv.iter_examples([f"{root}/per_task/{task}/train.jsonl"], 0)
        print(f"=== [2/3] {task}: {len(rows):,} rows (spec={r['spec']}, chunk_by={r['chunk_by']})",
              flush=True)
        # one pool per task: its spec / chunk_by are worker kwargs, set at init
        t0 = time.time()
        res = []
        with mp.Pool(args.jobs, initializer=_init,
                     initargs=(tok35_local, tok3_local, kw)) as pool:
            for i, x in enumerate(pool.imap(_render, rows, chunksize=4)):
                res.append(x)
                if i + 1 in (1, 2, 5, 10, 100) or (i + 1) % 5000 == 0:
                    el = time.time() - t0
                    print(f"  [{task}] {i + 1:,}/{len(rows):,} rendered, {el:.0f}s, "
                          f"eta {el / (i + 1) * (len(rows) - i - 1):.0f}s", flush=True)
        l35 = np.array([x[0] if x[0] is not None else -1 for x in res], dtype=np.int64)
        l3 = np.array([x[2] if x[2] is not None else -1 for x in res], dtype=np.int64)

        # verify against the shipped Qwen3.5 build and the fresh Qwen3 build, example by example
        info: Dict[str, Any] = {"rows": len(rows)}
        for name, d, eos, lens, dig_i in (
            ("qwen35", f"{shipped}/{task}", ids35.eos, l35, 1),
            ("qwen3", f"{full_root}/{task}", ids3.eos, l3, 3),
        ):
            ids, mask, ends = read_shard(d, eos)
            starts = np.concatenate([[0], ends[:-1]])
            present = np.flatnonzero(lens >= 0)
            assert present.size == ends.size, (
                f"{task}/{name}: {present.size} rows render under seq-len, shard has {ends.size}")
            assert np.array_equal(lens[present], ends - starts), f"{task}/{name}: length mismatch"
            bad_rows = [int(row) for k, row in enumerate(present)
                        if _digest(ids[starts[k]:ends[k]], mask[starts[k]:ends[k]])
                        != res[row][dig_i]]
            info[f"{name}_rerender_mismatches"] = len(bad_rows)
            if bad_rows:  # recorded, and fails the job at the end (lengths already agree)
                failures.append(f"{task}/{name}: {len(bad_rows)} examples differ from the "
                                f"re-render (rows {bad_rows[:5]})")
                print("  MISMATCH", failures[-1], flush=True)
            fl = first_labels(mask, ends)
            if name == "qwen35":  # baseline: labels start at </think> \n\n, after <think> \n\n
                try:
                    scaffold_check(ids, fl, f"{task}/{name}",
                                   [(-2, sc35["think"]), (-1, sc35["nn"]), (0, sc35["end"]),
                                    (1, sc35["nn"])])
                except AssertionError as exc:
                    failures.append(str(exc))
                    print("  SCAFFOLD", exc, flush=True)
            info[f"{name}_examples"] = int(ends.size)
            info[f"{name}_tokens"] = int(ids.size)
            if name == "qwen3":
                scaffold_check(ids, fl, f"{task}/{name}",
                               [(0, sc3["think"]), (1, sc3["nn"]), (2, sc3["end"]),
                                (3, sc3["nn"])])
                q3 = (ids, mask, starts, ends, fl, present)
        print(f"  [{task}] qwen35 shard vs re-render: {info['qwen35_examples']:,} ex, "
              f"{info['qwen35_rerender_mismatches']} differ; qwen3 shard vs re-render: "
              f"{info['qwen3_examples']:,} ex, {info['qwen3_rerender_mismatches']} differ",
              flush=True)

        # ---- filter + label alignment ----
        ids, mask, starts, ends, fl, present = q3
        keep_rows = np.flatnonzero((l35 >= 0) & (l35 <= CAP))
        assert np.all(l3[keep_rows] >= 0), f"{task}: a kept row is missing from the Qwen3 build"
        pos_in_shard = np.searchsorted(present, keep_rows)
        assert np.array_equal(present[pos_in_shard], keep_rows)
        mask = mask.copy()
        mask[fl[pos_in_shard]] = False  # <think>
        mask[fl[pos_in_shard] + 1] = False  # \n\n
        sel_ids = np.concatenate([ids[starts[k]:ends[k]] for k in pos_in_shard])
        sel_mask = np.concatenate([mask[starts[k]:ends[k]] for k in pos_in_shard])
        sel_lens = ends[pos_in_shard] - starts[pos_in_shard]
        sel_ends = np.cumsum(sel_lens)
        out = f"{out_root}/{task}"
        if os.path.exists(out) and os.listdir(out):
            raise SystemExit(f"{out} exists and is not empty")
        n_parts = write_parts(out, sel_ids, sel_mask, sel_ends)
        np.save(f"{out}/kept_rows.npy", keep_rows)
        meta = json.load(open(f"{full_root}/{task}/metadata.json"))
        meta.update(
            num_instances=int(keep_rows.size), num_dropped=int(len(rows) - keep_rows.size),
            num_tokens=int(sel_ids.size), num_loss_tokens=int(sel_mask.sum()),
            max_example_len=int(sel_lens.max()), min_example_len=int(sel_lens.min()),
            filtered_from=f"{full_root}/{task}",
            filter=f"keep a row iff its Qwen3.5 render ({conv_args['tokenizer']}, manifest args, "
                   f"incl. EOS) is <= {CAP} tokens: the baseline 32k arm's example set",
            label_mask_aligned="unlabeled <think> \\n\\n at the label start: labels begin at "
                               "</think>, as in the Qwen3.5 baseline shards",
        )
        json.dump(meta, open(f"{out}/metadata.json", "w"), indent=2)
        info.update(
            kept=int(keep_rows.size), kept_q35_tokens=int(l35[keep_rows].sum()),
            kept_q3_tokens=int(sel_ids.size), kept_q3_max_len=int(sel_lens.max()),
            kept_q35_max_len=int(l35[keep_rows].max()), loss_tokens=int(sel_mask.sum()),
            q3_over_q35_token_ratio=float(sel_ids.size / max(1, l35[keep_rows].sum())),
            q3_dropped_by_converter=int((l3 < 0).sum()), parts=n_parts,
        )
        report["tasks"][task] = info
        print("TASK", task, json.dumps(info), flush=True)

    tot = {k: sum(v[k] for v in report["tasks"].values())
           for k in ("rows", "kept", "kept_q35_tokens", "kept_q3_tokens", "loss_tokens")}
    tot["kept_q3_max_len"] = max(v["kept_q3_max_len"] for v in report["tasks"].values())
    tot["window"] = 1 << int(np.ceil(np.log2(tot["kept_q3_max_len"])))
    report["total"] = tot
    report["elapsed_s"] = round(time.time() - t_start)
    report["failures"] = failures
    json.dump(report, open(f"{out_root}/build_report.json", "w"), indent=2)
    print(f"\n{'task':<15}{'rows':>9}{'kept':>9}{'q35 tok':>15}{'q3 tok':>15}{'ratio':>7}"
          f"{'q3 max':>9}", flush=True)
    for t, v in report["tasks"].items():
        print(f"{t:<15}{v['rows']:>9,}{v['kept']:>9,}{v['kept_q35_tokens']:>15,}"
              f"{v['kept_q3_tokens']:>15,}{v['q3_over_q35_token_ratio']:>7.3f}"
              f"{v['kept_q3_max_len']:>9,}", flush=True)
    print("TOTAL", json.dumps(tot), flush=True)
    if failures:
        raise SystemExit("BUILD VERIFY FAILED:\n  " + "\n  ".join(failures))
    print(f"BUILD OK -> {out_root} ({report['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
