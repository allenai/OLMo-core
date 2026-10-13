"""Results log: append-only ``state/results.jsonl``, one row per (run, task, rung, metric) number.

Python API: :func:`add`, :func:`load`, :func:`check`, :func:`rows_from_json` (accepts contract rows,
``results_io`` §8 records and ``*_multirung.json`` eval outputs).
CLI: ``ops result add|check|show``.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

from opslib import paths
from opslib.registry import append_rows, format_table, guess_experiment, load_runs, now_iso, read_jsonl

FIELDS = (
    "run_id", "experiment", "task", "rung", "metric", "value", "eval_size", "se", "cot_label",
    "eval_backend", "ckpt_path", "git_commit", "eval_job_id", "source_json", "recorded_at",
    "recorded_by", "flags", "notes",
)
REQUIRED = ("run_id", "task", "metric", "value", "eval_size")
PROVENANCE = ("ckpt_path", "git_commit", "eval_job_id")
COT_LABELS = ("long-cot", "short-cot", "no-cot")
MIN_EVAL_SIZE = 500
SE_MULT = 3.0
ABS_TOL = 0.1  # deviation threshold when either side has no SE


def results_path(path=None) -> Path:
    """Resolve results.jsonl at call time (honours ``OPS_STATE_DIR``)."""
    if path:
        return Path(path)
    return Path(os.environ.get("OPS_STATE_DIR", paths.STATE_DIR)) / "results.jsonl"


def binomial_se(p: float, eval_size: int) -> float:
    """``sqrt(p(1-p)/eval_size)``."""
    return math.sqrt(p * (1.0 - p) / eval_size)


def rung_key(rung) -> float:
    """Sort key for rung labels (``2k``, ``rung_16384``, 2048, ``1M``)."""
    if rung is None:
        return -1
    s = str(rung).lower().replace("rung_", "").replace("r", "")
    m = re.fullmatch(r"([\d.]+)([km]?)", s)
    if not m:
        return float("inf")
    return float(m.group(1)) * {"": 1, "k": 1024, "m": 1024 * 1024}[m.group(2)]


def normalize(fields: dict, backfill: bool = False) -> dict:
    """Validate + fill defaults (se, flags, experiment, recorded_*); raises ``ValueError``."""
    if "n" in fields:
        raise ValueError("field 'n' is reserved for CORPUS size; use 'eval_size' for the eval-set size")
    unknown = sorted(set(fields) - set(FIELDS))
    if unknown:
        raise ValueError(f"unknown fields {unknown}; allowed: {', '.join(FIELDS)}")
    row = {k: v for k, v in fields.items() if v is not None}
    required = REQUIRED if not backfill else tuple(k for k in REQUIRED if k != "eval_size")
    missing = [k for k in required if row.get(k) in (None, "")]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    try:
        row["value"] = float(row["value"])
    except (TypeError, ValueError):
        raise ValueError(f"value must be numeric, got {row['value']!r}")
    flags = list(row.get("flags") or [])
    if row.get("eval_size") is not None:
        try:
            row["eval_size"] = int(row["eval_size"])
        except (TypeError, ValueError):
            raise ValueError(f"eval_size must be an int, got {row['eval_size']!r}")
        if row["eval_size"] <= 0:
            raise ValueError("eval_size must be > 0")
        if row["eval_size"] < MIN_EVAL_SIZE:
            flags.append(f"eval_size<{MIN_EVAL_SIZE}")
        if row.get("se") is None and 0.0 <= row["value"] <= 1.0:
            row["se"] = round(binomial_se(row["value"], row["eval_size"]), 6)
    else:
        row["eval_size"] = None
        flags.append("eval_size_missing")
    if row.get("cot_label") is not None and row["cot_label"] not in COT_LABELS:
        raise ValueError(f"cot_label must be one of {COT_LABELS}")
    if backfill:
        flags.append("backfill")
        if any(not row.get(k) for k in PROVENANCE):
            flags.append("provenance_incomplete")
    row["flags"] = sorted(set(flags))
    if not row.get("experiment"):
        run = load_runs().get(row["run_id"]) if not backfill else None
        row["experiment"] = (run or {}).get("experiment") or guess_experiment(row["run_id"])
    row.setdefault("experiment", None)
    row.setdefault("recorded_at", now_iso())
    row.setdefault("recorded_by", "backfill" if backfill else (os.environ.get("CLAUDE_SESSION_ID") or "manual"))
    return row


def add(_path=None, backfill: bool = False, **fields) -> dict:
    """Validate and append one result row; returns the stored row."""
    row = normalize(fields, backfill=backfill)
    append_rows(results_path(_path), [row])
    return row


def load(path=None) -> list:
    """All result rows in file order."""
    return read_jsonl(results_path(path))


def _key(r):
    return (r.get("run_id"), r.get("task"), str(r.get("rung")), r.get("metric"))


def _deviation(new, ref):
    """``(threshold, label)`` for comparing two rows: 3 combined SE, or 0.1 abs without SE."""
    if new.get("se") is not None and ref.get("se") is not None:
        return SE_MULT * math.hypot(new["se"], ref["se"]), f"{SE_MULT:g} combined SE"
    return ABS_TOL, f"{ABS_TOL} abs (no SE)"


def check(row: dict, prior=None, path=None) -> list:
    """Warnings for ``row`` against prior rows (default: everything in results.jsonl).

    (a) duplicate run/task/rung/metric; (b) deviation: vs the most recent prior eval of the SAME
    run (re-eval drift), else vs the experiment's best for task/rung/metric (a new best beyond noise);
    (c) provenance gaps; (d) value exactly 0.0 or 1.0.
    """
    prior = load(path) if prior is None else prior
    warns = []
    same = [r for r in prior if _key(r) == _key(row)]
    if same:
        warns.append(f"duplicate: {len(same)} prior row(s) for run/task/rung/metric {_key(row)}")
    value = float(row["value"])
    if same:
        ref, what = same[-1], "most recent prior eval of this run"
    else:
        peers = [
            r for r in prior
            if r.get("experiment") == row.get("experiment") and r.get("experiment")
            and (r.get("task"), str(r.get("rung")), r.get("metric")) == (row.get("task"), str(row.get("rung")), row.get("metric"))
        ]
        ref = max(peers, key=lambda r: r["value"]) if peers else None
        what = "experiment best"
        if ref is not None and value <= ref["value"]:
            ref = None  # worse than the best arm is normal, not suspicious
    if ref is not None:
        tol, label = _deviation(row, ref)
        if abs(value - ref["value"]) > tol:
            warns.append(
                f"deviation: {value:.4f} vs {what} {ref['value']:.4f} ({ref.get('run_id')}) "
                f"differs by {abs(value - ref['value']):.4f} > {label} -- explain before quoting"
            )
    gaps = [k for k in PROVENANCE if not row.get(k)]
    if gaps:
        warns.append(f"provenance gap: missing {', '.join(gaps)}")
    if value in (0.0, 1.0):
        warns.append(f"value is exactly {value:.4f}: exactly 0.0000 is a bug signature -- check generation outputs")
    if row.get("eval_size") is None:
        warns.append("eval_size missing: cannot attach an error bar")
    elif row["eval_size"] < MIN_EVAL_SIZE:
        se = row.get("se")
        warns.append(f"eval_size={row['eval_size']} only" + (f" (+-{se:.3f})" if se is not None else "") + " -- flag inline")
    return warns


# ----------------------------------------------------------------- input shapes

_MULTIRUNG_KEY = re.compile(r"^([a-z][a-z0-9_]*?)_(\d+(?:\.\d+)?[kKmM])$")


def _run_from_ckpt(path):
    """Run name from a checkpoint path (drops trailing ``stepN`` / ``model_and_optim``)."""
    parts = [p for p in str(path).rstrip("/").split("/") if p]
    while parts and (re.fullmatch(r"step\d+(-unsharded)?", parts[-1]) or parts[-1] in ("model_and_optim", "hf")):
        parts.pop()
    return parts[-1] if parts else None


def rows_from_json(obj, defaults=None) -> list:
    """Expand one parsed JSON value into contract field dicts (not yet validated).

    Accepts: a list of any of these; a contract row (``metric``+``value``); a ``results_io`` §8
    record (``metric_name``/``metric_value``/``rung_tokens``/``provenance``); a native
    ``*_multirung.json`` (``model_path`` + ``<task>_<rung>: float`` keys; no eval_size inside).
    """
    defaults = dict(defaults or {})
    if isinstance(obj, list):
        return [r for o in obj for r in rows_from_json(o, defaults)]
    if not isinstance(obj, dict):
        raise ValueError(f"unsupported JSON value of type {type(obj).__name__}")
    if "metric" in obj and "value" in obj:
        return [{**defaults, **obj}]
    if "metric_name" in obj and "metric_value" in obj:
        prov = obj.get("provenance") or {}
        row = {
            "run_id": obj.get("run_id") or _run_from_ckpt(prov.get("ckpt_path", "")),
            "task": obj.get("task"), "rung": obj.get("rung_tokens"), "metric": obj["metric_name"],
            "value": obj["metric_value"], "eval_size": obj.get("eval_size"), "cot_label": obj.get("cot_label"),
            "eval_backend": prov.get("eval_backend"), "ckpt_path": prov.get("ckpt_path"),
            "git_commit": prov.get("git_commit"), "eval_job_id": prov.get("eval_job_id"),
        }
        return [{**row, **{k: v for k, v in defaults.items() if v is not None}}]
    if "model_path" in obj:
        base = {"run_id": _run_from_ckpt(obj["model_path"]), "ckpt_path": obj["model_path"], "metric": "f1",
                "eval_backend": "native"}
        base.update({k: v for k, v in defaults.items() if v is not None})
        out = []
        for k, v in obj.items():
            m = _MULTIRUNG_KEY.match(k)
            if m and isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append({**base, "task": m.group(1), "rung": m.group(2), "value": v})
        return out
    raise ValueError("unrecognised JSON shape (need metric/value, metric_name/metric_value, or model_path)")


def load_json_file(path, defaults=None) -> list:
    """Read a .json (object or list) or .jsonl file and expand it with :func:`rows_from_json`."""
    text = Path(path).read_text()
    defaults = {"source_json": str(Path(path).resolve()), **(defaults or {})}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows_from_json(obj, defaults)


# ----------------------------------------------------------------------------- CLI

SHOW_COLS = ("run_id", "task", "rung", "metric", "value", "se", "eval_size", "flags")


def _cli_rows(args) -> list:
    fields = {k: getattr(args, k) for k in ("run_id", "experiment", "task", "rung", "metric", "value",
                                            "eval_size", "cot_label", "eval_backend", "ckpt_path",
                                            "git_commit", "eval_job_id") if getattr(args, k) is not None}
    if args.json_file:
        return load_json_file(args.json_file, defaults=fields)
    return [fields]


def _cmd_add(args):
    prior = load()
    stored, bad = [], 0
    for fields in _cli_rows(args):
        try:
            row = normalize(fields)
        except ValueError as e:
            print(f"REJECTED {fields.get('run_id')}/{fields.get('task')}/{fields.get('rung')}: {e}", file=sys.stderr)
            bad += 1
            continue
        if args.check:
            for w in check(row, prior=prior + stored):
                print(f"WARN {row['run_id']} {row['task']} {row.get('rung')}: {w}", file=sys.stderr)
        stored.append(row)
    append_rows(results_path(), stored)
    print(json.dumps(stored, indent=1) if args.json else f"added {len(stored)} result row(s)" + (f", rejected {bad}" if bad else ""))
    return 1 if bad else 0


def _cmd_check(args):
    prior, n_warn = load(), 0
    for fields in _cli_rows(args):
        try:
            row = normalize(fields)
        except ValueError as e:
            print(f"INVALID: {e}")
            n_warn += 1
            continue
        warns = check(row, prior=prior)
        n_warn += len(warns)
        print(f"{row['run_id']} {row['task']} {row.get('rung')} {row['metric']}={row['value']:.4f}: "
              + ("OK" if not warns else "\n  " + "\n  ".join(warns)))
    return 1 if n_warn else 0


def _cmd_show(args):
    rows = [r for r in load()
            if (not args.experiment or r.get("experiment") == args.experiment)
            and (not args.task or r.get("task") == args.task)
            and (not args.run_id or r.get("run_id") == args.run_id)]
    if args.best:
        best = {}
        for r in rows:
            k = (r.get("task"), str(r.get("rung")), r.get("metric"))
            if k not in best or r["value"] > best[k]["value"]:
                best[k] = r
        rows = list(best.values())
    rows.sort(key=lambda r: (r.get("task") or "", rung_key(r.get("rung")), r.get("metric") or "", r.get("run_id") or ""))
    if args.json:
        print(json.dumps(rows, indent=1))
        return
    table = [{**r, "value": f"{r['value']:.4f}", "se": None if r.get("se") is None else f"{r['se']:.3f}",
              "flags": ",".join(r.get("flags") or []) or None} for r in rows]
    print(format_table(table, SHOW_COLS))
    print(f"({len(rows)} rows)")


def _add_fields(p):
    p.add_argument("--json", dest="json_file", metavar="FILE", help="result JSON/JSONL (object, list, §8 record, multirung)")
    p.add_argument("--run-id")
    p.add_argument("--experiment")
    p.add_argument("--task")
    p.add_argument("--rung")
    p.add_argument("--metric")
    p.add_argument("--value", type=float)
    p.add_argument("--eval-size", type=int)
    p.add_argument("--cot-label", choices=COT_LABELS)
    p.add_argument("--eval-backend")
    p.add_argument("--ckpt-path")
    p.add_argument("--git-commit")
    p.add_argument("--eval-job-id")


def add_parser(sub):
    """Register ``ops result add|check|show``."""
    p = sub.add_parser("result", help="results log (state/results.jsonl)")
    rs = p.add_subparsers(dest="result_cmd")
    a = rs.add_parser("add", help="validate + append result rows")
    _add_fields(a)
    a.add_argument("--check", action="store_true", help="print check() warnings for each row")
    a.add_argument("--print-json", dest="json", action="store_true")
    a.set_defaults(func=_cmd_add)
    c = rs.add_parser("check", help="warnings for candidate rows; writes nothing")
    _add_fields(c)
    c.set_defaults(func=_cmd_check)
    s = rs.add_parser("show", help="table of results")
    s.add_argument("--experiment")
    s.add_argument("--task")
    s.add_argument("--run-id")
    s.add_argument("--best", action="store_true", help="best row per task/rung/metric")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_cmd_show)
    p.set_defaults(func=lambda args: p.print_help() or 2)
