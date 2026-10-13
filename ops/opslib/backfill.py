"""One-shot (idempotent) import of legacy launch ledgers, orchestrator state and result tables.

``ops backfill ledgers``: every ``*LEDGER*.tsv`` under ``debug/`` and the memexpress training hub
-> launch events (+ final_state updates where the ledger itself says DONE/FAILED/CANCELLED), then
final_state updates from orchestrator state JSONs, then git_commit from harvested provenance.json.
``ops backfill results``: ds64 / flop_scaling CSVs, ctc_suite all_results.jsonl and Prasann's rows
of ``../results-hub/results.csv``.

Re-running never duplicates: an event is skipped if an identical (source, run_id, event, job_id,
launched_at, final_state) key already exists; a result if (source_json, run_id, task, rung, metric) does.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from opslib import paths, registry, results

ULID = re.compile(r"\b01[0-9A-HJKMNP-TV-Z]{24}\b")
VESSL = re.compile(r"\bjob-[a-z0-9]{8,}\b")
SLURM_ID = re.compile(r"^(?:slurm-)?(\d{5,8})$")
SLURM_HINTS = ("jsteinhardt", "berkeleynlp", "horton", "mooney", "sneetches", "cubbins", "balrog", "feanor",
               "lorax", "mcfuzz", "saruman", "shadowfax", "smaug", "sunstone", "thidwick", "rainbowquartz",
               "smokyquartz")
BEAKER_HINTS = ("ai2/", "jupiter", "ceres", "saturn", "neptune", "titan", "beaker", "gantry")

# ledger directory (relative to the repo) -> experiment key; nearest ancestor wins.
EXPERIMENT_BY_DIR = {
    "debug/ds64": "ds64", "debug/ds64_fast2k": "ds64_fast2k", "debug/ds64_fast8k": "ds64_fast8k",
    "debug/flop_scaling": "flop_scaling", "debug/ffndrop_cpt": "ffndrop",
    "src/scripts/train/memexpress/cpt/ffndrop": "ffndrop",
    "src/scripts/train/memexpress/cpt/softdetach": "softdetach",
    "src/scripts/train/memexpress/ffnmoe": "ffnmoe", "debug/ctc_modelscale": "ctc_modelscale",
    "debug/ctc_olmo_hybrid": "ctc_olmo_hybrid", "debug/xformmatch_querypos": "xformmatch_querypos",
    "debug/flexcompute_train": "flexcompute", "debug/ctc_llama8b": "ctc_llama8b",
    "debug/ctc_purechunk": "ctc_purechunk", "debug/pooled_kv": "pooled_kv",
    "debug/ctc_crossfamily": "ctc_crossfamily", "debug/ctc_vllm_validation": "ctc_vllm_validation",
}
LEDGER_ROOTS = ("debug", "src/scripts/train/memexpress")

# (path in repo, experiment, read from git HEAD instead of the working tree?)
STATE_FILES = [
    ("debug/ds64/orchestrator_ds64_state.json", "ds64", False),
    ("debug/ds64_fast2k/state.json", "ds64_fast2k", False),
    ("debug/ds64_fast8k/state.json", "ds64_fast8k", False),
    # the working-tree copy was overwritten by a 4-run orchestrator; HEAD holds the full 93-run grid
    ("debug/flop_scaling/orchestrator35_state.json", "flop_scaling", True),
    ("debug/ffndrop_cpt/pipeline_state.json", "ffndrop", False),
]
STATE_GLOBS = [("debug/flop_scaling", "*_state.json", "flop_scaling")]
PROVENANCE_DIRS = ("results/ds64/harvest/runs", "results/flop_scaling/harvest/runs")


# ------------------------------------------------------------------ small helpers

def experiment_for(relpath: str) -> str:
    """Experiment key for a ledger path: nearest mapped ancestor dir, else its top-level folder."""
    parts = Path(relpath).parent.parts
    for i in range(len(parts), 0, -1):
        key = "/".join(parts[:i])
        if key in EXPERIMENT_BY_DIR:
            return EXPERIMENT_BY_DIR[key]
    return parts[1] if len(parts) > 1 else parts[0]


def job_id_from(*cells) -> str | None:
    """First real job id among ``cells`` (Beaker ULID, vessl ``job-…``, slurm id); junk -> None."""
    for c in cells:
        c = (c or "").strip()
        m = SLURM_ID.match(c)
        if m:
            return m.group(1)
        for pat in (ULID, VESSL):
            m = pat.search(c)
            if m:
                return m.group(0)
    return None


def cluster_for(text: str | None, job_id: str | None) -> str:
    """Map free-text cluster/node cells + job-id shape onto the contract enum."""
    t = (text or "").lower()
    if job_id and ULID.fullmatch(job_id):
        return "beaker"
    if (job_id and job_id.startswith("job-")) or "vessl" in t:
        return "vessl"
    if "lambda" in t or t.startswith("lane"):
        return "lambda"
    if any(h in t for h in BEAKER_HINTS):
        return "beaker"
    if "login" in t or t.startswith("cpu"):
        return "local"
    if job_id or any(h in t for h in SLURM_HINTS):
        return "slurm"
    return "local"


def final_state_for(status: str | None) -> str | None:
    """Ledger/orchestrator status text -> done|failed|cancelled|None (non-terminal)."""
    s = (status or "").strip().upper()
    if s.startswith("DONE"):
        return "done"
    if s.startswith("CANCEL"):
        return "cancelled"
    if "FAIL" in s or s.startswith("SIGSEGV"):
        return "failed"
    return None


def valid_run_name(name: str | None) -> bool:
    """A usable run name: non-empty, no whitespace, not a placeholder."""
    return (bool(name) and name not in ("-", "?") and not re.search(r"[\s(),/]", name)
            and not ULID.fullmatch(name) and not name.isupper())


def read_tsv(path) -> list:
    """TSV rows via ``csv`` (handles quoted multi-line cells), dropping blank and ``#`` lines."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    return [r for r in rows if r and any(c.strip() for c in r) and not r[0].lstrip().startswith("#")]


# --------------------------------------------------------------- ledger shapes

def _row(run_id, *, kind="train", job=None, cluster_text=None, launched=None, status=None, notes=None, **extra):
    """Common record produced by every shape parser (pre-event)."""
    return dict(run_id=run_id, kind=kind, job_id=job, cluster=cluster_for(cluster_text, job),
                launched_at=registry.to_iso_utc(launched), status=status,
                notes=(notes or "").strip() or None, **{k: v for k, v in extra.items() if v not in (None, "", "-")})


def parse_campaign(rows, experiment):
    """Headerless ds64-style: campaign scale task arm budget run_name cluster launched status notes."""
    out = []
    for r in rows:
        if len(r) < 10:
            continue
        _, _, task, arm, _budget, run, cl, launched, status, notes = r[:10]
        job = job_id_from(notes)
        kind = "data" if arm == "DATA" else "train"
        node = None
        out.append(_row(run, kind=kind, job=job, cluster_text=cl, launched=launched, status=status,
                        notes=notes.split("\n")[0] if not job else None, task=task, arm=arm, node=node))
    return out


def parse_modelscale(rows, experiment):
    """Headerless ctc_modelscale: widths 3/4 (ts run job [note]) and 9/10 (lane|beaker|vessl|node layouts)."""
    out = []
    for r in rows:
        if len(r) in (3, 4):
            ts, run, job = r[:3]
            out.append(_row(run, job=job_id_from(job), launched=ts, notes=r[3] if len(r) == 4 else None,
                            status=r[3] if len(r) == 4 else None))
            continue
        if len(r) < 9:
            continue
        ts, run, task, arm, _scale, _seq, where, col7, col8 = r[:9]
        note = r[9] if len(r) > 9 else None
        if job_id_from(col8) or where.startswith("lane") or where in ("beaker", "vessl"):
            job, cl, node = job_id_from(col8), f"{where} {col7}", (where if where not in ("beaker", "vessl") and not where.startswith("lane") else None)
        else:  # header layout: ... cluster experiment_id notes
            job, cl, node, note = job_id_from(col7), where, None, col8
        out.append(_row(run, job=job, cluster_text=cl, launched=ts, status=note, notes=note,
                        task=task, arm=arm, node=node))
    return out


def parse_softdetach_eval(rows, experiment):
    """Headerless softdetach EVAL_LEDGER: EVAL|CTC run arm experiment_id submitted [note]."""
    out = []
    for r in rows:
        if len(r) < 5:
            continue
        what, run, arm, ex, ts = r[:5]
        job = job_id_from(ex)
        out.append(_row(f"{run}__{what.lower()}__{job}", kind="eval", job=job, launched=ts,
                        notes=r[5] if len(r) > 5 else what, parent_run=run, arm=arm))
    return out


def parse_header(rows, experiment):
    """Ledgers with a header row; column roles resolved by name (covers ~12 header variants)."""
    head, out = [h.strip() for h in rows[0]], []
    for i, cells in enumerate(rows[1:], 2):
        d = dict(zip(head, (c.strip() for c in cells)))
        g = lambda *ks: next((d[k] for k in ks if d.get(k) not in (None, "", "-")), None)  # noqa: E731
        run = g("run_name", "run")
        notes = g("notes", "note", "details")
        status = g("status", "state")
        task, arm = g("task", "eval_task"), g("arm", "mode", "arms")
        job = job_id_from(g("experiment_id", "job_id"), notes)
        launched = g("launched_at", "launched", "date", "launched_utc")
        cl_text = " ".join(x for x in (g("cluster"), g("node")) if x)
        kind, parent, tag = "train", None, job or f"row{i}"  # tag disambiguates synthesized run ids
        if arm and arm.isupper() and task is None:  # olmo_hybrid status-only row (CANCELLED/RESUBMITTED)
            status, arm = arm, None
            if not job:
                out.append(dict(run_id=run, status_only=True, status=status, notes=notes))
                continue
        if "ckpt" in d:  # eval ledgers keyed by checkpoint (+ rung)
            parent = registry_run_from_ckpt(d["ckpt"])
            kind = "eval"
            rung = g("rung")
            run = f"{parent}__eval_{task}_{arm}_r{rung}" if rung else f"{experiment}__{tag}"
        elif "gen" in d:  # flexcompute: one job = several arms
            kind = "eval" if "eval" in (d.get("gen") or "") else "train"
            run = f"flexcompute-{d['gen']}-{job}"
        elif "kind" in d and d["kind"] != "train":  # xformmatch_querypos
            kind, parent = ("eval" if "eval" in d["kind"] else "other"), run
            run = f"{run}__{d['kind']}__{tag}"
        elif "start_layer" in d:  # ffnmoe
            if arm and (arm.startswith("eval") or arm == "bench"):
                kind, parent = "eval", (run if valid_run_name(run) else None)
                run = f"{run if valid_run_name(run) else 'ffnmoe'}__{arm}__{job}"
            elif arm and arm.startswith("setup"):
                kind, run = "other", f"ffnmoe-{arm}-{job}"
        if not valid_run_name(run):
            if not job:
                continue  # free-text note rows (PIVOT/bugfix/...) carry no job
            kind, notes = "other", " ".join(x for x in (run, notes) if x)
            run = f"{experiment}-job-{job}"
        out.append(_row(run, kind=kind, job=job, cluster_text=cl_text or g("cluster"), launched=launched,
                        status=status, notes=None if notes and ULID.match(notes) else notes,
                        task=task, arm=arm, parent_run=parent,
                        node=d.get("node") if d.get("node") and "," not in d["node"] else None))
    return out


def registry_run_from_ckpt(ckpt: str) -> str:
    """Run name of a checkpoint cell (strips ``/stepN`` and a trailing ``-YYYYMMDDTHHMMSS-0700`` stamp)."""
    base = results._run_from_ckpt(ckpt) or ckpt
    return re.sub(r"-\d{8}T\d{6}[-+]\d{4}$", "", base)


def parse_ledger(path, experiment) -> list:
    """Dispatch a ledger file to its shape parser."""
    rows = read_tsv(path)
    if not rows:
        return []
    first = rows[0]
    if first[0] in ("EVAL", "CTC"):
        return parse_softdetach_eval(rows, experiment)
    if re.match(r"\d{4}-\d{2}-\d{2}T", first[0]) and len(first) != 0 and not any(
            c in ("launched_at", "run_name") for c in first):
        return parse_modelscale(rows, experiment)
    if len(first) >= 10 and not any(c in ("job_id", "run_name", "run", "launched_at") for c in first):
        return parse_campaign(rows, experiment)
    return parse_header(rows, experiment)


def find_ledgers(repo) -> list:
    """All ``*LEDGER*.tsv`` under the ledger roots (repo-relative, sorted)."""
    out = []
    for root in LEDGER_ROOTS:
        base = Path(repo) / root
        if base.exists():
            out += [str(p.relative_to(repo)) for p in base.rglob("*LEDGER*.tsv")]
    return sorted(out)


# ------------------------------------------------------------- events + state

def _event_key(r):
    return (r.get("source"), r.get("run_id"), r.get("event"), r.get("job_id"), r.get("launched_at"),
            r.get("final_state"))


def ledger_events(repo, relpath) -> list:
    """Launch (+ terminal-status update) events for one ledger."""
    exp, src, events, launched = experiment_for(relpath), f"backfill:{relpath}", [], set()
    for rec in parse_ledger(Path(repo) / relpath, exp):
        final = final_state_for(rec.pop("status", None))
        if rec.pop("status_only", False):  # e.g. "<run>(<slurm id>)  CANCELLED" -> only for known runs
            m = re.fullmatch(r"(.+)\((\d+)\)", rec["run_id"] or "")
            rec["run_id"] = m.group(1) if m else rec["run_id"]
            if final and rec["run_id"] in launched:
                events.append(registry.make_update(rec["run_id"], final_state=final, state_source=relpath,
                                                   source=src, notes=rec.get("notes")))
            continue
        launch = registry.make_launch(experiment=exp, launched_by="backfill", source=src,
                                      ts=registry.now_iso(), **rec)
        if rec.get("launched_at") is None:
            launch.pop("launched_at", None)  # unknown, rather than "now"
        events.append(launch)
        launched.add(rec["run_id"])
        if final:
            events.append(registry.make_update(rec["run_id"], final_state=final, state_source=relpath, source=src))
    return events


def _read_state(repo, rel, from_head):
    if from_head:
        try:
            res = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), "show", f"HEAD:{rel}"],
                                 capture_output=True, text=True, timeout=20)
            if res.returncode == 0:
                return json.loads(res.stdout), f"git:HEAD:{rel}"
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
    p = Path(repo) / rel
    return (json.loads(p.read_text()), rel) if p.exists() else (None, rel)


def _state_final(entry) -> str | None:
    st = entry.get("state")
    if st == "F":  # ffndrop pipeline: F(inalized) + exit code
        code = entry.get("code")
        return "done" if code == 0 else ("cancelled" if code == "canceled" else "failed")
    return final_state_for(st)


def state_events(repo, known: dict) -> list:
    """final_state updates from orchestrator state JSONs; unknown runs get a launch event first."""
    files = list(STATE_FILES)
    for sub, pat, exp in STATE_GLOBS:
        for p in sorted((Path(repo) / sub).glob(pat)):
            rel = str(p.relative_to(repo))
            if rel not in [f[0] for f in files]:
                files.append((rel, exp, False))
    events = []
    for rel, exp, head in files:
        data, src_name = _read_state(repo, rel, head)
        if not isinstance(data, dict):
            continue
        src = f"backfill:{src_name}"
        for section, kind in (("runs", "train"), ("sft", "train"), ("evals", "eval")):
            for name, entry in (data.get(section) or {}).items():
                if not isinstance(entry, dict):
                    continue
                rid = name if kind == "train" else f"{name}__eval"
                job = job_id_from(entry.get("ex"))
                if rid not in known:
                    events.append(registry.make_launch(
                        run_id=rid, experiment=exp, kind=kind, cluster="beaker" if job else "local",
                        job_id=job, parent_run=name if kind == "eval" else None, task=entry.get("task"),
                        arm=entry.get("arm"), launched_by="backfill", source=src, ts=registry.now_iso(),
                        notes=entry.get("note")))
                    events[-1].pop("launched_at", None)
                    known[rid] = {"job_id": job}
                final = _state_final(entry)
                if final:
                    events.append(registry.make_update(rid, final_state=final, state_source=src_name,
                                                       source=src, job_id=job))
    return events


def provenance_events(repo, known: dict) -> list:
    """git_commit for train runs from harvested ``results/*/harvest/runs/<run>/provenance.json``."""
    events = []
    for d in PROVENANCE_DIRS:
        base = Path(repo) / d
        if not base.exists():
            continue
        for run_dir in sorted(base.iterdir()):
            if run_dir.name not in known or known[run_dir.name].get("git_commit"):
                continue
            prov = run_dir / "provenance.json"
            try:
                commit = json.loads(prov.read_text()).get("git_commit")
            except (OSError, json.JSONDecodeError):
                continue
            if commit:
                src = f"backfill:{d}/{run_dir.name}/provenance.json"
                events.append(registry.make_update(run_dir.name, git_commit=commit, source=src))
                known[run_dir.name]["git_commit"] = commit
    return events


def backfill_ledgers(repo=None, runs_file=None, dry_run=False) -> dict:
    """Import all ledgers + state + provenance into runs.jsonl; returns counts."""
    repo = Path(repo or paths.REPO_ROOT)
    path = registry.runs_path(runs_file)
    existing = registry.read_jsonl(path)
    seen = {_event_key(r) for r in existing}
    events, per_ledger = [], {}
    for rel in find_ledgers(repo):
        ev = ledger_events(repo, rel)
        per_ledger[rel] = sum(e["event"] == "launch" for e in ev)
        events += ev
    known = registry.fold(existing + events)
    events += state_events(repo, known)
    known = registry.fold(existing + events)
    events += provenance_events(repo, known)
    new = []
    for e in events:
        k = _event_key(e)
        if k not in seen:
            seen.add(k)
            new.append(e)
    if not dry_run:
        registry.append_rows(path, new)
    runs = registry.fold(existing + new)
    return {"ledgers": per_ledger, "events_new": len(new), "events_skipped": len(events) - len(new),
            "runs_total": len(runs),
            "experiments": dict(Counter(str(r.get("experiment")) for r in runs.values())),
            "final_state": dict(Counter(str(r.get("final_state")) for r in runs.values())),
            "git_commit_coverage": sum(bool(r.get("git_commit")) for r in runs.values())}


# --------------------------------------------------------------------- results

def _per_rung_csv(repo, rel, experiment, run_col="run", job_col=None):
    """Rows from a wide CSV with ``f1_<rung>`` columns (one result row per non-empty rung)."""
    p = Path(repo) / rel
    if not p.exists():
        return []
    out = []
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            for col, val in r.items():
                m = re.fullmatch(r"f1_(\d+k)", col or "")
                if not m or val in (None, ""):
                    continue
                out.append(dict(run_id=r[run_col], experiment=experiment, task=r.get("task"), rung=m.group(1),
                                metric="f1", value=float(val), eval_job_id=(r.get(job_col) or None) if job_col else None,
                                source_json=rel, eval_backend="native"))
    return out


def hub_rows(hub_csv, users=("PrasannS", "prasann")) -> list:
    """Prasann's rows of the retired results-hub CSV (``who_ran`` filter)."""
    p = Path(hub_csv)
    if not p.exists():
        return []
    out = []
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("who_ran") not in users:
                continue
            ckpt = r.get("weka_model_location") or None
            run = r.get("model_train_data_hparams") or (results._run_from_ckpt(ckpt) if ckpt else None)
            task = r["eval_name"] + (f"@{r['eval_version']}" if r.get("eval_version") else "")
            try:
                value = float(r["metric_value"])
            except ValueError:
                continue
            q = [f for f in (r.get("quality_flag") or "").split(";") if f and f != "small_eval_set"]
            out.append(dict(
                run_id=run or f"hub-{r['id']}", experiment=registry.guess_experiment(run) or "results_hub_legacy",
                task=task, rung=r.get("eval_context_length") or None, metric=r["metric_name"], value=value,
                eval_size=int(r["eval_data_quantity"]) if (r.get("eval_data_quantity") or "").isdigit() else None,
                ckpt_path=ckpt, git_commit=r.get("git_commit") or None, source_json=f"results-hub:{r['id']}",
                flags=q, notes=r.get("other_notes") or None, recorded_at=(r.get("record_added") or None),
            ))
    return out


def collect_results(repo) -> list:
    """All legacy result rows (unvalidated dicts), deduped by run/task/rung/metric (first source wins)."""
    repo = Path(repo)
    rows = _per_rung_csv(repo, "results/ds64/results.csv", "ds64", job_col="eval_ex")
    fs = repo / "results/flop_scaling"
    for rel in ["results/flop_scaling/results35.csv"] + sorted(
            str(p.relative_to(repo)) for p in fs.glob("results_*.csv") if p.name != "results35.csv"):
        rows += _per_rung_csv(repo, rel, "flop_scaling")
    ctc = repo / "results/ctc_suite/all_results.jsonl"
    if ctc.exists():
        latest = {}  # results_io semantics: the LAST line per (task, model, arm, rung) wins
        for r in results.load_json_file(ctc):
            latest[(r.get("run_id"), r.get("task"), str(r.get("rung")), r.get("metric"))] = r
        rows += [{**r, "experiment": "ctc_suite", "source_json": "results/ctc_suite/all_results.jsonl"}
                 for r in latest.values()]
    rows += hub_rows(repo.parent / "results-hub" / "results.csv")
    out, seen = [], set()
    for r in rows:
        k = (r.get("run_id"), r.get("task"), str(r.get("rung")), r.get("metric"))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def backfill_results(repo=None, results_file=None, dry_run=False) -> dict:
    """Validate + append legacy results (flags ``backfill``/``provenance_incomplete``); returns counts."""
    repo = Path(repo or paths.REPO_ROOT)
    path = results.results_path(results_file)
    existing = results.load(path)
    seen = {(r.get("source_json"), r.get("run_id"), r.get("task"), str(r.get("rung")), r.get("metric")) for r in existing}
    new, bad, by_src = [], 0, Counter()
    for fields in collect_results(repo):
        try:
            row = results.normalize(fields, backfill=True)
        except ValueError:
            bad += 1
            continue
        k = (row.get("source_json"), row.get("run_id"), row.get("task"), str(row.get("rung")), row.get("metric"))
        if k in seen:
            continue
        seen.add(k)
        new.append(row)
        by_src[(row.get("source_json") or "").split(":")[0]] += 1
    if not dry_run:
        registry.append_rows(path, new)
    return {"rows_new": len(new), "rejected": bad, "by_source": dict(by_src),
            "exact_0_or_1": sum(r["value"] in (0.0, 1.0) for r in new)}


# ----------------------------------------------------------------------------- CLI

def _cmd(args):
    if args.what == "ledgers":
        res = backfill_ledgers(repo=args.repo, dry_run=args.dry_run)
    else:
        res = backfill_results(repo=args.repo, dry_run=args.dry_run)
    print(json.dumps(res, indent=1, sort_keys=True))
    if args.dry_run:
        print("(dry run: nothing written)")


def add_parser(sub):
    """Register ``ops backfill ledgers|results``."""
    p = sub.add_parser("backfill", help="import legacy ledgers / result tables into state/")
    p.add_argument("what", choices=("ledgers", "results"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--repo", help="repo root to read (default: paths.REPO_ROOT)")
    p.set_defaults(func=_cmd)
