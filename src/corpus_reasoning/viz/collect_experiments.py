"""
Stage 2 — Experiments source.

Gathers the CPT data-mixing SFT experiments from the OLMo-core training scripts:
parses each script's ``# Question this answers:`` comment block as its
description, pulls the key config constants (CPT_FRAC, SEQUENCE_LENGTH, ...), and
reads the git commit that introduced it. Merges a curated RULER results table
(``viz/results.json``). Writes ``outputs/experiments.json`` for the renderer.

Usage:
    python -m viz.collect_experiments
    OLMO_CORE_ROOT=/path/to/OLMo-core python viz/collect_experiments.py
"""

import json
import os
import re
import subprocess

try:
    from . import config
except ImportError:
    import config

# The CPT-mix experiment family, in presentation order. ``phase`` groups the
# small probes from the full-scale long-context run.
EXPERIMENTS = [
    {
        "key": "noruler-4k",
        "title": "No-RULER baseline (4k)",
        "script": "Qwen3-4B-dense-noruler-4k-SFT.py",
        "phase": "Probe (4k–5k)",
        "cpt_frac": 0.0,
    },
    {
        "key": "cptmix-4k",
        "title": "CPT-mix 15% (4k)",
        "script": "Qwen3-4B-dense-noruler-4k-CPTmix-SFT.py",
        "phase": "Probe (4k–5k)",
    },
    {
        "key": "cptmix-5k",
        "title": "CPT-mix 30% (5k)",
        "script": "Qwen3-4B-dense-noruler-5k-CPTmix-SFT.py",
        "phase": "Probe (4k–5k)",
    },
    {
        "key": "cpt40-8k-debug",
        "title": "CPT-mix 40% (8k debug)",
        "script": "Qwen3-4B-dense-10task1k-cpt40-8k-debug-SFT.py",
        "phase": "Scale (64k)",
    },
    {
        "key": "cpt40-64k",
        "title": "CPT-mix 40% (64k, 10-task long run)",
        "script": "Qwen3-4B-dense-10task1k-cpt40-64k-SFT.py",
        "phase": "Scale (64k)",
    },
]

# Constants worth surfacing in the config table.
CONST_KEYS = [
    "SEQUENCE_LENGTH",
    "CPT_FRAC",
    "NUM_NODES",
    "NUM_STEPS",
    "GLOBAL_BATCH_SIZE",
]


def parse_description(src: str) -> str:
    """Extract the comment paragraph containing 'Question this answers'."""
    lines = src.splitlines()
    # find runs of consecutive comment lines
    runs = []
    cur = []
    for ln in lines[:120]:
        if ln.lstrip().startswith("#"):
            body = ln.lstrip()[1:].strip()
            # drop pure separator lines (e.g. "# ----------")
            if body and set(body) <= set("-=*#_ "):
                body = ""
            cur.append(body)
        else:
            if cur:
                runs.append(cur)
                cur = []
    if cur:
        runs.append(cur)
    for run in runs:
        joined = " ".join(x for x in run if x)
        if "Question this answers" in joined or "RULER" in joined or "CPT" in joined:
            return joined
    # fall back to the longest comment run
    if runs:
        return " ".join(max(runs, key=lambda r: sum(len(x) for x in r)))
    return ""


def parse_constants(src: str) -> dict:
    out = {}
    for key in CONST_KEYS:
        m = re.search(rf"^{key}\s*=\s*([^\n#]+)", src, re.M)
        if m:
            out[key] = m.group(1).strip()
    return out


def git_commit(script_path: str) -> str:
    try:
        rel = os.path.relpath(script_path, config.OLMO_CORE_ROOT)
        res = subprocess.run(
            ["git", "log", "-1", "--format=%h %s", "--", rel],
            cwd=config.OLMO_CORE_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return res.stdout.strip()
    except Exception:
        return ""


def load_results():
    results_path = os.path.join(config.VIZ_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return {}


NARRATIVE = (
    "Continued-pretraining (CPT) data mixing for long-context instruction tuning. "
    "Instruction-tuning a long-context base model on the corpus-reasoning task suite "
    "degrades held-out RULER retrieval (catastrophic forgetting). These experiments test "
    "whether mixing a fraction of raw CPT text (dolma3longmino — the corpus the base was "
    "trained on, no completion mask ⇒ full-sequence LM loss) back into the SFT mixture "
    "recovers that long-context ability. The probes isolate the effect of the CPT fraction "
    "(0% → 15% → 30%) at short windows; the scale phase pushes the mix to 40% and trains at "
    "the 64k lengths RULER actually tests, on the full 10-task × 1000-example suite."
)


def main():
    config.ensure_out_dir()
    print(f"[collect_experiments] reading SFT scripts from {config.SFT_SCRIPTS_DIR}")

    experiments = []
    for exp in EXPERIMENTS:
        path = os.path.join(config.SFT_SCRIPTS_DIR, exp["script"])
        entry = dict(exp)
        if os.path.exists(path):
            src = open(path).read()
            entry["description"] = parse_description(src)
            entry["config"] = parse_constants(src)
            entry["commit"] = git_commit(path)
            entry["found"] = True
            # CPT fraction from the constant if not pinned in the manifest.
            if "cpt_frac" not in entry and "CPT_FRAC" in entry["config"]:
                try:
                    entry["cpt_frac"] = float(entry["config"]["CPT_FRAC"])
                except ValueError:
                    pass
        else:
            entry["description"] = ""
            entry["config"] = {}
            entry["commit"] = ""
            entry["found"] = False
            print(f"  WARNING: script not found: {exp['script']}")
        print(f"  {exp['key']:16s} found={entry['found']}  {entry['commit'][:50]}")
        experiments.append(entry)

    payload = {
        "experiments": experiments,
        "ruler_results": load_results(),
        "narrative": NARRATIVE,
        "olmo_core_root": config.OLMO_CORE_ROOT,
    }
    with open(config.EXPERIMENTS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    found = sum(1 for e in experiments if e["found"])
    print(f"[collect_experiments] wrote {config.EXPERIMENTS_JSON} "
          f"({found}/{len(experiments)} scripts found)")


if __name__ == "__main__":
    main()
