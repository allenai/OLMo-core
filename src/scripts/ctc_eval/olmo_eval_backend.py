"""
The ``olmo-eval`` backend of :mod:`launch_ctc_evals`: grade a checkpoint on the CTC suite AS
olmo-eval defines it -- its data (``PrasannSinghal/ctc-suite-eval``), its prompt builders, parsers
and scorers -- through its native olmo_core provider, with prompts rendered exactly as the setA SFT
data was (``CTC_SUITE_PROMPT_FORMAT=chat``). That makes the eval IID with training by construction
rather than by a separately maintained driver.

It plans (row, rung, limit) cells from an eval-size policy, costs each from measured throughput,
splits any cell too big for the wall-clock budget into exact shards (``CTC_SUITE_SHARDS``: the
shards partition the very rows an unsharded run grades), packs cells into single-GPU jobs, and
submits one gantry job per bin.

Cost model (fast-compressive-landmark Qwen3.5-4B, H100, bs=1, FLA autotune fix, measured
2026-09-23 on the outlier ladder): prefill GPU-seconds per rung in :data:`PREFILL_S`, plus
:data:`DECODE_S_PER_TOKEN` per decoded token. Answer lengths per (row, rung) are measured from the
eval rows' rendered targets (``debug/ctc_eval_speed/answer_tokens.json``). Dense arms are cheaper; the
model is a ceiling for them.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
ROSTER_JSON = os.path.join(REPO, "debug", "ctc_sft_setA", "olmo_eval_roster.json")
ANSWER_TOKENS_JSON = os.path.join(REPO, "debug", "ctc_eval_speed", "answer_tokens.json")

#: The 12 rows setA trains IID with, and the two held-out OOD rows (standing rule: OOD always).
SETA_ROWS = ["ctc_nq", "ctc_hpqa", "ctc_qdmatch_nq", "ctc_outlier", "ctc_oolong",
             "ctc_contradiction", "ctc_xabsence", "ctc_reorder", "ctc_rerank", "ctc_strmatch",
             "ctc_textgroups", "ctc_grouping"]
OOD_ROWS = {"ctc_contra_fever": ("contra_fever", "contradiction"),
            "ctc_outlier_review": ("outlier_review", "outlier")}
OOD_RUNGS = ["r2k", "r4k", "r8k", "r16k", "r32k", "r64k", "r128k", "r256k"]

#: GPU-seconds of prefill per example (compressive landmark, measured; 256k extrapolated x2.7).
PREFILL_S = {"r2k": 0.2, "r4k": 0.3, "r8k": 0.5, "r16k": 0.9, "r32k": 1.7, "r64k": 4.2,
             "r128k": 12.0, "r256k": 33.0}
DECODE_S_PER_TOKEN = 0.030
#: Rows at or above this rung ship 125 examples (roster ``eval_size``).
LONG_RUNG_ROWS = 125
#: rerank decode cap (``CTC_SUITE_RERANK_DECODE_TOKENS``): scored on the first 10 ids only.
RERANK_DECODE_TOKENS = 160
#: Rungs whose eval rows can exceed Qwen3.5's 262,144 positions for some rows (measured).
OVERFLOW_256K = {"ctc_hpqa", "ctc_outlier", "ctc_qdmatch_nq", "ctc_rerank"}

DEFAULT_POLICY = "r2k-r32k:500,r64k:100,r128k:50,r256k:50"


@dataclass
class Cell:
    row: str
    subset: str
    rung: str
    limit: int
    cost_s: float
    shard: Optional[Tuple[int, int]] = None

    @property
    def task(self) -> str:
        return f"{self.row}:{self.rung}"


def parse_policy(policy: str) -> Dict[str, int]:
    """``"r2k-r32k:500,r64k:100"`` -> ``{"r2k": 500, ..., "r64k": 100}``."""
    order = list(PREFILL_S)
    out: Dict[str, int] = {}
    for part in policy.split(","):
        rungs, n = part.split(":")
        if "-" in rungs:
            lo, hi = rungs.split("-")
            for r in order[order.index(lo): order.index(hi) + 1]:
                out[r] = int(n)
        else:
            out[rungs] = int(n)
    return out


def plan(rows: List[str], policy: Dict[str, int], rerank_cap: bool = True) -> List[Cell]:
    R = json.load(open(ROSTER_JSON))["roster"]
    answers = json.load(open(ANSWER_TOKENS_JSON)) if os.path.exists(ANSWER_TOKENS_JSON) else {}
    cells = []
    for row in rows:
        if row in R:
            subset, rungs = R[row]["subset"], R[row]["rungs"]
            eval_size = R[row].get("eval_size", {})
        else:
            subset, rungs, eval_size = OOD_ROWS[row][0], OOD_RUNGS, {}
        measured = answers.get(row) or answers.get(
            {"ctc_contra_fever": "ctc_contradiction",
             "ctc_outlier_review": "ctc_outlier"}.get(row, ""), {})
        for rung in rungs:
            if rung not in policy:
                continue
            available = eval_size.get(rung, LONG_RUNG_ROWS if rung in ("r256k",) else 500)
            limit = min(policy[rung], available)
            # answer length: measured at this rung, else at the longest measured rung
            ans = measured.get(rung) or (max(measured.values()) if measured else 32)
            if row == "ctc_rerank" and rerank_cap:
                ans = min(ans, RERANK_DECODE_TOKENS)
            cost = limit * (PREFILL_S[rung] + DECODE_S_PER_TOKEN * ans)
            cells.append(Cell(row, subset, rung, limit, cost))
    return cells


def pack(cells: List[Cell], budget_s: float) -> List[List[Cell]]:
    """Split cells bigger than the budget into exact shards, then LPT-pack into single-GPU bins."""
    pieces: List[Cell] = []
    for c in cells:
        n = max(1, math.ceil(c.cost_s / budget_s))
        if n == 1:
            pieces.append(c)
            continue
        for i in range(n):
            pieces.append(Cell(c.row, c.subset, c.rung, c.limit, c.cost_s / n, shard=(i, n)))
    pieces.sort(key=lambda c: -c.cost_s)
    bins: List[List[Cell]] = []
    loads: List[float] = []
    for p in pieces:
        # a bin runs ONE olmo-eval call, so it may hold at most one shard of any cell
        ok = [i for i in range(len(bins)) if loads[i] + p.cost_s <= budget_s
              and all(c.task != p.task for c in bins[i])]
        if ok:
            best = min(ok, key=lambda i: loads[i])
            bins[best].append(p)
            loads[best] += p.cost_s
        else:
            bins.append([p])
            loads.append(p.cost_s)
    return bins


def job_command(ckpt: str, cells: List[Cell], out_dir: str, tokenizer: str,
                max_model_len: int, job_tag: str) -> Tuple[str, Dict[str, str]]:
    """:returns: ``(bash command, env)`` for one bin."""
    shards = {f"{c.subset}:{c.rung}": f"{c.shard[0]}/{c.shard[1]}" for c in cells if c.shard}
    env = {"CTC_SUITE_PROMPT_FORMAT": "chat",
           "CTC_SUITE_RERANK_DECODE_TOKENS": str(RERANK_DECODE_TOKENS)}
    if shards:
        env["CTC_SUITE_SHARDS"] = json.dumps(shards)
    # ONE olmo-eval call per bin: every -t reuses the loaded model (a load is 1-11 min on weka).
    # Overrides after a -t apply to that task.
    tasks = " ".join(f"-t {c.task} -o limit={c.limit}" for c in cells)
    parts = [
        f"olmo-eval run -m {ckpt} {tasks} -H default "
        f"-o provider.kind=olmo_core -o provider.tokenizer={tokenizer} "
        f"-o provider.max_model_len={max_model_len} -o provider.kwargs.batch_size=1 "
        f"-o provider.kwargs.eos_token_id=248046 -o provider.kwargs.pad_token_id=248044 "
        f"-o provider.kwargs.validate_checkpoint=false "
        f"-o provider.kwargs.allow_tokenizer_fallback=true "
        f"-O {out_dir}/{job_tag} 2>&1 | grep -v Warning | tail -80; "
        f"echo \"=== JOB {job_tag} rc=${{PIPESTATUS[0]}} $(date -u +%T)\""]
    return "set -uo pipefail; " + " ".join(parts), env


def submit(run_name: str, bins: List[List[Cell]], ckpt: str, out_root: str, *, tokenizer: str,
           max_model_len: int, olmo_eval_dataset: str, cluster: str, priority: str,
           dry_run: bool) -> List[str]:
    """Submit one single-GPU gantry job per bin. :returns: the job names."""
    names = []
    install = ("cp -r /olmoeval_src /tmp/olmoeval && pip install /tmp/olmoeval "
               "'transformers==5.7.0' 'huggingface_hub==1.12.2' && pip install -e '.[fla]' "
               "&& pip install dataclass-extensions")
    for i, cells in enumerate(bins):
        name = f"ctcoe-{run_name}-{i:02d}"[:120]
        cmd, env = job_command(ckpt, cells, out_root, tokenizer, max_model_len, f"job{i:02d}")
        argv = ["gantry", "run", "--name", name, "-w", "ai2/flex2", "-b", "ai2/oe-other",
                "--cluster", cluster, "--gpus", "1", "--priority", priority,
                "--weka", "oe-training-default:/weka/oe-training-default",
                "--branch", "prasann/landmark",
                "--dataset", f"{olmo_eval_dataset}:/olmoeval_src",
                "--beaker-image", "tylerr/olmo-core-tch291cu128-2025-11-25",
                "--python-manager", "conda", "--system-python", "--install", install,
                "--allow-dirty", "--timeout", "0", "--shared-memory", "32GiB", "--yes"]
        for k, v in env.items():
            argv += ["--env", f"{k}={v}"]
        argv += ["--", "bash", "-c", cmd]
        names.append(name)
        if dry_run:
            continue
        subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    return names


def describe(bins: List[List[Cell]], budget_s: float) -> str:
    total = sum(c.cost_s for b in bins for c in b)
    lines = [f"{len(bins)} single-GPU jobs, {total / 3600:.1f} GPU-h estimated, "
             f"longest bin {max(sum(c.cost_s for c in b) for b in bins) / 60:.0f} min "
             f"(budget {budget_s / 60:.0f} min + setup)"]
    for i, b in enumerate(bins):
        lines.append(f"  job {i:02d} {sum(c.cost_s for c in b) / 60:5.0f} min  " + ", ".join(
            f"{c.task}x{c.limit}" + (f"[{c.shard[0]}/{c.shard[1]}]" if c.shard else "")
            for c in b))
    risky = sorted({c.task for b in bins for c in b if c.rung == "r256k" and c.row in OVERFLOW_256K})
    if risky:
        lines.append("  ⚠ some eval rows of " + ", ".join(risky) + " exceed 262,144 tokens; the "
                     "provider LEFT-TRUNCATES them. Quote those cells with that caveat.")
    return "\n".join(lines)
