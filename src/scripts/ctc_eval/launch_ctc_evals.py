"""
Evaluate a checkpoint — one command, every standing rule applied, results-hub row emitted.

A complete eval is never one submission. It is a base pass over all nine tasks, plus one xlong pass
per YaRN group (the OOD ladders have no xlong rung files, so including them there would re-run the
base ladder under an xlong tag and write a duplicate row rather than a new measurement), plus a
ledger written AT SUBMISSION TIME. Getting that sequence right by hand is where launches go wrong,
so this drives it from one command.

This ORCHESTRATES ``run_q4b_beaker_multirung_eval.py`` rather than replacing it: that launcher and
its on-node runner are the proven path and stay the only thing that talks to Beaker.

The four standing rules are applied unless explicitly overridden, and every override is recorded in
the ledger rather than left implicit:

1. xlong rungs always (``--no-xlong`` to skip).
2. YaRN by rung -- factor 2 at 256k/512k, 4 at 1M, 8 at 2M. Passes are grouped by factor because
   one submission cannot mix them.
3. OOD ladders always (``fiqa``, ``scifact``, ``outlier_review``, ``contra_fever``).
4. A launch ledger every time, holding every results-hub column except the metric itself.

    python src/scripts/ctc_eval/launch_ctc_evals.py \
        --ckpt /weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_suite/ckpts/<run>/step3000 \
        --run-name q35-4b-setA-dense --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))          # src/scripts/ctc_eval
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
LAUNCHER = os.path.join(
    REPO, "src", "scripts", "train", "memexpress", "singletask_ladder",
    "run_q4b_beaker_multirung_eval.py",
)

IN_DIST = ["contra", "nq", "rerank", "outlier", "oolong"]
OOD = ["fiqa", "scifact", "outlier_review", "contra_fever"]

#: Rung -> YaRN factor. One submission cannot mix factors, so this also defines the pass grouping.
YARN = {"64k": 1, "128k": 1, "256k": 2, "512k": 2, "1M": 4, "2M": 8}

#: Variant -> (results-hub model_type, attention_type, whether vLLM can serve it).
#: Only plain causal attention has an HF/vLLM mapping. The mask variants reconstruct per-token
#: chunk roles from the box markers and have no vLLM path at all, so they run native at batch 1 --
#: which is a different budget, not a slower version of the same one.
VARIANTS = {
    "dense":         ("dense",       "full",                 True),
    "landmark":      ("landmark",    "sparse_landmark",      False),
    "compressive":   ("compressive", "compressive_landmark", False),
    "docchunk":      ("docchunk",    "document_chunked",     False),
}


def variant_from_run_name(run_name: str) -> str:
    """
    Infer the arm from the run name, the same convention the eval emitter uses.

    :param run_name: The training run name.

    :returns: One of :data:`VARIANTS`.
    """
    n = run_name.lower()
    for key in ("compressive", "landmark", "docchunk"):
        if key in n:
            return key
    return "dense"


def build_passes(tasks_base: List[str], xlong_rungs: List[str], xlong: bool) -> List[Dict]:
    """
    :returns: The submissions to make, base first, then one per YaRN factor.
    """
    passes = [{"tag": "base", "tasks": tasks_base, "xlong": False, "rungs": [], "yarn": 1}]
    if not xlong:
        return passes
    by_factor: Dict[int, List[str]] = {}
    for r in xlong_rungs:
        by_factor.setdefault(YARN[r], []).append(r)
    for factor in sorted(by_factor):
        passes.append({
            # OOD ladders have no xlong rung files: the runner prints "no xlong rungs for TASK=..."
            # and silently re-runs the BASE ladder under an xlong tag -- a duplicate row.
            "tag": "xlong" if factor == 1 else f"xlong_yarn{factor}",
            "tasks": IN_DIST, "xlong": True, "rungs": by_factor[factor], "yarn": factor,
        })
    return passes


def ledger(args, variant: str, passes: List[Dict], git_commit: str) -> str:
    """:returns: The launch-ledger YAML, holding every results-hub column but the metric."""
    model_type, attn, _ = VARIANTS[variant]
    today = datetime.date.today().isoformat()
    lines = [
        f"# Launch ledger — {args.run_name}",
        f"# Written at submission time by launch_ctc_evals.py. `pull-evals` joins metric_value on.",
        "",
        "checkpoint:",
        f"  run_name: {args.run_name}",
        f"  ckpt: {args.ckpt}",
        f"  model_type: {model_type}",
        f"  model_subtype: {args.model_subtype}",
        f"  attention_type: {attn}",
        f"  pipeline_stage: {args.pipeline_stage}",
        f"  chat_template: {args.chat_template}",
        f"  model_slug: {args.model_slug}",
        "  training_description: >",
        f"    {args.notes or '(fill in)'}",
        "",
        "common:",
        f"  who_ran: {os.environ.get('USER', 'unknown')}",
        f"  date_eval_ran: {today}",
        f"  eval_version: {args.ladder_version}",
        f"  eval_set_weka_pointer: {args.eval_bundle}",
        f"  git_commit: {git_commit}",
        f"  cluster: {args.cluster}",
        f"  priority: {args.priority}",
        f"  tokenizer: {args.tokenizer or '(runner default)'}",
        f"  query_position: {args.query_position}",
        f"  backend: {args.backend}",
        "  landmark_top_k_fixed_val: \"\"",
        f"  landmark_top_k_percentage: {args.landmark_top_k_pct}",
        f"  landmark_nonselected_percentage: {args.landmark_nonselected_pct}",
        "",
        "passes:",
    ]
    for p in passes:
        lines += [
            f"  - eval_tag: {p['tag']}",
            f"    tasks: [{', '.join(p['tasks'])}]",
            f"    rungs: [{', '.join(p['rungs']) or 'base ladder 2k-32k'}]",
            f"    yarn_factor: {p['yarn']}",
            f"    ckpt_used: {args.ckpt}",
            "    status: submitted",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="ABSOLUTE weka step dir")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--arm", default="auto", choices=["auto", *VARIANTS],
                    help="default: inferred from the run name, the emitter's own convention")
    ap.add_argument("--backend", default="native", choices=["native", "vllm", "olmo-eval"],
                    help="olmo-eval = grade on the suite's own data/graders via its native "
                         "olmo_core provider, prompts rendered as the SFT data (chat); IID with "
                         "setA by construction. See olmo_eval_backend.py.")
    ap.add_argument("--budget-hours", type=float, default=1.5,
                    help="olmo-eval backend: wall-clock budget per single-GPU job")
    ap.add_argument("--policy", default="",
                    help="olmo-eval backend: eval sizes per rung, e.g. "
                         "'r2k-r32k:500,r64k:100,r128k:50,r256k:50' (the default)")
    ap.add_argument("--rows", default="setA",
                    help="olmo-eval backend: 'setA' (12 IID rows + 2 OOD) or a comma list")
    ap.add_argument("--olmo-eval-dataset", default="prasanns/olmo-eval-src-37563d01-1790134494",
                    help="Beaker dataset holding the olmo-eval source (branch commits not on PyPI)")
    ap.add_argument("--setup-minutes", type=float, default=8.0,
                    help="olmo-eval backend: per-job install + model load, off the budget")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--no-xlong", dest="xlong", action="store_false",
                    help="override standing rule 1; recorded in the ledger")
    ap.add_argument("--no-ood", dest="ood", action="store_false",
                    help="override standing rule 3; recorded in the ledger")
    ap.add_argument("--xlong-rungs", default="64k,128k,256k")
    ap.add_argument("--query-position", default="both",
                    help="MUST match the SFT shards; a mismatch reads as a capability gap")
    ap.add_argument("--chat-template", default="chat",
                    help="forwarded as the launcher's --prompt-format (chat = SFT, raw = BASE/CPT)")
    ap.add_argument("--pipeline-stage", default="SFT")
    ap.add_argument("--model-subtype", default="")
    ap.add_argument("--model-slug", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--ladder-version", default="v2")
    ap.add_argument("--eval-bundle",
                    default="/weka/oe-training-default/ai2-llm/checkpoints/prasanns/"
                            "_eval_bundle_eval500_v2_clean")
    ap.add_argument("--landmark-top-k-pct", default="0.1")
    ap.add_argument("--landmark-nonselected-pct", default="0.1")
    ap.add_argument("--priority", default="urgent", help="never below urgent (project directive)")
    ap.add_argument("--max-length", type=int, default=40960)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    variant = variant_from_run_name(args.run_name) if args.arm == "auto" else args.arm
    _, _, vllm_ok = VARIANTS[variant]

    if args.backend == "vllm":
        if not vllm_ok:
            raise SystemExit(
                f"--backend vllm is impossible for arm {variant!r}: the mask variants reconstruct\n"
                f"per-token chunk roles from the box markers and have NO HF or vLLM mapping. They\n"
                f"run native at batch size 1, which is a different budget, not a slower version of\n"
                f"the same one. Use --backend native.")
        raise SystemExit(
            "--backend vllm is not wired into the results-hub launch path yet. olmo-eval pins\n"
            "vllm[runai]==0.19.1, which predates Qwen3.5/GDN and cannot serve this model family;\n"
            "it needs overriding to 0.25.1 first. See records/ctc-fast-suite-eval.md.\n"
            "The validated drivers meanwhile are debug/ctc_vllm_validation/run_vllm_eval*.py.")

    if args.backend == "olmo-eval":
        return _run_olmo_eval(args, variant)

    if args.query_position != "both":
        print(f"NOTE  query_position={args.query_position}: numbers are NOT comparable with "
              f"'both' runs. Recorded in the ledger.", file=sys.stderr)

    tasks_base = IN_DIST + (OOD if args.ood else [])
    rungs = [r.strip() for r in args.xlong_rungs.split(",") if r.strip()]
    bad = [r for r in rungs if r not in YARN]
    if bad:
        raise SystemExit(f"unknown xlong rung(s) {bad}; have {', '.join(YARN)}")
    passes = build_passes(tasks_base, rungs, args.xlong)

    git_commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or "unknown"

    print(f"=== {args.run_name} | arm={variant} | backend={args.backend} | "
          f"{len(passes)} passes ===")
    for p in passes:
        print(f"  {p['tag']:<14} {len(p['tasks'])} tasks  yarn={p['yarn']}  "
              f"rungs={','.join(p['rungs']) or 'base 2k-32k'}")
    if not args.xlong:
        print("  ⚠ standing rule 1 OVERRIDDEN: no xlong rungs")
    if not args.ood:
        print("  ⚠ standing rule 3 OVERRIDDEN: no OOD ladders")

    led_dir = os.path.join(REPO, "records", "eval_launches")
    os.makedirs(led_dir, exist_ok=True)
    led_path = os.path.join(
        led_dir, f"{datetime.date.today().isoformat()}_{args.run_name}.yaml")
    text = ledger(args, variant, passes, git_commit)
    if args.dry_run:
        print(f"\n--- ledger (NOT written; --dry-run) -> {led_path}\n{text}")
    else:
        with open(led_path, "w") as f:
            f.write(text)
        print(f"\nledger -> {led_path}")

    for p in passes:
        cmd = [sys.executable, LAUNCHER, args.run_name, args.cluster,
               "--task", ",".join(p["tasks"]), "--ckpt", args.ckpt,
               "--query-position", args.query_position, "--prompt-format", args.chat_template,
               "--ladder-version", args.ladder_version, "--priority", args.priority,
               "--variant", variant, "--max-length", str(args.max_length)]
        if args.tokenizer:
            cmd += ["--tokenizer", args.tokenizer]
        if p["xlong"]:
            cmd += ["--xlong", "--xlong-rungs", ",".join(p["rungs"])]
        if args.dry_run:
            cmd += ["--dry-run"]
        print(f"\n=== pass {p['tag']} ===\n{' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")}).returncode
        if rc != 0:
            print(f"  pass {p['tag']} FAILED rc={rc}", file=sys.stderr)
            raise SystemExit(rc)

    print(f"\n{len(passes)} passes {'planned' if args.dry_run else 'submitted'}. "
          f"Score them with the `pull-evals` skill, which reads the ledger back.")


def _run_olmo_eval(args, variant: str) -> None:
    """Plan, pack, ledger and submit the olmo-eval backend (see olmo_eval_backend.py)."""
    import olmo_eval_backend as OE

    rows = (OE.SETA_ROWS + list(OE.OOD_ROWS)) if args.rows == "setA" else [
        r.strip() for r in args.rows.split(",") if r.strip()]
    policy = OE.parse_policy(args.policy or OE.DEFAULT_POLICY)
    budget_s = args.budget_hours * 3600 - args.setup_minutes * 60
    cells = OE.plan(rows, policy)
    bins = OE.pack(cells, budget_s)
    print(f"=== {args.run_name} | arm={variant} | backend=olmo-eval (native olmo_core, chat) ===")
    print(OE.describe(bins, budget_s))

    out_root = os.path.join("/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_olmoeval_ctc",
                            args.run_name)
    git_commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or "unknown"
    passes = [{"tag": f"olmoeval_job{i:02d}", "tasks": [c.task for c in b], "rungs": [],
               "yarn": 1} for i, b in enumerate(bins)]
    args.eval_bundle = "hf://PrasannSinghal/ctc-suite-eval (olmo-eval, CTC_SUITE_PROMPT_FORMAT=chat)"
    text = ledger(args, variant, passes, git_commit) + (
        f"olmo_eval:\n  results_root: {out_root}\n  policy: {args.policy or OE.DEFAULT_POLICY}\n"
        f"  budget_hours: {args.budget_hours}\n  source_dataset: {args.olmo_eval_dataset}\n")
    led_dir = os.path.join(REPO, "records", "eval_launches")
    led_path = os.path.join(led_dir, f"{datetime.date.today().isoformat()}_{args.run_name}.yaml")
    if args.dry_run:
        print(f"\n--- ledger (NOT written; --dry-run) -> {led_path}")
    else:
        os.makedirs(led_dir, exist_ok=True)
        with open(led_path, "w") as f:
            f.write(text)
        print(f"\nledger -> {led_path}")
    names = OE.submit(args.run_name, bins, args.ckpt, out_root,
                      tokenizer=args.tokenizer or "Qwen/Qwen3.5-0.8B", max_model_len=262144,
                      olmo_eval_dataset=args.olmo_eval_dataset, cluster=args.cluster,
                      priority=args.priority, dry_run=args.dry_run)
    print(f"\n{len(names)} jobs {'planned' if args.dry_run else 'submitted'}: "
          f"{names[0]} .. {names[-1]}")


if __name__ == "__main__":
    main()
