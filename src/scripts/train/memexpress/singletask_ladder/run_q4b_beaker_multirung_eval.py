"""
Submit a multi-rung NATIVE long-context eval as a **Beaker** job for one trained checkpoint --
fully on Beaker, NO local sync. This is the on-Beaker counterpart of the LOCAL driver
``run_q4b_stl_multirung_eval.sbatch``.

The eval reads everything from weka (eval code + eval data + the distcp checkpoint), so the job needs
no data copied to the node:

  * eval CODE + DATA + the goal-rung ladder files live in the weka eval bundle
    (``checkpoints/prasanns/_eval_bundle`` + ``_eval_bundle_eval500``); upload/refresh it with
    ``upload_lc_eval_bundle.sh`` before launching.
  * the checkpoint is the just-trained distcp step dir under
    ``checkpoints/prasanns/<run_name>/step*`` -- the on-node runner auto-globs the latest complete step.

The actual eval logic is the on-node runner ``run_beaker_multirung_eval.sh`` (uploaded to the bundle):
it runs ``torchrun --nproc_per_node=8 scripts/eval/eval_lc_native.py`` (8-way DP, native olmo_core
generate -- NO HF/vLLM, required for landmark/compressive). gantry's own torchrun wrapping is disabled
(``torchrun=False``) so the runner can issue its own (and multiple, for rerank) torchrun calls.

Per-task rungs (matching the local driver):
  contra 2k,8k,16k,32k | nq 3k,8k,16k,32k | outlier 3k,8k,16k,32k | oolong 8k,16k,32k |
  rerank CE files k20/k50/k100 (NDCG@10 + Kendall-tau). ``--max-test-samples 600``.

Usage::

    # one (run, task) -> one Beaker job; variant inferred from the run name
    PYTHONPATH=src python src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py \\
        q4b-dense-contra-ladder32k-10k ai2/neptune --task contra

    # all 5 tasks for a run (5 jobs)
    PYTHONPATH=src python src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py \\
        q4b-landmark-nq-ladder32k-10k ai2/neptune --task all

    # validate without submitting
    PYTHONPATH=src python src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py \\
        q4b-dense-contra-ladder32k-10k ai2/neptune --task contra --dry-run
"""

import argparse
import sys

from olmo_core.internal.common import build_launch_config, get_root_dir
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.utils import prepare_cli_environment

ALL_TASKS = [
    "contra",
    "nq",
    "rerank",
    "outlier",
    "oolong",
    "fiqa",
    "scifact",
    "outlier_review",
    "contra_fever",
]
VARIANTS = ["dense", "landmark", "compressive", "docchunk"]


def variant_from_run_name(run_name: str) -> str:
    # docchunk run names use the explicit "docchunk_dense" token; check docchunk first.
    if "docchunk" in run_name:
        return "docchunk"
    found = [v for v in ("dense", "landmark", "compressive") if v in run_name]
    if len(found) != 1:
        raise SystemExit(
            f"could not infer variant from run name {run_name!r} (found {found}); pass --variant."
        )
    return found[0]


def build_eval_launch_config(
    *,
    run_name,
    task,
    variant,
    cluster,
    step,
    ckpt,
    results_dir,
    prompt_format,
    query_position,
    ngpu,
    max_test,
    max_length,
    batch_size,
    priority,
    ladder_version,
    xlong,
    xlong_rungs,
    cot_mode,
    tokenizer="",
 dc_rung_files: str = "", dc_rungs: str = ""):
    root_dir = get_root_dir(cluster)  # e.g. /weka/oe-training-default/ai2-llm (mounts weka bucket)
    # Eval CODE now ships IN the cloned repo (src/scripts/ctc_eval); the runner runs from the repo root
    # (gantry cwd). DATA still comes from weka (the runner derives BUNDLE/EVAL500 from WEKA_LLM=root_dir).
    runner = "src/scripts/train/memexpress/singletask_ladder/run_beaker_multirung_eval.sh"

    # The on-node runner reads its inputs from env; gantry torchrun wrapping is disabled so the runner
    # can drive its own 8-way `torchrun`. cmd[0]="bash" => not auto-prefixed with `python`.
    inner = (
        f"RUN={run_name} TASK={task} VARIANT={variant} STEP='{step}' CKPT='{ckpt}' "
        f"EVAL_OUT_DIR='{results_dir}' PROMPT_FORMAT='{prompt_format}' "
        f"QUERY_POSITION='{query_position}' "
        f"MAX_TEST={max_test} MAX_LENGTH={max_length} BATCH_SIZE={batch_size} NGPU={ngpu} "
        f"LADDER_XLONG={int(xlong)} XLONG_RUNGS='{xlong_rungs}' COT_MODE='{cot_mode}' "
        f"LADDER_VERSION={ladder_version} "
        + (f"TOKENIZER={tokenizer} " if tokenizer else "")
        + (f"DC_RUNG_FILES='{dc_rung_files}' " if dc_rung_files else "")
        + (f"DC_RUNGS='{dc_rungs}' " if dc_rungs else "")
        + f"WEKA_LLM={root_dir} bash {runner}"
    )
    cmd = ["bash", "-lc", inner]

    # Ladder in the job name for anything but v2, so a v2 and a v3 submission for the same
    # (task, run) are distinguishable in the Beaker UI instead of two identically-named jobs.
    name = f"ev-{task}-{run_name}" + (f"-{ladder_version}" if ladder_version != "v2" else "")
    launch_config = build_launch_config(
        name=name,
        cmd=cmd,
        cluster=cluster,
        root_dir=root_dir,
        task_name="eval",
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=1,
        num_gpus=ngpu,
    )
    launch_config.torchrun = False  # the runner issues its own torchrun(s)
    launch_config.allow_dirty = True  # ship the (uncommitted) launcher via an ephemeral ref
    launch_config.priority = priority
    launch_config.step_soft_timeout = (
        None  # we submit with follow=False (don't block on many evals);
    )
    launch_config.step_timeout = None  # the 10-min default soft timeout forbids follow=False
    return launch_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_name", help="trained run name (checkpoints under checkpoints/prasanns/<run_name>)"
    )
    ap.add_argument("cluster", help="Beaker cluster, e.g. ai2/neptune (weka-backed)")
    ap.add_argument(
        "--task",
        default="all",
        help=f"comma list from {ALL_TASKS}, or 'all' (one Beaker job per task).",
    )
    ap.add_argument(
        "--variant",
        default=None,
        choices=VARIANTS,
        help="dense|landmark|compressive|docchunk (default: infer from run name).",
    )
    ap.add_argument(
        "--step", default="", help="pin a step dir (e.g. step580); default = latest complete."
    )
    ap.add_argument(
        "--ckpt",
        default="",
        help="ABSOLUTE weka step dir to eval ANY checkpoint, e.g. "
        "/weka/oe-training-default/ai2-llm/checkpoints/<you>/<run>/step1234 . "
        "Overrides run_name globbing (run_name is then just a results label).",
    )
    ap.add_argument(
        "--results-dir",
        default="",
        help="ABSOLUTE weka dir for the per-task result JSONs "
        "(default: checkpoints/prasanns/<run_name>/eval).",
    )
    ap.add_argument(
        "--query-position",
        choices=["both", "after", "before"],
        default="both",
        help="Prompt layout; MUST match the SFT shards the model was trained on. "
        "xlong5_2k256k_qwen35 -> both (default, and what every result before "
        "2026-08-11 used); xlong5_2k256k_qwen35_qafter -> after. It lands in the "
        "inner eval_command, so the launch ledger records it automatically.",
    )
    ap.add_argument(
        "--prompt-format",
        choices=["chat", "raw", "alpaca"],
        default="chat",
        help="chat=SFT (apply_chat_template, matches training); raw=BASE/CPT models; alpaca=legacy.",
    )
    ap.add_argument("--max-test", type=int, default=600)
    ap.add_argument("--max-length", type=int, default=40960)
    ap.add_argument(
        "--batch-size", type=int, default=2
    )  # 40960-ctx generation on ~48GB neptune GPUs; 8 OOMs
    ap.add_argument(
        "--ngpu",
        type=int,
        default=2,
        help="GPUs per eval job (data-parallel over examples). 4B model fits on 1-2 GPUs; "
        "2 lets ~4x more evals run concurrently than 8 and fits fragmented free slots.",
    )
    ap.add_argument(
        "--tokenizer",
        default="",
        help="HF tokenizer id, forwarded to the on-node runner as TOKENIZER. "
        "MUST match the model family: the runner defaults to Qwen/Qwen3-4B "
        "(vocab 151936), so a Qwen3.5 checkpoint (vocab 248320) evaluated "
        "without this scores ~0 on EVERY task while the job reports success -- "
        "wrong token ids, not a broken model. Qwen3.5 -> Qwen/Qwen3.5-4B-Base.",
    )
    ap.add_argument("--priority", default="urgent")  # never below urgent (user directive)
    ap.add_argument(
        "--ladder-version",
        choices=["v2", "v3"],
        default="v2",
        help="v2 is the ONLY supported ladder: every rung of a task shares the SAME "
        "500 questions/answers and only distractors vary (reads the "
        "_eval_bundle_eval500_v2 weka bundle). v1 is DISABLED -- its per-rung "
        "question resampling put eval-set noise into every rung-to-rung delta, "
        "and both the runner and eval_lc_native.py now reject it.",
    )
    ap.add_argument(
        "--cot-mode",
        choices=["none", "plan"],
        default="none",
        help="docchunk OOLONG only: 'plan' builds the CoT prefill (match a CoT-trained "
        "checkpoint); default 'none' keeps the no-CoT eval byte-identical.",
    )
    ap.add_argument(
        "--xlong",
        action="store_true",
        help="OPT-IN: also run the ultra-long 64k/128k/256k/512k/1M/2M rungs (contra|nq|outlier). "
        "Forces bs=1 + raises MAX_LENGTH on-node. Use an 80GB GPU (ai2/jupiter); "
        "256k needs bs=1 single-GPU. Files must be built by build_xlong_rungs.py "
        "and uploaded to the v2 eval bundle.",
    )
    ap.add_argument(
        "--xlong-rungs",
        default="64k,128k",
        help="which xlong sizes to add when --xlong: 64k,128k,256k,512k,1M,2M. Anything above "
        "256k needs a YaRN serving copy (past Qwen3.5's native 262,144) and more "
        "than one 80GB GPU (KV ~32KB/token: 2M alone is ~69GB).",
    )
    ap.add_argument(
        "--dc-rung-files",
        default="",
        help="docchunk variant only: JSON {task: {label: weka path}} of explicit rung files "
        "(overrides the bundle's defaults; used to score marker-trained models on the exact "
        "rung files a dense campaign used)",
    )
    ap.add_argument("--dc-rungs", default="", help="docchunk: comma rung labels to score (pairs with --dc-rung-files)")
    ap.add_argument("--dry-run", action="store_true", help="build + print the job, do NOT submit.")
    args = ap.parse_args()

    prepare_cli_environment()

    variant = args.variant or variant_from_run_name(args.run_name)
    tasks = (
        ALL_TASKS if args.task == "all" else [t.strip() for t in args.task.split(",") if t.strip()]
    )
    bad = [t for t in tasks if t not in ALL_TASKS]
    if bad:
        raise SystemExit(f"unknown task(s) {bad}; choose from {ALL_TASKS}.")

    print(
        f"=== Beaker multirung eval | run={args.run_name} variant={variant} "
        f"tasks={tasks} cluster={args.cluster} dry_run={args.dry_run} ==="
    )
    for task in tasks:
        # docchunk now evaluates the FULL ladder (all 9 tasks incl. OOD) via
        # eval_lc_native_docchunk_ladder.py (box-marker chunked prefill + bs=1 KV-cached decode).
        lc = build_eval_launch_config(
            run_name=args.run_name,
            task=task,
            variant=variant,
            cluster=args.cluster,
            step=args.step,
            ckpt=args.ckpt,
            results_dir=args.results_dir,
            prompt_format=args.prompt_format,
            query_position=args.query_position,
            ngpu=args.ngpu,
            max_test=args.max_test,
            max_length=args.max_length,
            batch_size=args.batch_size,
            priority=args.priority,
            ladder_version=args.ladder_version,
            xlong=args.xlong,
            xlong_rungs=args.xlong_rungs,
            cot_mode=args.cot_mode,
            tokenizer=args.tokenizer,
            dc_rung_files=args.dc_rung_files,
            dc_rungs=args.dc_rungs,
        )
        print(f"\n--- [{task}] {lc.name} ---")
        print(f"    cmd: {lc.cmd[-1]}")
        if args.dry_run:
            print("    [dry-run] not submitting.")
            continue
        workload = lc.launch(follow=False)
        print(f"    submitted: {getattr(workload, 'id', workload)}")
    print("\n=== done ===")


if __name__ == "__main__":
    main()
