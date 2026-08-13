"""
Submit the multi-rung long-context eval for a **HuggingFace-format** model as a Beaker job.

The hf-backend twin of
``src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py``: same eval
bundle, same ladder table, same scorer, one Beaker job per ``(model, task)``. It exists because
HiLS-Attention cannot be expressed in olmo_core, and a third-party model is only worth measuring
if it is measured on the SAME ladder as the models it will be compared against.

The on-node half is ``run_beaker_hf_eval.sh``, which installs the HiLS runtime when the checkpoint
needs it and drives ``torchrun --nproc_per_node=N eval_lc_native.py --backend hf``.

Models must be weka-staged first (``src/scripts/data/stage_hf_models_weka.py``) -- never a Hub id,
which would make every job depend on huggingface.co being up at startup.

Usage::

    # Pass A: base ladder, all 9 tasks (raw prompts -- these are BASE models)
    PYTHONPATH=src python src/scripts/train/memexpress/hils_eval/run_hf_beaker_eval.py \\
        --model /weka/oe-training-default/amandab/hf_models/tencent__HiLS-Attention-7B \\
        --model-name hils-7b ai2/jupiter-cirrascale-2 --task all \\
        --prompt-format raw --eval-tag base-raw

    # Pass B: xlong rungs, in-distribution tasks only
    ... --task contra,nq,outlier,rerank,oolong --xlong --xlong-only --xlong-rungs 64k,128k \\
        --eval-tag xlong-raw
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
# The four OOD ladders have no xlong rung files; sending them through --xlong would silently
# re-run the base ladder under an xlong tag (a duplicate row, not a new measurement).
XLONG_TASKS = ["contra", "nq", "outlier", "rerank", "oolong"]

DEFAULT_RESULTS_ROOT = "/weka/oe-training-default/amandab/hf_evals"


def build_eval_launch_config(
    *,
    model,
    model_name,
    task,
    cluster,
    results_dir,
    prompt_format,
    chat_template,
    attn_impl,
    tokenizer,
    query_position,
    ngpu,
    max_test,
    max_length,
    batch_size,
    priority,
    ladder_version,
    xlong,
    xlong_only,
    xlong_rungs,
    eval_tag,
):
    root_dir = get_root_dir(cluster)
    runner = "src/scripts/train/memexpress/hils_eval/run_beaker_hf_eval.sh"

    inner = (
        f"MODEL='{model}' MODEL_NAME='{model_name}' TASK={task} "
        f"EVAL_OUT_DIR='{results_dir}' PROMPT_FORMAT='{prompt_format}' "
        f"CHAT_TEMPLATE='{chat_template}' ATTN_IMPL='{attn_impl}' TOKENIZER='{tokenizer}' "
        f"QUERY_POSITION='{query_position}' "
        f"MAX_TEST={max_test} MAX_LENGTH={max_length} BATCH_SIZE={batch_size} NGPU={ngpu} "
        f"LADDER_XLONG={int(xlong)} XLONG_ONLY={int(xlong_only)} "
        f"XLONG_RUNGS='{xlong_rungs}' EVAL_TAG='{eval_tag}' "
        f"LADDER_VERSION={ladder_version} WEKA_LLM={root_dir} bash {runner}"
    )
    cmd = ["bash", "-lc", inner]

    name = f"ev-{task}-{model_name}" + (f"-{eval_tag}" if eval_tag else "")
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
    launch_config.torchrun = False  # the runner issues its own torchrun
    launch_config.allow_dirty = True
    launch_config.priority = priority
    launch_config.step_soft_timeout = None
    launch_config.step_timeout = None
    return launch_config, inner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cluster", help="Beaker cluster, e.g. ai2/jupiter-cirrascale-2 (weka-backed)")
    ap.add_argument("--model", required=True, help="ABSOLUTE weka path to the HF checkpoint dir.")
    ap.add_argument(
        "--model-name",
        required=True,
        help="short label for result files and the Beaker job name, e.g. hils-7b.",
    )
    ap.add_argument("--task", default="all", help=f"comma list from {ALL_TASKS}, or 'all'.")
    ap.add_argument(
        "--prompt-format",
        choices=["raw", "chat"],
        default="raw",
        help="raw = bare prompt as a completion (correct for BASE models, which is what both "
        "HiLS-7B and Olmo-3-1025-7B are); chat = wrap in a chat template.",
    )
    ap.add_argument(
        "--chat-template",
        default="",
        help="jinja file for --prompt-format chat. Empty -> the runner's default "
        "(src/scripts/ctc_eval/lib/chat_templates/olmo3_chatml.jinja).",
    )
    ap.add_argument(
        "--attn-impl",
        default="",
        help="dense-layer attention impl (flash_attention_3|flash_attention_2|sdpa|eager). "
        "Empty -> the harness probes fa3, then fa2, then sdpa.",
    )
    ap.add_argument(
        "--tokenizer", default="", help="Empty (default) -> the model dir's own tokenizer."
    )
    ap.add_argument("--query-position", choices=["both", "after", "before"], default="both")
    ap.add_argument("--results-dir", default="", help=f"default: {DEFAULT_RESULTS_ROOT}/<name>")
    ap.add_argument("--ngpu", type=int, default=8)
    ap.add_argument("--max-test", type=int, default=600)
    ap.add_argument("--max-length", type=int, default=40960)
    # 4, not 8: the hf backend has no chunked prefill, and a 7B at the 32k rung with batch 8 sits
    # close enough to 80 GB that the control OOM'd there. Batch size changes speed, not scores.
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--priority", default="urgent")
    ap.add_argument("--ladder-version", default="v2", choices=["v2", "v3", "fast"])
    ap.add_argument("--xlong", action="store_true", help="add the ultra-long rungs.")
    ap.add_argument(
        "--xlong-only",
        action="store_true",
        help="replace the base rungs instead of appending (use once Pass A is submitted).",
    )
    ap.add_argument("--xlong-rungs", default="64k,128k")
    ap.add_argument(
        "--eval-tag",
        default="",
        help="suffixes result files AND the job name, so parallel conditions on the same "
        "checkpoint (raw vs chat, base vs xlong) never overwrite each other. Effectively "
        "required here, since every model runs in at least two prompt conditions.",
    )
    ap.add_argument("--dry-run", action="store_true", help="print, do not submit.")
    args = ap.parse_args()

    prepare_cli_environment()

    tasks = ALL_TASKS if args.task == "all" else [t.strip() for t in args.task.split(",")]
    unknown = [t for t in tasks if t not in ALL_TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}; valid: {ALL_TASKS}")
    if args.xlong:
        dropped = [t for t in tasks if t not in XLONG_TASKS]
        if dropped:
            # Silently running these would produce a base-ladder duplicate wearing an xlong tag.
            raise SystemExit(
                f"tasks {dropped} have no xlong rung files -- drop them from an --xlong "
                f"submission (xlong tasks: {XLONG_TASKS})."
            )
    if not args.eval_tag:
        print(
            "WARNING: no --eval-tag. Two conditions on the same model (raw vs chat, base vs "
            "xlong) will write to the SAME result files and the second will overwrite the first.",
            file=sys.stderr,
        )

    results_dir = args.results_dir or f"{DEFAULT_RESULTS_ROOT}/{args.model_name}"

    for task in tasks:
        launch_config, inner = build_eval_launch_config(
            model=args.model,
            model_name=args.model_name,
            task=task,
            cluster=args.cluster,
            results_dir=results_dir,
            prompt_format=args.prompt_format,
            chat_template=args.chat_template,
            attn_impl=args.attn_impl,
            tokenizer=args.tokenizer,
            query_position=args.query_position,
            ngpu=args.ngpu,
            max_test=args.max_test,
            max_length=args.max_length,
            batch_size=args.batch_size,
            priority=args.priority,
            ladder_version=args.ladder_version,
            xlong=args.xlong,
            xlong_only=args.xlong_only,
            xlong_rungs=args.xlong_rungs,
            eval_tag=args.eval_tag,
        )
        print(f"\n--- [{task}] {launch_config.name} ---")
        print(f"    {inner}")
        if args.dry_run:
            continue
        workload = launch_config.launch(follow=False)
        # BeakerWorkload is a protobuf with fields (experiment, environment, status, budget_id) --
        # it has no `.id`, and printing the object itself dumps a multi-line repr instead of an
        # identifier. The experiment id is what `pull-evals` queries, so it has to land in the
        # ledger cleanly; a ledger entry without one is barely better than no entry.
        exp_id = getattr(getattr(workload, "experiment", None), "id", None)
        print(f"    submitted: {exp_id or workload}")


if __name__ == "__main__":
    main()
