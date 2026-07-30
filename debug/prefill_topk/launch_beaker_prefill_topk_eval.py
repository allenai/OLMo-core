"""
Submit the PREFILL-WIDE top-k landmark sweep as a Beaker job (weka-mounted, nothing synced locally).

Adapted from ``src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py``; the
only differences are the on-node runner (``debug/prefill_topk/run_beaker_prefill_topk_eval.sh``) and
the per-config sweep it drives. The production eval scripts are untouched.

⚠ gantry clones the repo at the CURRENT COMMIT -- commit AND push
``src/olmo_core/nn/attention/landmark_prefill_topk.py`` plus this directory before launching, or the
job runs old code (see the ``gantry-needs-pushed-commit`` trap).

Usage::

    PYTHONPATH=src python debug/prefill_topk/launch_beaker_prefill_topk_eval.py \\
        q4b-compressive-5task-32k-nocpt-fixdata ai2/jupiter --dry-run

    PYTHONPATH=src python debug/prefill_topk/launch_beaker_prefill_topk_eval.py \\
        q4b-compressive-5task-32k-nocpt-fixdata ai2/jupiter
"""

import argparse

from olmo_core.internal.common import build_launch_config, get_root_dir
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.utils import prepare_cli_environment

RUNNER = "debug/prefill_topk/run_beaker_prefill_topk_eval.sh"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_name", help="run under checkpoints/prasanns/<run_name> (checkpoint + label)")
    ap.add_argument("cluster", help="Beaker cluster, e.g. ai2/jupiter (weka-backed, H100)")
    ap.add_argument("--task", default="contradiction", help="ladder task key (default contradiction)")
    ap.add_argument("--rungs", default="2k,8k,16k,32k")
    ap.add_argument("--configs", default="",
                    help="';'-separated '<tag>|<extra eval flags>'; empty = the runner's default sweep "
                         "(baseline, prefill top-k at 10/25/50%%, and 10%% with a hard drop).")
    ap.add_argument("--step", default="", help="pin a step dir (e.g. step580); default = latest complete.")
    ap.add_argument("--ckpt", default="", help="ABSOLUTE weka step dir; overrides run_name globbing.")
    ap.add_argument("--results-dir", default="",
                    help="ABSOLUTE weka dir for result JSONs (default <run>/eval_prefill_topk).")
    ap.add_argument("--prompt-format", choices=["chat", "raw", "alpaca"], default="chat")
    ap.add_argument("--max-test", type=int, default=600)
    ap.add_argument("--max-length", type=int, default=40960)
    ap.add_argument("--ngpu", type=int, default=2,
                    help="GPUs per job (data-parallel over examples); the 4B model fits on 1-2.")
    ap.add_argument("--priority", default="urgent")  # never below urgent (user directive)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prepare_cli_environment()
    root_dir = get_root_dir(args.cluster)

    inner = (
        f"RUN={args.run_name} TASK={args.task} RUNGS='{args.rungs}' "
        f"STEP='{args.step}' CKPT='{args.ckpt}' EVAL_OUT_DIR='{args.results_dir}' "
        f"CONFIGS='{args.configs}' PROMPT_FORMAT='{args.prompt_format}' "
        f"MAX_TEST={args.max_test} MAX_LENGTH={args.max_length} NGPU={args.ngpu} "
        f"WEKA_LLM={root_dir} bash {RUNNER}"
    )
    lc = build_launch_config(
        name=f"evptk-{args.task}-{args.run_name}"[:100],
        cmd=["bash", "-lc", inner],
        cluster=args.cluster,
        root_dir=root_dir,
        task_name="eval",
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=1,
        num_gpus=args.ngpu,
    )
    lc.torchrun = False  # the runner issues its own torchrun per config
    lc.allow_dirty = True
    lc.priority = args.priority
    lc.step_soft_timeout = None
    lc.step_timeout = None

    print(f"--- {lc.name} ---\n    cmd: {lc.cmd[-1]}")
    if args.dry_run:
        print("    [dry-run] not submitting.")
        return
    workload = lc.launch(follow=False)
    print(f"    submitted: {getattr(workload, 'id', workload)}")


if __name__ == "__main__":
    main()
