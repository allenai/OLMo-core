"""Run eval_lc_native on BEAKER against a weka checkpoint + the weka-staged length-mix rungs.

Replaces the mooney push+local eval relay (user directive 2026-08-31: no mooney; Beaker for all
scaling runs). The command is byte-equivalent to the mooney wrap in cert_chaser.sh — same
tokenizer/ids/flags/rung labels — only the paths move to weka:

  ckpt:  <root>/checkpoints/prasanns/ctc_suite/ckpts/<savedir>       (written by the train job)
  rungs: <root>/checkpoints/prasanns/outlier_lengthmix/eval_rungs    (staged by weka-sync20; has
         outlier/, nq/, qdmatch_nq/ subdirs — one root serves all three ladder tasks)
  out:   /results (the Beaker result dataset; fetch with `beaker experiment results`)

Usage:
  python beaker_native_lengthmix_eval.py RUN SAVEDIR --ladder-tasks qdmatch_nq --ladder-rungs 32k,64k
"""

import argparse

from olmo_core.internal.common import build_launch_config, get_root_dir
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.utils import prepare_cli_environment


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_name")
    ap.add_argument("savedir", help="ckpt dir name under ctc_suite/ckpts/ (run-name-YYYY...)")
    ap.add_argument("--ladder-tasks", required=True)
    ap.add_argument("--ladder-rungs", required=True)
    ap.add_argument(
        "--eval500-root",
        default=None,
        help="weka-relative rung bundle under checkpoints/prasanns/ (default: "
        "outlier_lengthmix/eval_rungs, which holds ONLY outlier/nq/qdmatch_nq). "
        "contradiction lives in _eval_bundle_eval500_v3 (realistic mode, the one "
        "that matches our training generator), oolong + xabsence_exact in "
        "_eval_bundle_eval500_v2_clean. Pointing at the wrong bundle does not "
        "error -- every rung MISSING-skips and the job writes an empty JSON with "
        "exit 0.",
    )
    ap.add_argument(
        "--ladder-version",
        default="v2",
        choices=["v2", "v3"],
        help="v3 = realistic-mode contradiction gold. Use v3 for contradiction; the "
        "v2/both gold scores a realistic-trained ckpt ~0.39 f1 too low.",
    )
    ap.add_argument("--no-sparse-decode", action="store_true",
                    help="drop OLMO_LANDMARK_SPARSE_DECODE=1 (the ~2x fast decode path). NOTE "
                         "this does NOT make a sparse arm runnable on an L40S: the landmark "
                         "PREFILL kernel itself asks triton for 104KB of shared memory against "
                         "sm_89's 99KB limit, so neptune is out for every sparse-landmark "
                         "checkpoint regardless of this flag. Send sparse evals to an A100 (164KB) "
                         "or H100 (228KB) cluster.")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--ngpu", type=int, default=2,
                    help="Data-parallel eval ranks. Use 1 for a long rung whose per-example "
                         "generation length varies a lot (grouping at 32k with max_new=2048): the "
                         "two ranks shard the examples, one finishes far ahead of the other, and "
                         "the NCCL watchdog eventually aborts the job with a collective timeout "
                         "that reads as a hang. Single-rank has no collectives to desync.")
    ap.add_argument("--max-length", type=int, default=72000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = get_root_dir(args.cluster)
    ckpt = f"{root}/checkpoints/prasanns/ctc_suite/ckpts/{args.savedir}"
    ev500 = f"{root}/checkpoints/prasanns/{args.eval500_root or 'outlier_lengthmix/eval_rungs'}"
    # `export VAR=...;` rather than a `VAR=x cmd` prefix. The prefix form binds only to the FIRST
    # command in the chain, so once the scipy guard below was added ahead of torchrun, the eval ran
    # without PYTHONPATH and died on `No module named 'ctc_eval'`.
    inner = (
        "export PYTHONPATH=$PWD/src/scripts:$PWD/src; "
        f"export EVAL500_ROOT={ev500}; "
        + ("" if args.no_sparse_decode else "export OLMO_LANDMARK_SPARSE_DECODE=1; ")
        # Grader dependencies the stable image does not ship: reorder needs scipy
        # (kendalltau/spearmanr), grouping needs sklearn (adjusted_rand/NMI). Both used to surface
        # as ModuleNotFoundError AFTER the checkpoint had loaded -- minutes of GPU time to learn
        # about a missing wheel. Guarded, so a task that needs neither pays two import checks.
        + 'python -c "import scipy, sklearn" 2>/dev/null '
        + '|| pip install -q scipy scikit-learn; '
        + f"python -m torch.distributed.run --nproc_per_node={args.ngpu} --master_port=29513 "
        f"src/scripts/ctc_eval/eval/eval_lc_native.py --model-path {ckpt} "
        f"--out /results/{args.run_name}_native_multirung.json --tokenizer Qwen/Qwen3.5-0.8B "
        f"--max-length {args.max_length} --max-test-samples 600 --batch-size 1 --skip-ruler --skip-gen "
        f"--landmark-mem-id 248200 --landmark-pad-id 248203 --eos-token-id 248044 "
        f"--prompt-format chat --query-position after "
        f"--ladder --ladder-version {args.ladder_version} "
        f"--ladder-tasks {args.ladder_tasks} --ladder-rungs {args.ladder_rungs}"
    )
    lc = build_launch_config(
        name=f"nev-{args.run_name}",
        cmd=["bash", "-lc", inner],
        cluster=args.cluster,
        root_dir=root,
        task_name="eval",
        beaker_image=OLMoCoreBeakerImage.stable,
        workspace="ai2/flex2",
        budget="ai2/oe-other",
        num_nodes=1,
        num_gpus=args.ngpu,
    )
    lc.torchrun = False
    lc.allow_dirty = True
    lc.priority = "urgent"
    lc.step_soft_timeout = None
    lc.step_timeout = None
    print(f"--- {lc.name}\n    cmd: {inner}")
    if args.dry_run:
        print("    [dry-run] not submitting.")
        return
    workload = lc.launch(follow=False)
    wl_id = getattr(workload, "id", None) or getattr(getattr(workload, "experiment", None), "id", None)
    print(f"    submitted: {wl_id}")
    print(f"SUBMITTED id={wl_id}")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
