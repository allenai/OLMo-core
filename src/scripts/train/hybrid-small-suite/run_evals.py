# uv run /Users/yashasbls/Desktop/OLMo-core-all/hybrid-final/src/scripts/train/hybrid-small-suite/run_evals.py --dry-run

import argparse
import subprocess
import time

GROUP = "yashasbls-hybrid-small-evals-v4"
CLUSTER = "ai2/jupiter"
PRIORITY = "urgent"
NUM_GPUS = 1
WORKSPACE = "ai2/linear-rnns"
BUDGET = "ai2/oe-other"

CKPT_BASE = "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls"

# Each stage maps size -> list of HF checkpoint paths.
# Single-model stages have one-element lists; grid searches have multiple.
all_stages: dict[str, dict[str, list[str]]] = {
    "pretraining": {
        "275m": [f"{CKPT_BASE}/hybrid-small-275M-Cx100/step161186-hf/"],
        "810m": [f"{CKPT_BASE}/hybrid-small-810M-Cx100/step269926-hf/"],
        "1.4b": [f"{CKPT_BASE}/hybrid-small-1.4B-Cx100/step308433-hf/"],
    },
    "midtraining": {
        "275m": [f"{CKPT_BASE}/hybrid-small-midtraining-275M-v2-lr1.6e-3/step38147-hf/"],
        "810m": [f"{CKPT_BASE}/hybrid-small-midtraining-v2-810M-lr4e-4/step23842-hf/"],
        "1.4b": [f"{CKPT_BASE}/hybrid-small-midtraining-v2-1.4b-lr4e-4/step11921-hf/"],
    },
    "long_context": {
        "275m": [f"{CKPT_BASE}/hybrid-small-long-context-v2-275m/step47684-hf/"],
        "810m": [f"{CKPT_BASE}/hybrid-small-long-context-v2-810m/step23842-hf/"],
        "1.4b": [f"{CKPT_BASE}/hybrid-small-long-context-v2-1.4b/step23842-hf/"],
    },
    "long_context_debug_v3": {
        "275m": [
            f"{CKPT_BASE}/hybrid-small-lc-v3-275m-lr1.6e-3/step47684-hf/",
            f"{CKPT_BASE}/hybrid-small-lc-v3-275m-lr8e-4/step47684-hf/",
            f"{CKPT_BASE}/hybrid-small-lc-v3-275m-lr4e-4/step47684-hf/",
            f"{CKPT_BASE}/hybrid-small-lc-v3-275m-lr2e-4/step47684-hf/",
        ],
    },
    # "sft_think": {
    #     "275m": [
    #         f"{CKPT_BASE}/hybrid-small-sft-think-275M-lr{lr}/step23206-hf/"
    #         for lr in ["1e-4", "2e-4", "4e-4", "8e-4"]
    #     ],
    # },
}

# (task_name, num_gpus)
TEST_TASKS = [("olmobase:easy:qa:rc", 1)]

PPLX_TASKS = [
    ("c4_100k:ppl", 1),
]

DOWNSTREAM_TASKS = [
    # Math — 8 GPUs (10h single-GPU)
    ("olmobase:math", 8),
    # GenQA — 4 GPUs (3.5-7.5h single-GPU)
    ("olmobase:gen", 4),
    # MC Non-STEM — 2 GPUs (2-5h single-GPU)
    ("olmobase:mcqa_non_stem", 1),
    # MC STEM — 1 GPU (~1h)
    ("olmobase:mcqa_stem", 1),
    # Code — 1 GPU (~10min)
    ("olmobase:easy:code:bpb", 1),
    # LBPP, BBH, MMLU Pro, DM Math — not yet in olmo-eval-internal
    ("olmobase:easy:qa:rc", 2)
]

LC_TASKS = [
    ("ruler_all__4096", 1),
    ("ruler_all__8192", 1),
    ("ruler_all__16384", 1),
    ("ruler_all__32768", 1),
    ("ruler_all__65536", 1),
    ("ruler_all__131072", 1),
]

POSTTRAIN_TASKS = [
    # Knowledge/Reasoning
    ("mmlu", 1),                           # MMLU
    # ("popqa", 1),                         # PopQA — not yet in olmo-eval-internal
    # ("bbh", 1),                           # BBH — not yet in olmo-eval-internal
    ("gpqa_diamond", 1),                   # GPQA
    ("zebralogic:chat", 1),                # Zebra Logic
    # Math
    ("aime_2024:pass_at_32", 1),           # AIME'24
    ("aime_2025:pass_at_32", 1),           # AIME'25
    ("math500", 1),                        # MATH Ω
    # Code (require sandbox)
    ("humaneval_plus:chat:pass_at_1", 1),  # HE+
    ("mbpp_plus:pass_at_1", 1),            # MBPP+
    # ("livecode_bench", 1),               # LCB — not yet in olmo-eval-internal
    ("deepseek_leetcode", 1),              # LCB proxy
    # Instruction Following
    ("ifeval_ood", 1),                     # IFEval
    ("ifbench", 1),                        # IFBench
    # ("arena_eval_3", 1),                  # AE3 — not yet in olmo-eval-internal
]

SAFETY_TASKS: list[tuple[str, int]] = [
    # TODO(yashasbls): ask maliam
]


def build_command(
    model_path: str,
    tasks: list[str],
    num_gpus: int = NUM_GPUS,
    is_lc: bool = False,
    group: str = GROUP,
) -> list[str]:
    # Derive a clean name: last two path components joined with underscore, slashes removed
    parts = model_path.rstrip("/").split("/")
    model_short = "_".join(parts[-2:]).lower()
    tasks_short = "-".join(t.replace(":", "_") for t in tasks[:2])
    if len(tasks) > 2:
        tasks_short += f"-and-{len(tasks) - 2}-more"
    exp_name = f"{model_short}-{tasks_short}"

    cmd = ["uv", "run", "olmo-eval", "beaker", "launch"]
    cmd += ["-H", "default"]
    cmd += ["-n", exp_name]
    cmd += ["-o", f"provider.num_instances={num_gpus}"]
    cmd += ["-o", "provider.kwargs.enforce_eager=true"]
    cmd += ["-o", "provider.kwargs.mamba_ssm_cache_dtype=float32"]
    cmd += ["-o", "provider.add_bos_token=false"]
    cmd += ["-o", "provider.kind=vllm"]
    cmd += ["-o", "provider.package=wheel"]
    cmd += ["-o", "provider.dependencies=[transformers @ git+https://github.com/yashassamaga/transformers.git@hybrid-small-suite]"]
    cmd += ["-o", "provider.kwargs.attention_backend=FLASH_ATTN"]
    if is_lc:
        cmd += ["-o", "provider.max_model_len=131072"]
    cmd += ["-m", model_path]
    for task in tasks:
        cmd += ["-t", task]
    cmd += ["--gpus", str(num_gpus)]
    cmd += ["--retries", "3"]
    cmd += ["--priority", PRIORITY]
    cmd += ["--group", group]
    cmd += ["--cluster", CLUSTER]
    cmd += ["--workspace", WORKSPACE]
    cmd += ["--budget", BUDGET]
    # cmd += ["--store"]
    cmd += ["--inspect"]
    # cmd += ["--gcp-credentials"]
    cmd += ["--image", "yashasbls/olmo-eval-vllm-g79d31a3f9-tch2100cu128-2026-05-23"]
    cmd += ["--secret-env", "yashasbls_HF_TOKEN:HF_TOKEN"]
    # cmd += ["--env", "VLLM_PYTHON=/opt/vllm-venv/bin/python"]
    cmd += ["--env", "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"]
    cmd += ["--no-follow"]
    cmd += ["-y"]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["275m", "810m", "1.4b"],
        default=["275m"],
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(all_stages.keys()),
        default=["pretraining"],
    )
    parser.add_argument(
        "--eval-type",
        nargs="+",
        choices=["test", "downstream", "lc", "safety", "pplx", "posttrain"],
        default=["test"],
    )
    parser.add_argument("--group", type=str, default=GROUP, help="Beaker workgroup for launched jobs.")
    parser.add_argument("--gpus", type=int, default=NUM_GPUS, help="Number of GPUs per job.")
    parser.add_argument("--model", type=str, default=None, help="Custom model path or HF name (overrides --sizes/--stages)")
    parser.add_argument("--delay", type=int, default=0, help="Seconds to wait between launching each eval job")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    num_gpus = args.gpus
    group = args.group

    eval_type_map = {
        "test": TEST_TASKS,
        "downstream": DOWNSTREAM_TASKS,
        "lc": LC_TASKS,
        "safety": SAFETY_TASKS,
        "pplx": PPLX_TASKS,
        "posttrain": POSTTRAIN_TASKS,
    }
    tasks = []
    lc_task_names = {t[0] for t in LC_TASKS}
    for et in args.eval_type:
        tasks.extend(eval_type_map[et])

    launched = 0
    if args.model:
        # Custom model: run all tasks against this single model
        for task_name, task_gpus in tasks:
            gpus = num_gpus if num_gpus != NUM_GPUS else task_gpus
            cmd = build_command(args.model, [task_name], gpus, is_lc=task_name in lc_task_names, group=group)
            print(f"\n=== custom | {task_name} | {gpus} GPUs ===")
            print(" ".join(cmd))
            if not args.dry_run:
                if launched > 0 and args.delay > 0:
                    time.sleep(args.delay)
                subprocess.run(cmd, check=True)
                launched += 1
    else:
        for stage in args.stages:
            checkpoints = all_stages[stage]
            for size in args.sizes:
                if size not in checkpoints:
                    print(f"[skip] {stage} has no size {size}")
                    continue
                for model_path in checkpoints[size]:
                    short_name = model_path.rstrip("/").split("/")[-1]
                    for task_name, task_gpus in tasks:
                        gpus = num_gpus if num_gpus != NUM_GPUS else task_gpus
                        cmd = build_command(model_path, [task_name], gpus, is_lc=task_name in lc_task_names, group=group)
                        print(f"\n=== {stage}/{size}/{short_name} | {task_name} | {gpus} GPUs ===")
                        print(" ".join(cmd))
                        if not args.dry_run:
                            if launched > 0 and args.delay > 0:
                                time.sleep(args.delay)
                            subprocess.run(cmd, check=True)
                            launched += 1
                        launched += 1


if __name__ == "__main__":
    main()