# uv run /Users/yashasbls/Desktop/OLMo-core-all/hybrid-final/src/scripts/train/hybrid-small-suite/run_evals.py --dry-run

import argparse
import subprocess

GROUP = "yashasbls-hybrid-small-tests"
CLUSTER = "ai2/jupiter"
PRIORITY = "urgent"
NUM_GPUS = 1
WORKSPACE = "ai2/linear-rnns"
BUDGET = "ai2/oe-omai"

pretraining_hf_checkpoints = {
    "275m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-275M-Cx100/step161186-hf/",
    "810m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-810M-Cx100/step269926-hf/",
    "1.4b": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-1.4B-Cx100/step308433-hf/",
}

midtraining_hf_checkpoints = {
    "275m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-275M-v2-lr1.6e-3/step38147-hf/",
    "810m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-v2-810M-lr4e-4/step23842-hf/",
    "1.4b": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-midtraining-v2-1.4b-lr4e-4/step11921-hf/",
}

long_context_hf_checkpoints = {
    "275m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-long-context-v2-275m/step47684-hf/",
    "810m": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-long-context-v2-810m/step23842-hf/",
    "1.4b": "/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-long-context-v2-1.4b/step23842-hf/",
}

all_stages = {
    "pretraining": pretraining_hf_checkpoints,
    "midtraining": midtraining_hf_checkpoints,
    "long_context": long_context_hf_checkpoints,
}

TEST_TASKS = ["olmobase:easy:qa:rc"]

DOWNSTREAM_TASKS = [
    # Math
    "olmobase:math",
    # Code
    "olmobase:easy:code:bpb",
    # MC STEM
    "olmobase:mcqa_stem",
    # MC Non-STEM
    "olmobase:mcqa_non_stem",
    # GenQA
    "olmobase:gen",
    # LBPP, BBH, MMLU Pro, DM Math — not yet in olmo-eval-internal
]

LC_TASKS = [
    "ruler_all__4096",
    "ruler_all__8192",
    "ruler_all__16384",
    "ruler_all__32768",
    "ruler_all__65536",
    "ruler_all__131072",
]

SAFETY_TASKS: list[str] = [
    # TODO(yashasbls): ask maliam
]


def build_command(model_path: str, tasks: list[str], num_gpus: int = NUM_GPUS) -> list[str]:
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
    cmd += ["-o", 'provider.kwargs.compilation_config={"custom_ops":["-rms_norm"]}']
    cmd += ["-o", "provider.add_bos_token=false"]
    cmd += ["-o", "provider.prompt_logprobs=1"]
    cmd += ["-o", "provider.logprob_temperature=1.0"]
    cmd += ["-o", "provider.completion_use_prompt_token_ids=true"]
    cmd += ["-o", "provider.completion_client_side_stop_trim=true"]
    cmd += ["-o", "provider.completion_sentencepiece_cleanup=true"]
    cmd += ["-o", "provider.dependencies=[transformers @ git+https://github.com/yashassamaga/transformers.git@olmo-3.5-hybrid]"]
    cmd += ["-m", model_path]
    for task in tasks:
        cmd += ["-t", task]
    cmd += ["--gpus", str(num_gpus)]
    cmd += ["--priority", PRIORITY]
    cmd += ["--group", GROUP]
    cmd += ["--cluster", CLUSTER]
    cmd += ["--workspace", WORKSPACE]
    cmd += ["--budget", BUDGET]
    # cmd += ["--store"]
    cmd += ["--inspect"]
    # cmd += ["--gcp-credentials"]
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
        choices=["test", "downstream", "lc", "both", "safety"],
        default="test",
    )
    parser.add_argument("--gpus", type=int, default=NUM_GPUS, help="Number of GPUs per job.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    num_gpus = args.gpus

    if args.eval_type == "test":
        tasks = TEST_TASKS
    elif args.eval_type == "downstream":
        tasks = DOWNSTREAM_TASKS
    elif args.eval_type == "lc":
        tasks = LC_TASKS
    elif args.eval_type == "safety":
        tasks = SAFETY_TASKS
    else:  # both
        tasks = DOWNSTREAM_TASKS + LC_TASKS

    for stage in args.stages:
        checkpoints = all_stages[stage]
        for size in args.sizes:
            model_path = checkpoints[size]
            for task in tasks:
                cmd = build_command(model_path, [task], num_gpus)
                print(f"\n=== {stage}/{size} | {task} ===")
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()