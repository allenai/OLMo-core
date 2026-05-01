import argparse
import subprocess

COOKBOOK_EVAL = "/weka/oe-training-default/yashasbls/olmo-cookbook/.venv/bin/olmo-cookbook-eval"
DASHBOARD = "yashasbls-hybrid-small-tests"
CLUSTER = "ai2/jupiter"
PRIORITY = "urgent"
NUM_GPUS = 1
WORKSPACE = "ai2/linear-rnns"

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

TEST_TASKS = ["olmo3:dev:1b:main:v2"]

DOWNSTREAM_TASKS = [
    "olmo3:base_easy",
    "olmo3:dev:7b:main:v2:fast",
]

LC_TASKS = [
    "ruler:8k",
    "ruler:16k",
    "ruler:32k",
    "ruler:64k",
    "ruler:128k",
    "helmet:8k",
    "helmet:16k",
    "helmet:32k",
    "helmet:64k",
    "helmet:128k",
]

SAFETY_TASKS: list[str] = [
    # TODO(yashasbls): ask maliam
]

GANTRY_INSTALL = (
    "export PATH=/stage/.venv/bin:$PATH"
    " && uv pip list"
    " && uv pip install lm-eval==0.4.9.2"
    " && uv pip install git+https://github.com/yashassamaga/transformers.git@olmo-3.5-hybrid"
    " && uv pip install peft==0.18.1"
    " && uv pip list"
)

def build_command(model_path: str, tasks: list[str]) -> list[str]:
    return [
        COOKBOOK_EVAL, "evaluate", model_path,
        "--tasks", *tasks,
        "--priority", PRIORITY,
        "--cluster", CLUSTER,
        "--num-gpus", str(NUM_GPUS),
        "--model-backend", "vllm",
        "--partition-size", str(len(tasks)),
        "--huggingface-secret", "YASHASBLS_HF_TOKEN",
        "--use-hf-token",
        "--dashboard", DASHBOARD,
        "--workspace", WORKSPACE,
        "--vllm-use-v1-spec",
        "--vllm-memory-utilization=0.7",
        "--model-args", "trust_remote_code=true,max_length=4096",
        "--gantry-args", (
            "retries=0,"
            # f'install="{GANTRY_INSTALL}"'
        ),
        "--use-gantry",
        "--oe-eval-branch", "vllm-11-2-fix+better-time-tracking+eager-cascade",
        "--beaker-image", "tylerr/oe_eval_auto-e5670e07b",
        "--fim-tokens", "l2c",
    ]


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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
            cmd = build_command(model_path, tasks)
            print(f"\n=== {stage}/{size} ===")
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()