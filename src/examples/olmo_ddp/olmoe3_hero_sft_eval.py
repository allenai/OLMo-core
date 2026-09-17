"""Run the four requested post-training benchmarks on a verified final SFT export."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from olmoe3_hero_sft_resume import install_resume_cache
from olmoe3_hero_sft_tasks import TASKS

# Multiprocessing spawn imports this module before entering the inference worker.
install_resume_cache()


def code_preflight():
    """Verify correct and incorrect code in a credential-free, mount-free Modal sandbox."""
    from dataclasses import replace

    from olmo_eval.harness import get_harness_preset
    from olmo_eval.harness.sandbox.config import SandboxMode
    from olmo_eval.harness.sandbox.executor import SandboxExecutor

    async def run():
        base = get_harness_preset("codex_python").sandboxes[0]
        config = replace(
            base, mode=SandboxMode.MODAL, instances=1, min_instances=1, startup_timeout=300.0
        )
        assert not config.environment and not config.volumes and not config.required_secrets
        async with SandboxExecutor(config, name="hero-sft-humaneval-preflight") as executor:
            ok = await executor.execute_command(
                "python3 -c 'assert 2+2 == 4; print(\"SANDBOX_OK\")'", timeout=30
            )
            assert ok.success
            assert "SANDBOX_OK" in str(ok)
            bad = await executor.execute_command("python3 -c 'assert 2+2 == 5'", timeout=30)
            assert not bad.success
        print("SFT_CODE_SANDBOX_PREFLIGHT_PASSED", flush=True)

    asyncio.run(run())


def validate_export(model):
    """Check metadata, strict conversion and independent HF/vLLM qualification receipts."""
    from olmoe3_hero_sft_metadata import GENERATION, check_tokenizer
    from olmoe3_hero_sft_plan import DATA
    from transformers import AutoTokenizer, GenerationConfig

    root = model.parent
    receipt_path = model / "_HERO_CONVERSION_SUCCESS.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["passed"] and root.name == f"step{receipt['step']}"
    for file in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "sft-metadata-audit.json",
    ):
        assert (
            hashlib.sha256((model / file).read_bytes()).hexdigest()
            == receipt["output_sha256"][file]
        )
    check_tokenizer(
        AutoTokenizer.from_pretrained(model),
        AutoTokenizer.from_pretrained(DATA / "train/tokenizer"),
    )
    generation = GenerationConfig.from_pretrained(model)
    assert all(getattr(generation, key) == value for key, value in GENERATION.items())
    from olmoe3_hero_4t_eval_policy import validate_export

    validate_export(model, hash_weights=True)
    return receipt


def judge_alpaca(output):
    """Use the official AlpacaEval GPT-4.1 weighted judge and length correction."""
    import alpaca_eval

    rows = [
        json.loads(p.read_text()) for p in (output / "responses/hero_sft_alpaca").glob("*.json")
    ]
    rows.sort(key=lambda x: x["native_id"])
    assert len(rows) == 805 and len({r["native_id"] for r in rows}) == 805
    references = [r["reference"] for r in rows]
    models = [
        {
            "instruction": r["question"],
            "output": r["final_response"],
            "generator": "HERO_SFT",
            "dataset": r["reference"]["dataset"],
        }
        for r in rows
    ]
    assert all(a["instruction"] == b["instruction"] for a, b in zip(models, references))
    (output / "alpaca-model-outputs.json").write_text(json.dumps(models, indent=2) + "\n")
    (output / "alpaca-reference-outputs.json").write_text(json.dumps(references, indent=2) + "\n")
    leaderboard, annotations = alpaca_eval.evaluate(
        model_outputs=models,
        reference_outputs=references,
        annotators_config="weighted_alpaca_eval_gpt4.1",
        output_path=str(output / "alpaca-judge"),
        is_return_instead_of_print=True,
        precomputed_leaderboard=None,
        is_cache_leaderboard=False,
        fn_metric="get_length_controlled_winrate",
        sort_by="length_controlled_winrate",
        metric_kwargs={"save_weights_dir": str(output / "alpaca-judge/glm-weights")},
        caching_path=str(output / "alpaca-judge-cache.json"),
    )
    score = leaderboard.loc["HERO_SFT"].to_dict()
    assert len(annotations) == 805
    assert all(a.get("preference") is not None for a in annotations)
    assert all(math.isfinite(float(score[k])) for k in ("win_rate", "length_controlled_winrate"))
    result = {
        "judge": "weighted_alpaca_eval_gpt4.1",
        "scores_percent": score,
        "instances": 805,
        "reference": "GPT-4 baseline (AlpacaEval 2 prompts)",
        "note": "GPT-4.1 judge (Olmo recipe v3), not the GPT-4-Turbo judge of AE2.",
    }
    (output / "alpaca-results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def execute(receipt_path):
    """Load custom task adapters before the unchanged frozen evaluator CLI."""
    from olmo_eval.cli import main as cli_main

    receipt = json.loads(Path(receipt_path).read_text())
    os.environ["HERO_SFT_RESPONSE_AUDIT"] = str(Path(receipt["output"]) / "responses")
    sys.argv = receipt["command"]
    cli_main()


def main():
    from olmo_eval.evals.tasks.common import get_task
    from olmoe3_hero_sft_convert import export_root
    from olmoe3_hero_sft_plan import CAMPAIGN, MOUNT, find_run

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--bundle", choices=[*TASKS, "smoke"], required=True)
    args = parser.parse_args()
    assert MOUNT.is_mount()
    run = find_run(args.run)
    assert not run.smoke
    model = export_root(run) / run.arm / f"step{run.total_steps}/hf"
    conversion = validate_export(model)
    output = model.parent / "posttrain-evals-r1" / args.bundle
    # Math/IFBench/Alpaca can replay immutable completed responses after preemption.
    # Code requires fresh sandbox/handler state, so each attempt has its own folder.
    base_output = output
    if args.bundle in ("humaneval", "smoke"):
        assert os.environ.get("BEAKER_JOB_ID"), "A fresh Beaker attempt ID is required"
        output = output / "attempts" / os.environ["BEAKER_JOB_ID"]
    resume = args.bundle in ("math500", "ifbench", "alpaca") and (output / "recipe.json").is_file()
    if resume:
        assert args.bundle in ("math500", "ifbench", "alpaca")
        assert (output / "recipe.json").is_file(), "Resume needs the original recipe"
    else:
        output.mkdir(parents=True, exist_ok=True)
    # Retain the lock until this main invocation and its evaluator subprocess finish.
    output_lock = (output / "RESUME.lock").open("a")
    fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bundles = list(TASKS) if args.bundle == "smoke" else [args.bundle]
    if "humaneval" in bundles:
        code_preflight()
    if "alpaca" in bundles:
        assert os.environ.get(
            "OPENAI_API_KEY"
        ), "Alpaca judge credential required before generation"
        import alpaca_eval

        assert alpaca_eval is not None
    os.environ["OLMO_VLLM_TORCH_GROUPED_MOE"] = "1"
    os.environ["OLMO_VLLM_FLA_KDA"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    for forbidden in (
        "OLMO_HERO_PRECISE_INFERENCE",
        "OLMO_HF_MOE_CORE_REFERENCE",
        "OLMO_HF_MOE_REFERENCE_LOOP",
    ):
        assert not os.environ.get(forbidden), forbidden
    cmd = [
        "olmo-eval",
        "run",
        "--inspect",
        "--save-predictions",
        "--save-requests",
        "-H",
        "codex_python" if "humaneval" in bundles else "default",
    ]
    options = [
        "provider.kind=vllm",
        "provider.dtype=bfloat16",
        "provider.num_instances=1",
        "provider.max_model_len=65536",
        "provider.kwargs.enforce_eager=true",
        "provider.kwargs.mamba_ssm_cache_dtype=float32",
        "provider.kwargs.attention_backend=FLASH_ATTN",
        "provider.kwargs.enable_flashinfer_autotune=false",
        "provider.kwargs.language_model_only=true",
        "provider.kwargs.gpu_memory_utilization=0.75",
        "provider.kwargs.max_num_batched_tokens=4096",
        (
            "provider.kwargs.max_num_seqs=16"
            if args.bundle == "smoke"
            else "provider.kwargs.max_num_seqs=32"
        ),
        "provider.kwargs.disable_log_stats=false",
        "provider.kwargs.enable_prefix_caching=false",
        "provider.kwargs.model_impl=vllm",
        "provider.kwargs.tensor_parallel_size=1",
        "provider.kwargs.seed=1234",
        "provider.kwargs.generation_config=auto",
    ]
    if "humaneval" in bundles:
        options += [
            'sandboxes={"mode":"modal","instances":4,"min_instances":1,"max_concurrency":1}',
            "scoring_concurrency=4",
        ]
    for item in options:
        cmd += ["-o", item]
    cmd += [
        "-m",
        str(model),
        "-O",
        str(output),
        "--experiment-group",
        CAMPAIGN,
        "--experiment-name",
        run.run_id + "-" + args.bundle,
    ]
    definitions = {}
    for bundle in bundles:
        name, count = TASKS[bundle]
        task = get_task(name)
        instances = list(task.instances)
        assert len(instances) == count, (name, len(instances))
        definitions[name] = task.config.to_dict()
        cmd += ["-t", name]
        if args.bundle == "smoke":
            cmd += ["-o", "limit=3"]
    receipt = {
        "command": cmd,
        "output": str(output),
        "model": str(model),
        "tasks": definitions,
        "conversion_sha256": hashlib.sha256(
            (model / "_HERO_CONVERSION_SUCCESS.json").read_bytes()
        ).hexdigest(),
        "conversion_profile": conversion["inference_profile"],
        "inference_profile": "bf16-grouped-fla-pilot-v1",
        "epoch": run.epochs,
        "recipe_reference": "oe-eval-internal@e8c7fedaee8067421bf651419b2648117262a72a",
        "note": "Think chat adapters; no PT few-shot prompts; unfinished reasoning scores as empty final answer.",
    }
    recipe = output / "recipe.json"
    if resume:
        assert json.loads(recipe.read_text()) == json.loads(
            json.dumps(receipt)
        ), "Refuse model/recipe drift on resume"
        if (output / "success.json").is_file():
            proof = json.loads((output / "success.json").read_text())
            assert (
                proof["passed"] and proof["model"] == str(model) and proof["bundle"] == args.bundle
            )
            assert proof["recipe_sha256"] == hashlib.sha256(recipe.read_bytes()).hexdigest()
            assert (
                proof["metrics_sha256"]
                == hashlib.sha256((output / "metrics.json").read_bytes()).hexdigest()
            )
            print("SFT_POSTTRAIN_ALREADY_COMPLETE", flush=True)
            return
        os.environ["HERO_SFT_RESUME_RECIPE"] = str(recipe)
    else:
        recipe.write_text(json.dumps(receipt, indent=2) + "\n")
    print("SFT_POSTTRAIN_EVAL_START", json.dumps(cmd), flush=True)
    subprocess.run([sys.executable, __file__, "--execute", str(recipe)], check=True)
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    actual = {r["task"]: r for r in metrics["tasks"]}
    assert not metrics.get("errors") and set(actual) == set(definitions)
    counts = {}
    for bundle in bundles:
        name, total = TASKS[bundle]
        expected = 3 if args.bundle == "smoke" else total
        row = actual[name]
        assert (
            row["num_instances"] == expected
            and not row.get("error")
            and not row.get("instances_failed", 0)
        )
        rows = [json.loads(p.read_text()) for p in (output / "responses" / name).glob("*.json")]
        assert len(rows) == expected and len({r["native_id"] for r in rows}) == expected
        counts[name] = {
            "instances": expected,
            "reasoning_closed": sum(r["reasoning_closed"] for r in rows),
        }
    judge = judge_alpaca(output) if args.bundle == "alpaca" else None
    proof = {
        "passed": True,
        "model": str(model),
        "bundle": args.bundle,
        "counts": counts,
        "metrics": str(metrics_path),
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
        "alpaca": judge,
    }
    (output / "success.json").write_text(json.dumps(proof, indent=2, default=str) + "\n")
    if output != base_output:
        from olmoe3_lr_sweep_watch import atomic_json

        atomic_json(base_output / "success.json", {**proof, "attempt_output": str(output)})
    print("SFT_POSTTRAIN_EVAL_SUCCESS", json.dumps(proof, default=str), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--execute":
        execute(sys.argv[2])
    else:
        main()
