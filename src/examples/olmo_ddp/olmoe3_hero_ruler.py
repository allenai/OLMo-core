"""Standard RULER on exactly the approved hero exports; no model/kernel changes."""

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from olmoe3_hero_decay_plan import EVAL_ROOT, HELPER_REF, MOUNT
from olmoe3_lr_sweep_watch import atomic_json, log

CAMPAIGN = "olmo35-small-ruler-20260911"
LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072)
SELECTORS = tuple(f"ruler_all__{length}" for length in LENGTHS)
RECIPE = Path(__file__).with_name("hero_ruler_baseline_recipe.json")
AUTOMATION = MOUNT / "uploader/automation" / CAMPAIGN


def model_path(milestone, arm):
    """Resolve only the two existing PT exports and the two approved decay endpoints."""
    if arm not in ("emo", "non-emo"):
        raise ValueError(arm)
    if milestone == "1267b":
        return MOUNT / "scratch/hero-hf-20260909" / arm / "step75500/hf"
    if milestone == "decay2t":
        return EVAL_ROOT / arm / "step120000/hf"
    raise ValueError(milestone)


def normalized(config):
    """Allow only the two newly serialized null sampling fields, not recipe changes."""
    result = copy.deepcopy(config)
    for key in ("truncate_prompt_tokens", "truncation_side"):
        result["sampling_params"].setdefault(key, None)
    return json.loads(json.dumps(result))


def definitions(bundle="ruler"):
    """Check all 78 task definitions against the dense PT metrics, before GPU loading."""
    from olmo_eval.evals.suites import get_suite
    from olmo_eval.evals.tasks.common import get_task

    assert bundle == "ruler"
    recipe = json.loads(RECIPE.read_text())
    expected = copy.deepcopy(recipe["task_definitions"])
    expanded = {name for selector in SELECTORS for name in get_suite(selector).expand()}
    assert len(expected) == 78 and expanded == set(expected)
    for name, row in expected.items():
        current = json.loads(json.dumps(get_task(name).config.to_dict()))
        assert normalized(current) == normalized(row["config"]), f"Recipe drift: {name}"
        assert row["num_instances"] == 100 and current["limit"] == 100
        # The frozen worker checks raw serialized output configs. Null additions are harmless.
        row["config"] = current
    recipe["arguments"] = list(SELECTORS)
    log("RULER_DENSE_RECIPE_MATCH", tasks=len(expected), lengths=LENGTHS)
    return recipe, expected


def configure_provider(original, instances):
    """Use the established fast profile with long-context serving capacity, not new weights."""
    provider = original(True, instances)
    changes = {
        "provider.max_model_len": "131072",
        "provider.kwargs.max_num_seqs": "8",
    }
    result = [
        setting.split("=", 1)[0]
        + "="
        + changes.get(setting.split("=", 1)[0], setting.split("=", 1)[1])
        for setting in provider
    ]
    result.extend(["provider.add_bos_token=false", "provider.kwargs.enable_chunked_prefill=true"])
    return result


def check_metrics(metrics, expected, *, smoke=False):
    """Reject incomplete tasks, failures, non-finite scores, and changed benchmark recipes."""
    rows = metrics.get("tasks", [])
    actual = {r["task"]: r for r in rows}
    assert not metrics.get("errors") and len(rows) == len(expected)
    assert set(actual) == set(expected)
    for name, original in expected.items():
        row = actual[name]
        config = copy.deepcopy(original["config"])
        if smoke:
            config["limit"] = 1
        assert normalized(row["config"]) == normalized(config), name
        assert row["num_instances"] == (1 if smoke else 100), name
        assert not row.get("error") and not row.get("instances_failed", 0), name
        if "instances_processed" in row:
            assert row["instances_processed"] == row["num_instances"], name
    if not smoke:
        for selector in SELECTORS:
            value = metrics["summary"][selector]["score"]
            assert math.isfinite(value) and 0 <= value <= 1, selector


def success_path(model):
    """A separate pointer; never overwrite an OLMoBase result or conversion receipt."""
    return model.parent / f"{CAMPAIGN}-success.json"


def verify_success(model):
    """Recheck hashes and complete scores when the controller observes completion."""
    row = json.loads(success_path(model).read_text())
    assert row["passed"] and row["model"] == str(model) and row["campaign"] == CAMPAIGN
    assert (
        row["source_conversion_sha256"]
        == hashlib.sha256((model / "_HERO_CONVERSION_SUCCESS.json").read_bytes()).hexdigest()
    )
    assert row["metrics_sha256"] == hashlib.sha256(Path(row["metrics"]).read_bytes()).hexdigest()
    metrics = json.loads(Path(row["metrics"]).read_text())
    expected = json.loads(RECIPE.read_text())["task_definitions"]
    check_metrics(metrics, expected)
    assert metrics["config"]["provider"]["model"].rstrip("/") == str(model)
    return row


def main():
    """Operationally test 4K through 128K before executing the full six-length benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone", choices=("1267b", "decay2t"), required=True)
    parser.add_argument("--arm", choices=("emo", "non-emo"), required=True)
    parser.add_argument("--helper", type=Path, default=Path("/tmp/hero-ladder"))
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    recipe, expected = definitions()
    if args.check_only:
        return
    assert args.instances == 4
    assert MOUNT.is_mount(), "Refuse overlay scratch writes"
    assert (
        subprocess.check_output(
            ["git", "-C", str(args.helper), "rev-parse", "HEAD"], text=True
        ).strip()
        == HELPER_REF
    )
    assert not subprocess.check_output(
        ["git", "-C", str(args.helper), "status", "--porcelain"], text=True
    ).strip()
    sys.path.insert(0, str(args.helper / "ladders/olmoe3/workloads"))
    import hero_full_eval as runtime

    model = model_path(args.milestone, args.arm)
    assert model.resolve() == model and model.is_dir()
    runtime.FAST_MODELS = {
        model_path(m, a) for m in ("1267b", "decay2t") for a in ("emo", "non-emo")
    }
    conversion_sha = runtime.validate_source(model, True)
    if success_path(model).exists():
        verify_success(model)
        log("RULER_ALREADY_COMPLETE", model=str(model))
        return
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    provider = configure_provider(runtime.configure_inference, args.instances)
    log(
        "RULER_INFERENCE_PROFILE",
        model=str(model),
        provider=provider,
        profile=runtime.FAST_PROFILE,
        numerically_qualified=False,
        note="Operational long-context smoke is not a numerical parity qualification.",
    )
    attempt = os.environ.get("BEAKER_JOB_ID", str(time.time_ns())).lower()
    output = model.parent / f"pilot-olmobase-ruler-{CAMPAIGN}-{attempt}"
    smoke = model.parent / f"ruler-smoke-{CAMPAIGN}-{attempt}"
    smoke.mkdir(exist_ok=False)
    cmd = [
        "olmo-eval",
        "run",
        "--inspect",
        "--save-predictions",
        "--save-requests",
        "-H",
        "default",
    ]
    for setting in provider:
        cmd.extend(["-o", setting])
    cmd.extend(
        [
            "-m",
            str(model),
            "-O",
            str(smoke),
            "--experiment-group",
            CAMPAIGN,
            "--experiment-name",
            f"{args.milestone}-{args.arm}-ruler-long-smoke",
        ]
    )
    smoke_tasks = {
        f"ruler_niah_s_1__{length}": expected[f"ruler_niah_s_1__{length}"] for length in LENGTHS
    }
    for name in smoke_tasks:
        cmd.extend(["-t", name, "-o", "limit=1"])
    atomic_json(smoke / "smoke-recipe.json", {"command": cmd, "baseline_recipe": recipe})
    log("RULER_LONG_SMOKE_START", tasks=len(smoke_tasks))
    subprocess.run(cmd, check=True, timeout=1800)
    check_metrics(json.loads((smoke / "metrics.json").read_text()), smoke_tasks, smoke=True)
    atomic_json(
        smoke / "smoke-success.json",
        {
            "passed": True,
            "operational_only": True,
            "lengths": LENGTHS,
            "source_conversion_sha256": conversion_sha,
        },
    )
    log("RULER_LONG_SMOKE_PASSED", lengths=LENGTHS)
    # Reuse all full-suite validation and receipt logic; no inference source edits.
    runtime.BUNDLES["ruler"] = SELECTORS
    runtime.definitions = definitions
    original = runtime.configure_inference
    runtime.configure_inference = lambda fast, instances: configure_provider(original, instances)
    sys.argv = [
        runtime.__file__,
        "ruler",
        str(model),
        "--instances",
        str(args.instances),
        "--fast-pilot",
        "--output",
        str(output),
    ]
    runtime.main()
    metrics = json.loads((output / "metrics.json").read_text())
    check_metrics(metrics, expected)
    row = json.loads((output / "full-eval-pilot-success.json").read_text())
    row.update(
        campaign=CAMPAIGN,
        model=str(model),
        milestone=args.milestone,
        arm=args.arm,
        lengths=LENGTHS,
        smoke=str(smoke),
        recipe_sha256=hashlib.sha256(RECIPE.read_bytes()).hexdigest(),
    )
    atomic_json(success_path(model), row)
    verify_success(model)
    log("HERO_RULER_SUCCESS", **row)


if __name__ == "__main__":
    main()
