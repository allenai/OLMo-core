"""CPU checks for bounded RULER fan-out, benchmark integrity, and source gates."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/olmo_ddp"))
import olmoe3_hero_ruler as ruler
import olmoe3_hero_ruler_control as control


def metrics_fixture():
    expected = json.loads(ruler.RECIPE.read_text())["task_definitions"]
    metrics = {
        "tasks": list(copy.deepcopy(expected).values()),
        "errors": [],
        "summary": {name: {"score": 0.5} for name in ruler.SELECTORS},
    }
    return expected, metrics


def test_exact_targets_and_standard_suite():
    assert len(control.TARGETS) == 4
    assert len({ruler.model_path(m, a) for m, a in control.TARGETS}) == 4
    assert all("plus" not in name for name in ruler.SELECTORS)
    assert len(ruler.SELECTORS) == 6
    with pytest.raises(ValueError):
        ruler.model_path("300b", "emo")
    with pytest.raises(ValueError):
        ruler.model_path("1267b", "dense")


def test_complete_metrics_and_only_null_schema_normalization():
    expected, metrics = metrics_fixture()
    for row in metrics["tasks"]:
        row["config"]["sampling_params"].update(truncate_prompt_tokens=None, truncation_side=None)
    ruler.check_metrics(metrics, expected)
    metrics["tasks"][0]["config"]["sampling_params"]["max_tokens"] += 1
    with pytest.raises(AssertionError):
        ruler.check_metrics(metrics, expected)


@pytest.mark.parametrize("bad", ("missing", "duplicate", "errors", "instances", "nan", "plus"))
def test_incomplete_or_wrong_evals_rejected(bad):
    expected, metrics = metrics_fixture()
    if bad == "missing":
        metrics["tasks"].pop()
    elif bad == "duplicate":
        metrics["tasks"][0] = metrics["tasks"][1]
    elif bad == "errors":
        metrics["errors"] = ["inference failed"]
    elif bad == "instances":
        metrics["tasks"][0]["num_instances"] = 1
    elif bad == "nan":
        metrics["summary"][ruler.SELECTORS[0]]["score"] = float("nan")
    else:
        metrics["summary"] = {
            name.replace("ruler_", "ruler_plus_"): v for name, v in metrics["summary"].items()
        }
    with pytest.raises((AssertionError, KeyError)):
        ruler.check_metrics(metrics, expected)


def test_long_context_provider_changes_only_serving_limits():
    provider = [
        "provider.dtype=bfloat16",
        "provider.max_model_len=8192",
        "provider.kwargs.max_num_seqs=32",
        "provider.kwargs.mamba_ssm_cache_dtype=float32",
        "provider.kwargs.max_num_batched_tokens=4096",
    ]
    result = ruler.configure_provider(lambda fast, count: provider, 4)
    assert "provider.max_model_len=131072" in result
    assert "provider.kwargs.max_num_seqs=8" in result
    assert "provider.add_bos_token=false" in result
    assert "provider.kwargs.mamba_ssm_cache_dtype=float32" in result
    assert "provider.kwargs.max_num_batched_tokens=4096" in result
    assert "provider.dtype=bfloat16" in result


def test_qualification_requires_matching_source_hash(tmp_path):
    model = tmp_path / "hf"
    model.mkdir()
    assert control.qualified(model) is False
    marker = model / "_HERO_CONVERSION_SUCCESS.json"
    marker.write_text("{}")
    digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    (tmp_path / "conversion-success.json").write_text(json.dumps(dict(passed=True)))
    for name in ("vllm-parity-success.json", "eval-smoke-success.json"):
        (tmp_path / name).write_text(
            json.dumps(dict(passed=True, precise=True, conversion_sha256=digest))
        )
    assert control.qualified(model)
    marker.write_text('{"changed":true}')
    with pytest.raises(AssertionError):
        control.qualified(model)


def test_worker_spec_bounded_and_no_cleanup():
    template = {
        "tasks": [
            {
                "arguments": [
                    f"fetch {control.HELPER_REF}\ncheckout {control.HELPER_REF}\n"
                    "python ladders/olmoe3/workloads/hero_full_eval.py gen_mc "
                    "/weka/olmo-3p5-checkpoints/scratch/hero-hf-20260909/emo/step75500/hf --instances 4 --fast-pilot"
                ],
                "envVars": [],
                "resources": {"gpuCount": 4},
                "constraints": {"cluster": ["ai2/jupiter", "ai2/ceres"]},
            }
        ]
    }
    for milestone, arm in control.TARGETS:
        spec = control.worker_spec(template, milestone, arm, "a" * 40)
        t = spec["tasks"][0]
        assert t["context"]["minRuntime"] == "1h"
        assert t["context"]["priority"] == "urgent"
        assert t["resources"]["gpuCount"] == 4
        assert spec["retry"]["allowedTaskRetries"] == 0
        assert f"--milestone {milestone} --arm {arm}" in t["arguments"][0]
        assert "cleanup" not in t["arguments"][0]
