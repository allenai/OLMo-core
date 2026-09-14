"""Local prompt, reasoning-boundary and code-extraction regression checks."""

import json
import subprocess
import sys
from pathlib import Path

import olmoe3_hero_sft_tasks as tasks
import pytest
from olmo_eval.common.types import Instance, LMOutput, RequestType, Response
from olmo_eval.evals.tasks.common import get_task


@pytest.mark.parametrize("bundle", tasks.TASKS)
def test_chat_settings(bundle):
    task = get_task(tasks.TASKS[bundle][0])
    instance = Instance(question="Test question", gold_answer="42", metadata={"id": 1})
    request = task.format_request(instance)
    assert request.request_type == RequestType.CHAT
    assert task.config.num_fewshot == 0
    assert task.config.sampling_params == tasks.SAMPLING
    assert task.config.sampling_params.max_tokens == 32768


def test_reasoning_and_code(tmp_path, monkeypatch):
    monkeypatch.setenv("HERO_SFT_RESPONSE_AUDIT", str(tmp_path))
    task = get_task("hero_sft_humaneval")
    instance = Instance(
        question="def f():",
        gold_answer=None,
        metadata={"id": "HumanEval/0", "answer_prefix": "def f():\n"},
    )
    code = "def f():\n    return 42"
    raw = "Let me think.\n</think>\n```python\n" + code + "\n```"
    response = Response(
        instance=instance, request=task.format_request(instance), outputs=[LMOutput(raw)]
    )
    task._extract_answers([response])
    assert response.outputs[0].extracted_answer.strip() == code
    assert "Let me think" not in response.outputs[0].text
    records = list(tmp_path.glob("**/*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["raw_response"] == raw


def test_unfinished_reasoning_and_answer_tags():
    assert tasks.split_reasoning("still thinking") == ("still thinking", "", False)
    assert tasks.split_reasoning("<think>reason</think> <answer>42</answer>") == (
        "reason",
        "42",
        True,
    )


def test_ifbench_uses_final_only():
    task = get_task("hero_sft_ifbench")
    instance = Instance(question="Say YES", gold_answer=None, metadata={"id": 0})
    response = Response(
        instance=instance,
        request=task.format_request(instance),
        outputs=[LMOutput("hidden reasoning</think>YES")],
    )
    task._extract_answers([response])
    assert response.outputs[0].text == "YES"
    assert response.outputs[0].extracted_answer == "YES"


def test_native_cli_with_registered_task(tmp_path):
    recipe = tmp_path / "recipe.json"
    command = [
        "olmo-eval",
        "run",
        "-H",
        "default",
        "-o",
        "provider.kind=mock",
        "-m",
        "mock",
        "-O",
        str(tmp_path),
        "-t",
        "hero_sft_alpaca",
        "-o",
        "limit=1",
    ]
    recipe.write_text(json.dumps({"command": command, "output": str(tmp_path)}))
    script = Path(__file__).with_name("olmoe3_hero_sft_eval.py")
    result = subprocess.run(
        [sys.executable, str(script), "--execute", str(recipe)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-4000:]
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["tasks"][0]["num_instances"] == 1
    assert len(list((tmp_path / "responses/hero_sft_alpaca").glob("*.json"))) == 1
