"""Replay immutable completed generations; leave missing requests and scoring unchanged."""

import copy
import hashlib
import json
import os
from pathlib import Path


def install_resume_cache():
    """Install in the evaluator and spawned workers only for an approved resume recipe."""
    path = os.environ.get("HERO_SFT_RESUME_RECIPE")
    if not path:
        return
    from olmo_eval.common.types import LMOutput
    from olmo_eval.runners.asynq import processing
    from olmo_eval.runners.asynq.types import ResultItem
    from olmoe3_hero_sft_tasks import SAMPLING

    if getattr(processing.process_items, "hero_resume", False):
        return
    recipe = json.loads(Path(path).read_text())
    assert len(recipe["tasks"]) == 1
    task_name = next(iter(recipe["tasks"]))
    assert task_name in (
        "hero_sft_math500",
        "hero_sft_ifbench",
        "hero_sft_humaneval",
        "hero_sft_alpaca",
    )
    cache = {}
    for file in (Path(recipe["output"]) / "responses" / task_name).glob("*.json"):
        row = json.loads(file.read_text())
        key = str(row["native_id"])
        assert file.name == hashlib.sha256(key.encode()).hexdigest() + "-0.json"
        assert key not in cache and isinstance(row["raw_response"], str)
        cache[key] = row
    original = processing.process_items

    async def process_items(items, harness, result_queue, *args, **kwargs):
        missing = []
        replayed = 0
        for item in items:
            assert item.model_name == recipe["model"]
            assert item.sampling_params == SAMPLING
            key = str(item.instance.metadata["id"])
            saved = cache.get(key)
            if saved is None:
                missing.append(item)
                continue
            assert saved["question"] == item.instance.question
            assert saved["reference"] == item.instance.metadata.get("reference")
            prepared = harness._apply_config(item.request)
            result_queue.put(
                ResultItem(
                    model_name=item.model_name,
                    task_id=item.task_id,
                    instance_idx=item.instance_idx,
                    instance=item.instance,
                    request=prepared,
                    request_trace=harness.provider.describe_request(prepared, item.sampling_params),
                    outputs=[
                        LMOutput(
                            text=saved["raw_response"],
                            metadata=copy.deepcopy(saved["output_metadata"]),
                        )
                    ],
                    error=None,
                    attempt=item.attempt,
                )
            )
            replayed += 1
        print(
            "SFT_RESPONSE_CACHE_BATCH",
            json.dumps({"replayed": replayed, "generate": len(missing)}),
            flush=True,
        )
        if missing:
            await original(missing, harness, result_queue, *args, **kwargs)

    process_items.hero_resume = True
    processing.process_items = process_items
    print(
        "SFT_RESPONSE_CACHE_READY", json.dumps({"saved": len(cache), "task": task_name}), flush=True
    )
