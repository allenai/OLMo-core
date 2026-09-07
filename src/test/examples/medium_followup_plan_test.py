"""CPU-only invariants for diagnostic samples and the approved medium CBS fork."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "examples/olmo_ddp"))

from olmoe3_medium_cbs_control import replace_env, training_spec
from olmoe3_medium_cbs_plan import (
    BASELINE,
    BRANCH,
    FORK_TOKENS,
    TARGET_TOKENS,
    parent_retention_for_save,
    validate,
)
from olmoe3_medium_followup_plan import (
    VARIANTS,
    microbatch_sequence_sizes,
    parse_test,
    sample_offsets,
)


@pytest.mark.parametrize("size", [0, 1, 2, 128, 2048, 2049, 2**24, 2**24 + 1, 37_748_736, 10**12])
def test_exact_sample_bounds(size):
    offsets = sample_offsets(size)
    assert len(offsets) == min(size, 2048)
    assert len(set(offsets)) == len(offsets)
    assert offsets == sorted(offsets)
    assert all(0 <= i < size for i in offsets)
    if size:
        assert offsets[0] == 0 and offsets[-1] == size - 1


@pytest.mark.parametrize("variant", VARIANTS)
def test_geometry_names(variant):
    assert parse_test(f"{variant}-mb4-b16mi") == (variant, 4, 16_777_216)
    assert parse_test(variant) == (variant, None, None)
    with pytest.raises(ValueError):
        parse_test(f"{variant}-mb8-b16mi")


def test_shared_token_horizon_and_fork_protection():
    validate()
    assert TARGET_TOKENS == 100_663_296_000
    assert FORK_TOKENS == 67_108_864_000
    assert BASELINE.tokens_at(4000) == BRANCH.tokens_at(4000)
    assert BASELINE.tokens_at(6000) == BRANCH.tokens_at(5000) == TARGET_TOKENS
    assert BRANCH.tokens_at(4250) == BASELINE.tokens_at(4500)
    assert math.isclose(BRANCH.lr, BASELINE.lr * math.sqrt(2), rel_tol=1e-15)
    assert 4000 in list(range(0, 6001, BASELINE.interval))[-BASELINE.keep :]


@pytest.mark.parametrize("mb", [2, 3, 4])
def test_spec_preserves_secret_references_and_requires_128_gpus(mb):
    template = {
        "version": "v2",
        "tasks": [
            {
                "name": "old",
                "resources": {"gpuCount": 8},
                "envVars": [
                    {"name": "HF_TOKEN", "secret": "private-reference"},
                    {"name": "OLMOE3_MEDIUM_DIAGNOSTIC", "value": "1"},
                ],
            }
        ],
    }
    spec = training_spec(template, BRANCH, commit="a" * 40, variant="optimized", mb=mb)
    task = spec["tasks"][0]
    env = {v["name"]: v for v in task["envVars"]}
    assert env["HF_TOKEN"] == {"name": "HF_TOKEN", "secret": "private-reference"}
    assert env["OLMOE3_MEDIUM_DIAGNOSTIC"]["value"] == "0"
    assert env["OLMOE3_MEDIUM_BATCH"]["value"] == str(BRANCH.batch)
    assert env["OLMOE3_MEDIUM_MB"]["value"] == str(mb)
    assert sum(microbatch_sequence_sizes(BRANCH.batch, 128, mb)) * 128 * 8192 == BRANCH.batch
    assert task["replicas"] * task["resources"]["gpuCount"] == 128
    assert task["context"]["priority"] == "urgent"
    assert task["context"]["minRuntime"] == "1h"
    assert task["result"]["path"] == "/noop-results"
    assert template["tasks"][0]["name"] == "old"
    replace_env(task, {"HF_TOKEN": None})
    assert not any(v["name"] == "HF_TOKEN" for v in task["envVars"])


def test_off_cadence_saves_cannot_evict_fork():
    regular = list(range(0, 6000, 500))
    assert parent_retention_for_save(regular, 6000) == 5
    interrupted = regular + [4103, 4721, 5832]
    keep = parent_retention_for_save(interrupted, 6000)
    assert keep == 8
    assert 4000 in sorted(set(interrupted + [6000]))[-keep:]


def test_balanced_medium_geometry_preserves_exact_batches():
    assert microbatch_sequence_sizes(16_777_216, 128, 3) == [3, 3, 3, 3, 2, 2]
    assert microbatch_sequence_sizes(33_554_432, 128, 3) == [3, 3, 3, 3, 2, 2] * 2
    for gpus in (64, 128):
        for batch in (16_777_216, 33_554_432):
            sizes = microbatch_sequence_sizes(batch, gpus, 3)
            assert sum(sizes) * gpus * 8192 == batch
            assert min(sizes) >= 2 and max(sizes) <= 3
    assert parse_test("optimized-mb3-b16mi") == ("optimized", 3, 16_777_216)
    with pytest.raises(ValueError, match="fallback"):
        microbatch_sequence_sizes(8192 * 128, 128, 3)
