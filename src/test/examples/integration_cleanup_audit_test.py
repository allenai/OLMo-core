"""The smoke audit must not confuse authorized cleanup with a lost checkpoint."""

import copy
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "examples/olmo_ddp/olmoe3_integration_collect.py"
_SPEC = importlib.util.spec_from_file_location("integration_collect", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def evidence():
    records = [
        {
            "step": step,
            "source_complete": True,
            "remote_verified": True,
            "receipt_sha256": "verified-receipt",
            "local_deleted": step == 0,
            "local_present": step != 0,
            "retention_protected": step != 0,
            "successor_checkpoint_step": 4 if step == 0 else None,
            "deletion_last_error": None,
        }
        for step in (0, 4, 8)
    ]
    return {
        "auto_delete_enabled": True,
        "deletion_policy": {"min_local_checkpoints": 2, "grace_seconds": 3600},
        "checkpoints": records,
    }


def test_complete_local_source():
    assert _MODULE.checkpoint_accounted_for({"source_complete": True}, {}, False)


def test_verified_cleanup_requires_explicit_opt_in():
    item = {"step": 0, "source_complete": False, "local_deleted": True}
    assert _MODULE.checkpoint_accounted_for(item, evidence(), True)
    assert not _MODULE.checkpoint_accounted_for(item, evidence(), False)
    assert not _MODULE.checkpoint_accounted_for(item, {}, True)


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_complete", False),
        ("remote_verified", False),
        ("receipt_sha256", None),
        ("local_deleted", False),
        ("local_present", True),
        ("retention_protected", True),
        ("successor_checkpoint_step", 0),
        ("deletion_last_error", "verification failed"),
    ],
)
def test_incomplete_or_unsafe_deleted_record(key, value):
    manifest = evidence()
    manifest["checkpoints"][0][key] = value
    item = {"step": 0, "source_complete": False, "local_deleted": True}
    assert not _MODULE.checkpoint_accounted_for(item, manifest, True)


def test_newest_pair_and_verified_successor_required():
    item = {"step": 0, "source_complete": False, "local_deleted": True}
    for key in ("local_present", "remote_verified", "retention_protected"):
        manifest = copy.deepcopy(evidence())
        manifest["checkpoints"][1][key] = False
        assert not _MODULE.checkpoint_accounted_for(item, manifest, True)
    manifest = evidence()
    manifest["deletion_policy"]["grace_seconds"] = 0
    assert not _MODULE.checkpoint_accounted_for(item, manifest, True)
