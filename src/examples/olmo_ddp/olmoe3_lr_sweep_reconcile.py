"""One-shot audited repair for an explicitly selected ambiguous sweep submission.

This is an operator action, not automatic retry policy. The live controller never
resubmits an ambiguous intent; publishing its completed receipt is atomic. No
training settings, run identities, retention policies or checkpoints are changed.
"""

import argparse
import copy
import hashlib
import json

from beaker import Beaker, BeakerExperimentSpec
from olmoe3_lr_sweep_plan import AUTOMATION, MOUNT, WORKSPACE, checkpoint_complete, runs
from olmoe3_lr_sweep_watch import atomic_json, log, status


def main():
    """Prove the rejected name has no visible workload and submit one disambiguated name."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, choices=[r.run_id for r in runs()])
    args = parser.parse_args()
    assert MOUNT.is_mount(), "Reconciliation requires the real shared checkpoint mount"
    r = next(r for r in runs() if r.run_id == args.run_id)
    original = f"{r.run_id}-train"
    replacement = original + "-reconciled1"
    receipt_path = AUTOMATION / "submissions" / f"{original}.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("experiment_id"):
        log("already reconciled", **receipt)
        return
    assert receipt["phase"] == "submitting"
    spec = json.loads((AUTOMATION / "specs" / f"{original}.json").read_text())
    assert (
        hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        == receipt["spec_sha256"]
    )
    intent = AUTOMATION / "repairs" / f"{replacement}.json"
    with Beaker.from_env(default_workspace=WORKSPACE, check_for_upgrades=False) as b:
        matches = [
            w
            for w in b.workload.list(
                org=b.organization.get("ai2"), name_or_description=r.run_id, limit=100
            )
            if w.HasField("experiment") and w.experiment.name.startswith(original)
        ]
        assert not matches, "Existing workload requires reconciliation by ID, not another launch"
        assert not intent.exists(), "A prior repair intent must be reconciled before retry"
        assert json.loads(receipt_path.read_text()) == receipt
        if r.parent:
            parent_receipt = json.loads(
                (AUTOMATION / "submissions" / f"{r.parent}-train.json").read_text()
            )
            assert status(b.workload.get(parent_receipt["experiment_id"])) == "STATUS_SUCCEEDED"
            assert checkpoint_complete(r.parent_path), f"Incomplete fork: {r.parent_path}"
            assert (r.parent_path.parent / "audit/completed-step6000.json").is_file()
        assert len(spec["tasks"]) == 1
        task = spec["tasks"][0]
        assert task["replicas"] * task["resources"]["gpuCount"] == 64
        assert task["arguments"][2] == r.run_id
        assert task["context"]["priority"] == "urgent"
        assert task["context"]["minRuntime"] == "1h"
        assert (
            next(v["value"] for v in task["envVars"] if v["name"] == "GIT_REF")
            == receipt["source_commit"]
        )
        parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec))
        audit = {
            "original_name": original,
            "replacement_name": replacement,
            "reason": "Operator-authorized repair after Beaker database conflict; "
            "organization-wide name lookup found no experiment; original intent preserved",
            "original_receipt": receipt,
        }
        atomic_json(intent, {**audit, "phase": "submitting"})
        w = b.experiment.create(spec=parsed, name=replacement, workspace=b.workspace.get(WORKSPACE))
        atomic_json(intent, {**audit, "phase": "submitted", "experiment_id": w.experiment.id})
        atomic_json(
            receipt_path,
            {
                **receipt,
                "phase": "submitted",
                "experiment_id": w.experiment.id,
                "manual_reconciliation": str(intent),
            },
        )
        log("RECONCILED_SUBMISSION", experiment_id=w.experiment.id, name=replacement)


if __name__ == "__main__":
    main()
