"""CPU-only checks for readiness, lineage isolation and bounded dependency fan-out."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import olmoe3_hero_stable_control as control
import olmoe3_hero_stable_eval as worker
from olmoe3_hero_stable_launch import controller_spec


class Checks(unittest.TestCase):
    def setUp(self):
        self.specs = {stage: {} for stage in worker.STAGES}
        self.client = Mock()
        self.client.ensure.return_value = SimpleNamespace(experiment=SimpleNamespace(id="test"))
        self.fs = SimpleNamespace(f_bavail=20_000_000_000_000, f_frsize=1)

    def advance(self, states):
        self.client.report.side_effect = states
        with (
            patch.object(control, "stage_source", return_value=True),
            patch.object(control.os, "statvfs", return_value=self.fs),
            patch.object(control, "converted"),
            patch.object(control, "qualified"),
            patch.object(control, "completed_bundle"),
        ):
            return control.advance(self.client, "emo", self.specs)

    def test_missing_checkpoint_never_submits(self):
        with patch.object(control, "stage_source", return_value=False):
            self.assertEqual(
                control.advance(self.client, "emo", self.specs)["state"], "waiting_for_checkpoint"
            )
        self.client.ensure.assert_not_called()

    def test_conversion_failure_blocks_dependents(self):
        self.assertEqual(self.advance(["STATUS_FAILED"])["state"], "needs_attention")
        self.assertEqual(self.client.ensure.call_count, 1)

    def test_queued_conversion_blocks_dependents(self):
        self.assertEqual(self.advance(["STATUS_QUEUED"])["state"], "in_progress")
        self.assertEqual(self.client.ensure.call_count, 1)

    def test_qualification_failure_blocks_full_evals(self):
        self.assertEqual(
            self.advance(["STATUS_SUCCEEDED", "STATUS_FAILED"])["state"], "needs_attention"
        )
        self.assertEqual(self.client.ensure.call_count, 2)

    def test_successful_dependencies_release_exactly_three_bundles(self):
        result = self.advance(["STATUS_SUCCEEDED"] * 2 + ["STATUS_QUEUED"] * 3)
        self.assertEqual(set(result["jobs"]), set(worker.STAGES))
        self.assertEqual(self.client.ensure.call_count, 5)
        self.assertEqual(len(set(c.args[0] for c in self.client.ensure.call_args_list)), 5)

    def test_completed_chain(self):
        self.assertEqual(self.advance(["STATUS_SUCCEEDED"] * 5)["state"], "complete")

    def test_staged_copy_survives_original_removal_and_rejects_wrong_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "olmo-core"
            raw.mkdir()
            metadata = raw / ".metadata.json"
            metadata.write_text("{}")
            source = worker.parent("emo").root / "step120000"
            receipt = dict(
                source=str(source),
                destination=str(raw),
                step=120000,
                all_file_hashes_verified=True,
                sizes={".metadata.json": 2},
            )
            event = dict(
                run_id=worker.parent("emo").run_id,
                lineage_id=worker.parent("emo").run_id,
                step=120000,
                checkpoint_path=str(source),
                checkpoint_metadata_sha256=hashlib.sha256(b"{}").hexdigest(),
            )
            (root / "olmo-core-copy.json").write_text(json.dumps(receipt))
            (root / "parent-ready.json").write_text(json.dumps(event))
            with (
                patch.object(worker, "output_root", return_value=root),
                patch.object(worker, "validate_checkpoint"),
                patch.object(worker, "inventory", return_value={".metadata.json": (2, 0, 0)}),
            ):
                self.assertTrue(worker.staged("emo"))
                with self.assertRaises(AssertionError):
                    worker.staged("non-emo")
                event["checkpoint_metadata_sha256"] = "bad"
                (root / "parent-ready.json").write_text(json.dumps(event))
                with self.assertRaises(AssertionError):
                    worker.staged("emo")

    def test_false_ready_does_not_copy(self):
        with (
            patch.object(worker, "staged", return_value=False),
            patch.object(worker, "ready", return_value=False),
            patch.object(worker, "verified_copy") as copy,
        ):
            self.assertFalse(worker.stage_source("emo"))
            copy.assert_not_called()

    def test_only_two_stable_lineages(self):
        self.assertNotEqual(worker.ROOT, worker.decay_eval.EVAL_ROOT)
        for arm in worker.ARMS:
            self.assertEqual(worker.parent(arm).run_id, "olmo35-small-hero-20260907-" + arm)
        with self.assertRaises(ValueError):
            worker.parent("decay")

    def test_cpu_controller_omits_all_resources_and_results(self):
        template = {
            "tasks": [
                {
                    "resources": {"gpuCount": 8, "cpuCount": 16},
                    "result": {"path": "/results"},
                    "envVars": [],
                }
            ]
        }
        for check in (True, False):
            task = controller_spec(template, "a" * 40, check=check)["tasks"][0]
            self.assertNotIn("resources", task)
            self.assertNotIn("result", task)
            self.assertEqual(task["constraints"], {"cluster": ["ai2/phobos"]})
            self.assertEqual(task["context"]["priority"], "urgent")

    def test_restart_and_lost_response_never_duplicate_submissions(self):
        from beaker import BeakerExperimentSpec

        for failure in (None, "after_create", "before_create"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                spec = {
                    "tasks": [
                        {
                            "context": {"priority": "urgent", "minRuntime": "1h"},
                            "envVars": [{"name": "GIT_REF", "value": "a" * 40}],
                        }
                    ]
                }
                work = SimpleNamespace(
                    experiment=SimpleNamespace(id="fake-id", name="test-name"),
                    HasField=lambda field: field == "experiment",
                )
                remote = []
                b = Mock()
                b.workload.list.side_effect = lambda **kwargs: remote
                b.workload.get.return_value = work
                b.experiment.get_spec.return_value.to_json.return_value = spec

                def create(**kwargs):
                    if failure != "before_create":
                        remote.append(work)
                    if failure is not None:
                        raise ConnectionError("simulated lost create response")
                    return work

                b.experiment.create.side_effect = create
                with (
                    patch.object(control, "AUTOMATION", Path(temp)),
                    patch.object(BeakerExperimentSpec, "from_json", return_value=Mock()),
                ):
                    c = control.EvalController(b, "a" * 40)
                    if failure:
                        with self.assertRaises(ConnectionError):
                            c.ensure("test-name", spec)
                    else:
                        self.assertIs(c.ensure("test-name", spec), work)
                    restarted = control.EvalController(b, "a" * 40)
                    found = restarted.ensure("test-name", spec)
                    if failure == "before_create":
                        self.assertIsNone(found)
                    else:
                        self.assertIs(found, work)
                    self.assertEqual(b.experiment.create.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
