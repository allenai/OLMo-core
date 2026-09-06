"""CPU regression checks for sweep math and duplicate-safe submission."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import olmoe3_lr_sweep_watch as watch
from olmoe3_lr_sweep_plan import BATCH, extension_runs, runs, smoke_runs, validate_plan


class SweepTests(unittest.TestCase):
    def test_exported_replica_group_is_reconstructed(self):
        template = {
            "version": "v2",
            "tasks": [
                {
                    "name": f"train-replica-{i}",
                    "envVars": [],
                    "context": {},
                    "resources": {"gpuCount": 8},
                }
                for i in range(8)
            ],
        }
        spec = watch.training_spec(template, runs()[0], "commit")
        self.assertEqual(len(spec["tasks"]), 1)
        self.assertEqual(spec["tasks"][0]["replicas"], 8)
        self.assertTrue(spec["tasks"][0]["leaderSelection"])
        self.assertEqual(len(template["tasks"]), 8)

    def test_grid(self):
        self.assertEqual(len(validate_plan()), 30)
        for parent in (r for r in runs() if not r.parent):
            children = [r for r in runs() if r.parent == parent.run_id]
            self.assertEqual([r.start for r in children], [5700, 5400, 4800, 4200])
            self.assertTrue(all(r.keep == 1 for r in children))
            self.assertEqual(parent.keep, 7)
        self.assertEqual(6000 * BATCH, 100_663_296_000)

    def test_smoke_fork(self):
        parent, child = smoke_runs()
        self.assertEqual(child.start, 2)
        self.assertEqual(child.parent_path, parent.root / "step2")

    def test_higher_lr_extension_is_isolated(self):
        selected = extension_runs()
        self.assertEqual(len(selected), 5)
        self.assertEqual({r.lr for r in selected}, {0.0104})
        self.assertEqual(sum(r.parent is None for r in selected), 1)
        self.assertEqual({r.start for r in selected if r.parent}, {4200, 4800, 5400, 5700})
        self.assertEqual(len([r for r in runs() if r not in selected]), 25)

    def test_extension_fanout_only_after_trunk_succeeds(self):
        controller = watch.Controller.__new__(watch.Controller)
        controller.commit, controller.template = "extension-commit", {}
        controller.items = extension_runs()
        controller.reuse_smoke = controller.smoke_verified = True
        submitted = []
        parent_status = "STATUS_RUNNING"

        def ensure(name, spec):
            submitted.append(name)
            return parent_status if name.endswith("-trunk-train") else "STATUS_SUCCEEDED"

        controller.ensure, controller.report = ensure, lambda value: value
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(watch, "validation_spec", return_value={}),
            patch.object(watch, "training_spec", return_value={}),
            patch.object(watch, "checkpoint_complete", return_value=True),
            patch.object(Path, "is_file", return_value=True),
        ):
            controller.automation = Path(directory)
            self.assertFalse(controller.tick())
            self.assertEqual(len(submitted), 2)
            self.assertIn("extension-1p04em2", submitted[0])
            self.assertIn("lr1p04em2-trunk", submitted[1])
            parent_status = "STATUS_SUCCEEDED"
            submitted.clear()
            self.assertTrue(controller.tick())
            self.assertEqual(len(submitted), 6)  # Validation plus this LR's five trajectories.
            self.assertEqual(set(submitted[1:]), {f"{r.run_id}-train" for r in extension_runs()})

    def test_failed_create_never_blindly_retried(self):
        controller = watch.Controller.__new__(watch.Controller)
        controller.workspace, controller.commit = None, "test-commit"
        attempts = []

        def create(**kwargs):
            attempts.append(kwargs)
            raise TimeoutError("response lost after server may have accepted request")

        controller.beaker = SimpleNamespace(
            workload=SimpleNamespace(list=lambda **kwargs: []),
            experiment=SimpleNamespace(create=create),
        )
        spec = {
            "version": "v2",
            "tasks": [
                {
                    "name": "main",
                    "image": {"beaker": "test"},
                    "command": ["true"],
                    "result": {"path": "/noop-results"},
                    "context": {"priority": "urgent"},
                }
            ],
        }
        with (tempfile.TemporaryDirectory() as directory,):
            controller.automation = Path(directory)
            with self.assertRaises(TimeoutError):
                controller.ensure("unique-name", spec)
            self.assertIsNone(controller.ensure("unique-name", spec))
            self.assertEqual(len(attempts), 1)
            spec["description"] = "changed"
            with self.assertRaisesRegex(AssertionError, "Spec drift"):
                controller.ensure("unique-name", spec)

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            watch.atomic_json(path, {"step": 4200})
            self.assertEqual(json.loads(path.read_text()), {"step": 4200})


if __name__ == "__main__":
    unittest.main()
