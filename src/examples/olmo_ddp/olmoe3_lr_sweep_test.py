"""CPU regression checks for sweep math and duplicate-safe submission."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import olmoe3_lr_sweep_watch as watch
from olmoe3_lr_sweep_plan import BATCH, runs, smoke_runs, validate_plan


class SweepTests(unittest.TestCase):
    def test_grid(self):
        self.assertEqual(len(validate_plan()), 25)
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
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(watch, "AUTOMATION", Path(directory)),
        ):
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
