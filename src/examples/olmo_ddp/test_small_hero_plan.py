"""CPU-only schedule, namespace, storage and launch-spec regression checks."""

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import olmoe3_small_hero_plan as p
from olmoe3_small_hero_control import training_spec, validation_spec


class HeroPlanTest(unittest.TestCase):
    """Test the safety boundaries without Torch, cluster access, or writes to Weka."""

    def test_schedule(self):
        p.validate_plan()
        fixed = set(range(100, p.SWITCH_STEP + 1, 100))
        for step in range(p.FINAL_STEPS + 1):
            native = step == 0 or step in fixed or step % 500 == 0
            self.assertEqual(p.scheduled_save(step), native)
        self.assertTrue(p.scheduled_save(60000))
        self.assertFalse(p.scheduled_save(60100))
        self.assertTrue(p.scheduled_save(60500))
        self.assertEqual(p.INITIAL_STOP * p.BATCH, 3_003_121_664_000)

    def test_storage(self):
        self.assertEqual(p.disk_action(p.STOP_BYTES - 1), "stop")
        self.assertEqual(p.disk_action(p.STOP_BYTES), "warn")
        self.assertEqual(p.disk_action(p.WARN_BYTES), "ok")

    def test_independent_namespaces(self):
        self.assertEqual({r.bucket for r in p.runs()}, {p.BUCKET})
        self.assertEqual({r.prefix for r in p.runs()}, {"emo", "non-emo"})
        self.assertEqual({r.bucket for r in p.runs(True)}, {p.TEST_BUCKET})
        with self.assertRaises(StopIteration):
            p.find_run("unapproved")

    def test_specs(self):
        task = {
            "name": "train-replica-0",
            "envVars": [{"name": "GITHUB_TOKEN", "secret": "x"}],
            "datasets": [
                {"mountPath": "/gantry", "source": {"beaker": "entry"}},
                {"mountPath": "/weka/other", "source": {"weka": "other"}},
            ],
            "resources": {"gpuCount": 8},
            "context": {"priority": "normal"},
            "constraints": {
                "hostname": [f"good-node-{i}" for i in range(8)] + sorted(p.EXCLUDED_HOSTNAMES)
            },
            "synchronizedStartTimeout": "90m",
        }
        template = {"version": "v2", "tasks": [copy.deepcopy(task) for _ in range(8)]}
        before = json.dumps(template)
        for r in p.runs() + p.runs(True):
            spec = training_spec(template, r, "commit", r.smoke)
            self.assertEqual(len(spec["tasks"]), 1)
            t = spec["tasks"][0]
            self.assertEqual(t["replicas"] * t["resources"]["gpuCount"], 64)
            self.assertEqual(t["context"]["minRuntime"], "1h")
            self.assertEqual(t["context"]["priority"], "urgent")
            self.assertEqual(t["constraints"], {"hostname": [f"good-node-{i}" for i in range(8)]})
            self.assertFalse(set(t["constraints"]["hostname"]) & p.EXCLUDED_HOSTNAMES)
            self.assertEqual(t["result"]["path"], "/noop-results")
            self.assertEqual(
                {d["mountPath"] for d in t["datasets"]},
                {"/gantry", str(p.MOUNT), str(p.DOLMA_MOUNT)},
            )
            self.assertIn({"name": "GITHUB_TOKEN", "secret": "x"}, t["envVars"])
        validation = validation_spec(template, "commit")["tasks"][0]
        self.assertEqual(validation["context"]["minRuntime"], "0s")
        self.assertNotIn("gpuCount", validation["resources"])
        self.assertEqual(json.dumps(template), before)


if __name__ == "__main__":
    unittest.main()
