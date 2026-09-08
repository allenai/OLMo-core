"""CPU-only schedule, namespace, storage and launch-spec regression checks."""

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import olmoe3_small_hero_plan as p
from olmoe3_small_hero_cadence_20260908 import PREVIOUS_COMMIT, resume_spec
from olmoe3_small_hero_control import training_spec, validation_spec


class HeroPlanTest(unittest.TestCase):
    """Test the safety boundaries without Torch, cluster access, or writes to Weka."""

    def test_schedule(self):
        p.validate_plan()
        fixed = set(p.fixed_checkpoint_steps())
        for step in range(p.FINAL_STEPS + 1):
            native = step == 0 or step in fixed or step % 500 == 0
            self.assertEqual(p.scheduled_save(step), native)
        self.assertTrue(p.scheduled_save(60000))
        self.assertTrue(p.scheduled_save(17900))
        self.assertTrue(p.scheduled_save(18000))
        self.assertFalse(p.scheduled_save(18100))
        self.assertTrue(p.scheduled_save(18250))
        self.assertFalse(p.scheduled_save(18300))
        self.assertTrue(p.scheduled_save(59750))
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

    def test_resume_spec(self):
        run = p.runs()[0]
        task = {
            "name": "train-replica-0",
            "arguments": [
                "python",
                "src/examples/olmo_ddp/olmoe3_small_hero_node.py",
                run.run_id,
                "ai2/holmes",
            ],
            "envVars": [{"name": "GIT_REF", "value": PREVIOUS_COMMIT}],
            "resources": {"gpuCount": 8},
            "context": {"priority": "urgent", "minRuntime": "1h"},
        }
        original = {"tasks": [copy.deepcopy(task) for _ in range(8)]}
        before = copy.deepcopy(original)
        resumed = resume_spec(original, run, "new-commit", 16500, "wandb-id")
        self.assertEqual(original, before)
        task = resumed["tasks"][0]
        self.assertEqual(task["replicas"], 8)
        env = {e["name"]: e.get("value") for e in task["envVars"]}
        self.assertEqual(env["OLMO35_HERO_EXPECTED_START"], "16500")
        self.assertEqual(env["WANDB_RUN_ID"], "wandb-id")
        self.assertEqual(env["WANDB_RESUME"], "must")
        self.assertEqual(env["GIT_REF"], "new-commit")
        original["tasks"][0]["envVars"][0]["value"] = "unrelated-source"
        with self.assertRaises(AssertionError):
            resume_spec(original, run, "new-commit", 16500, "wandb-id")


if __name__ == "__main__":
    unittest.main()
