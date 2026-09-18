"""CPU regressions for timeout isolation and the bounded temperature sweep."""

import copy
import json
import os
import subprocess
import unittest
from unittest.mock import patch

import olmoe3_hero_4t_evals as worker


class EvalRecoveryTest(unittest.TestCase):
    def test_approved_storage_floor_only(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(worker.minimum_free_bytes(), 12_000_000_000_000)
        with patch.dict(os.environ, {"HERO_4T_EVAL_MIN_FREE_BYTES": "10000000000000"}):
            self.assertEqual(worker.minimum_free_bytes(), 10_000_000_000_000)
        with patch.dict(os.environ, {"HERO_4T_EVAL_MIN_FREE_BYTES": "0"}):
            with self.assertRaises(AssertionError):
                worker.minimum_free_bytes()

    def test_timeout_is_contained(self):
        with patch.object(worker.subprocess, "run", side_effect=subprocess.TimeoutExpired("x", 300)):
            with patch.object(worker, "log") as log:
                self.assertIsNone(worker.stage_call("decay"))
                self.assertEqual(log.call_args.kwargs["stage"], "decay")

    def test_failed_process_does_not_become_success(self):
        failed = subprocess.CompletedProcess([], 1, "", "failure")
        with patch.object(worker.subprocess, "run", return_value=failed):
            self.assertEqual(worker.stage_call("sft").returncode, 1)

    def test_only_temperature_changes_worker_spec(self):
        original = {
            "description": json.dumps({"model": "fixed-model", "sampling": "old"}),
            "tasks": [{"arguments": ["fixed-command"], "context": {},
                       "envVars": [{"name": "SECRET", "secret": "only-reference"}]}],
        }
        saved = copy.deepcopy(original)
        for temperature in worker.TEMPERATURES:
            spec = worker.temperature_spec(original, temperature)
            self.assertEqual(spec["tasks"][0]["arguments"], ["fixed-command"])
            self.assertEqual(spec["tasks"][0]["envVars"][0], original["tasks"][0]["envVars"][0])
            self.assertIn({"name": "HERO_SFT_TEMPERATURE", "value": str(temperature)},
                          spec["tasks"][0]["envVars"])
        self.assertEqual(original, saved)
        with self.assertRaises(AssertionError):
            worker.temperature_spec(original, 2)


if __name__ == "__main__":
    unittest.main()
