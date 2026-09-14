"""CPU checks for the exact two-arm milestone and bounded dependency fan-out."""

import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import olmoe3_hero_3003_eval as campaign


class Checks(unittest.TestCase):
    def advance(self, states, available=True):
        control = Mock()
        control.ensure.return_value = SimpleNamespace(experiment=SimpleNamespace(id="job"))
        control.report.side_effect = states
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(campaign, "available", return_value=available),
            patch.object(campaign.stable, "output_root", return_value=Path(tmp)),
            patch.object(
                campaign.os,
                "statvfs",
                return_value=SimpleNamespace(f_bavail=20_000_000_000_000, f_frsize=1),
            ),
            patch.object(campaign.stable, "converted"),
            patch.object(campaign.stable, "qualified"),
            patch.object(campaign.stable, "completed_bundle"),
            patch.object(campaign.ruler, "verify_success"),
        ):
            result = campaign.advance(control, "non-emo", {s: {} for s in campaign.STAGES})
        return result, control.ensure.call_count

    def test_waiting_does_not_submit(self):
        result, count = self.advance([], available=False)
        self.assertEqual((result["state"], count), ("waiting_for_checkpoint", 0))

    def test_failed_conversion_blocks_all(self):
        result, count = self.advance(["STATUS_FAILED"])
        self.assertEqual((result["state"], count), ("needs_attention", 1))

    def test_failed_qualification_blocks_evals(self):
        result, count = self.advance(["STATUS_SUCCEEDED", "STATUS_FAILED"])
        self.assertEqual((result["state"], count), ("needs_attention", 2))

    def test_only_three_eval_bundles(self):
        result, count = self.advance(["STATUS_SUCCEEDED"] * 2 + ["STATUS_QUEUED"] * 3)
        self.assertEqual(count, 5)
        self.assertEqual(set(result["jobs"]), set(campaign.STAGES))

    def test_completion_requires_every_bundle(self):
        result, count = self.advance(["STATUS_SUCCEEDED"] * 5)
        self.assertEqual((result["state"], count), ("complete", 5))

    def test_exact_identity_and_step(self):
        self.assertEqual(campaign.STEP * 16777216, 3003121664000)
        self.assertEqual(campaign.AUTOMATION.parent, campaign.MOUNT / "uploader/automation")
        self.assertEqual(campaign.STATE, campaign.MOUNT / "uploader/state")
        self.assertNotIn("decay2t", str(campaign.ROOT))
        for arm in campaign.ARMS:
            self.assertEqual(
                campaign.stable.parent(arm).run_id, "olmo35-small-hero-20260907-" + arm
            )
            self.assertIn("step179000", str(campaign.model_path("3003t", arm)))
        with self.assertRaises(ValueError):
            campaign.available("unknown")

    def test_cpu_has_no_resources(self):
        template = {"tasks": [{"resources": {"gpuCount": 1}, "envVars": []}]}
        for check in (True, False):
            task = campaign.controller_spec(template, "a" * 40, check)["tasks"][0]
            self.assertNotIn("resources", task)
            self.assertEqual(task["constraints"], {"cluster": ["ai2/phobos"]})
            self.assertEqual(task["result"], {"path": "/noop-results"})
            command = shlex.split(task["arguments"][-1])
            self.assertEqual(command[-1], "preflight" if check else "watch")
            self.assertTrue(command[-2].endswith("/olmoe3_hero_3003_eval.py"))


if __name__ == "__main__":
    unittest.main()
