"""Small CPU guards for the 4T watcher migration (no cluster writes)."""

import copy
import unittest
from unittest.mock import patch

import olmoe3_hero_sft_plan as plan
from olmoe3_hero_4t_pipeline import MT_LC_COMMIT, ensure
from olmoe3_hero_sft_convert import export_root
from olmoe3_hero_sft_eval_control import BUNDLES, model_path


class FourTSFTPlanTest(unittest.TestCase):
    def test_only_two_requested_runs(self):
        runs = plan.runs()
        self.assertEqual(len(runs), 2)
        self.assertEqual({r.arm for r in runs}, {"emo", "non-emo"})
        self.assertTrue(all(r.epochs == 2 and r.lr == 5e-5 and not r.emo for r in runs))
        self.assertEqual(len({r.run_id for r in runs + plan.runs(True)}), 4)

    def test_new_dataset_old_4t_sources(self):
        self.assertEqual(plan.DATA.name, "gptoss120b-high-olmo-thinker-20260917")
        for r in plan.runs():
            self.assertIn("4t-lc100b-noemo-20260916", str(r.source))
            self.assertEqual(r.source.name, "step5961")
            self.assertIn(r.arm, str(r.source))
            self.assertIn("4t-gptoss-high", r.run_id)

    def test_actual_packed_horizon_not_old_1810(self):
        with patch.object(plan, "data_plan", return_value={"steps_per_epoch": 1713}):
            for r in plan.runs():
                self.assertEqual(r.total_steps, 3426)
                self.assertEqual(r.checkpoint_steps, [1713, 3426])
                self.assertEqual(model_path(r), export_root(r) / r.arm / "step3426/hf")

    def test_reject_unrequested_schedules(self):
        for args in [("emo", "3em4"), ("emo", "5em5", False, 5)]:
            with self.assertRaises(AssertionError):
                plan.SFTRun(*args)

    def test_wide_sweep_preserves_baselines(self):
        trials = plan.lr_sweep_runs()
        self.assertEqual(len(trials), 16)
        self.assertEqual(len(plan.new_lr_runs()), 14)
        self.assertEqual(len({r.run_id for r in trials}), 16)
        self.assertEqual(max(plan.LRS.values()) / min(plan.LRS.values()), 40)
        self.assertTrue(all(not r.emo and r.epochs == 2 for r in trials))
        self.assertTrue(all("4t-lc100b-noemo" in str(r.source) for r in trials))
        self.assertEqual(
            {r.run_id for r in trials if r.lr_label == "5em5"}, {r.run_id for r in plan.runs()}
        )
        self.assertTrue(all(plan.find_run(r.run_id) == r for r in trials))

    def test_eval_bundles(self):
        self.assertEqual(set(BUNDLES), {"math500", "ifbench", "humaneval", "alpaca"})

    def test_preserve_old_worker_pin_and_restore_controller_pin(self):
        class FakeController:
            commit = "a" * 40

            def ensure(self, name, spec):
                self.seen_commit = self.commit
                return None

            def report(self, work):
                return "AMBIGUOUS"

        control = FakeController()
        spec = {"tasks": [{"envVars": [{"name": "GIT_REF", "value": MT_LC_COMMIT}]}]}
        original = copy.deepcopy(spec)
        with patch("beaker.BeakerExperimentSpec.from_json", return_value=None):
            ensure(control, "existing-mt", spec, {})
        self.assertEqual(control.seen_commit, MT_LC_COMMIT)
        self.assertEqual(control.commit, "a" * 40)
        self.assertEqual(spec, original)


if __name__ == "__main__":
    unittest.main()
