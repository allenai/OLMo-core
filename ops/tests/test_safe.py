"""Tests for opslib.safe: the never-raising launcher entry points (tmp state dirs only)."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from opslib import registry, safe


class SafeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._env = {k: os.environ.get(k) for k in ("OPS_STATE_DIR", "OPS_EXPERIMENT", "SLURM_JOB_ID",
                                                    "BEAKER_WORKLOAD_ID", "BEAKER_EXPERIMENT_ID", "BEAKER_JOB_ID")}
        for k in self._env:
            os.environ.pop(k, None)
        os.environ["OPS_STATE_DIR"] = self.tmp.name
        self.runs = Path(self.tmp.name) / "runs.jsonl"

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def quiet(self, fn, *a, **kw):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            res = fn(*a, **kw)
        return res, out.getvalue(), err.getvalue()

    def test_state_dir_env_override(self):
        self.assertEqual(safe.state_dir(), Path(self.tmp.name))

    def test_register_launch_writes_row(self):
        row, out, _ = self.quiet(safe.register_launch, run_id="r1", experiment="e", cluster="beaker",
                                 job_id="01J", git_commit="abc", command="x")
        self.assertEqual(row["source"], "launch")
        self.assertIn("registered run_id=r1", out)
        run = registry.load_runs(self.runs)["r1"]
        self.assertEqual((run["job_id"], run["git_commit"], run["kind"]), ("01J", "abc", "train"))

    def test_register_launch_git_fills_only_missing(self):
        row, _, _ = self.quiet(safe.register_launch, git_dir=Path(self.tmp.name) / "not-a-repo", run_id="r2",
                               experiment="e", cluster="slurm", git_commit="shipped")
        self.assertEqual(row["git_commit"], "shipped")  # the shipped commit is never overwritten
        self.assertNotIn("git_branch", row)  # unknown git fields are dropped, not written as null

    def test_errors_are_swallowed(self):
        row, _, err = self.quiet(safe.register_launch, run_id="r", experiment="e", cluster="mars")
        self.assertIsNone(row)
        self.assertIn("[ops] WARNING", err)
        row, _, err = self.quiet(safe.register_launch, run_id="r", experiment="e", cluster="slurm", bogus=1)
        self.assertIsNone(row)
        row, _, err = self.quiet(safe.update_run, "never-registered", final_state="done")
        self.assertIsNone(row)
        self.assertIn("[ops] WARNING", err)
        self.assertFalse(self.runs.exists())
        res, _, err = self.quiet(safe.record_results, "/nonexistent/x.json")
        self.assertIsNone(res)

    def test_update_run(self):
        self.quiet(safe.register_launch, run_id="r1", experiment="e", cluster="beaker")
        row, _, _ = self.quiet(safe.update_run, "r1", job_id="01J", final_state="done")
        self.assertEqual(row["event"], "update")
        run = registry.load_runs(self.runs)["r1"]
        self.assertEqual((run["job_id"], run["final_state"]), ("01J", "done"))

    def test_derive_experiment(self):
        self.assertEqual(safe.derive_experiment("ds64-outlier-x", "whatever"), "ds64")
        self.assertEqual(safe.derive_experiment("ctc-outlier-cmix", "ctc-suite-outlier"), "ctc_suite_outlier")
        self.assertEqual(safe.derive_experiment("zzz", None, default="d"), "d")
        os.environ["OPS_EXPERIMENT"] = "forced"
        self.assertEqual(safe.derive_experiment("ds64-x", "g"), "forced")

    def test_eval_job_id(self):
        self.assertIsNone(safe.eval_job_id())
        os.environ["SLURM_JOB_ID"] = "123"
        self.assertEqual(safe.eval_job_id(), "123")
        os.environ["BEAKER_WORKLOAD_ID"] = "01W"
        self.assertEqual(safe.eval_job_id(), "01W")

    def test_eval_pass_ids_and_registration(self):
        from datetime import datetime

        rid = safe.make_eval_run_id("train-a", "contra,nq", "yarn 2", when=datetime(2026, 10, 12, 9, 5))
        self.assertEqual(rid, "train-a__ev-contra-nq-yarn-2-202610120905")
        self.assertEqual(safe.unique_run_id("ev-nq-train-a"), "ev-nq-train-a")
        self.quiet(safe.register_launch, run_id="train-a", experiment="exp-a", cluster="beaker")
        row, _, _ = self.quiet(safe.register_eval_pass, "ev-nq-train-a", "train-a", ckpt_path="/c")
        self.assertEqual((row["kind"], row["parent_run"], row["experiment"], row["cluster"]),
                         ("eval", "train-a", "exp-a", "local"))
        again, _, _ = self.quiet(safe.register_eval_pass, "ev-nq-train-a", "train-a")
        self.assertEqual(again["run_id"], "ev-nq-train-a")  # already registered: no second launch row
        self.assertEqual(len(self.runs.read_text().splitlines()), 2)
        self.assertRegex(safe.unique_run_id("ev-nq-train-a"), r"^ev-nq-train-a-\d{12}$")

    def test_record_results_from_multirung(self):
        self.quiet(safe.register_launch, run_id="ds64-nq-dense-u16M", experiment="ds64x", cluster="beaker")
        p = Path(self.tmp.name) / "nq_multirung.json"
        p.write_text(json.dumps({
            "model_path": "/w/ckpts/ds64-nq-dense-u16M", "nq_8k": 0.5, "nq_16k": 0.4,
            "_meta": {"eval_size": {"nq_8k": 500}, "eval_job_id": "01J", "git_commit": "abc"},
        }))
        stored, _, err = self.quiet(safe.record_results, p)
        self.assertEqual(len(stored), 1)  # nq_16k has no eval_size -> rejected, reported
        self.assertIn("REJECTED", err)
        self.assertEqual(stored[0]["experiment"], "ds64x")  # from the canonical registry
        lines = (Path(self.tmp.name) / "results.jsonl").read_text().splitlines()
        self.assertEqual(json.loads(lines[0])["eval_job_id"], "01J")


if __name__ == "__main__":
    unittest.main()
