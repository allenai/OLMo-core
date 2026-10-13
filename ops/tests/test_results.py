"""Tests for opslib.results (tmp state dirs only)."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from opslib import results


class ResultsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._env = os.environ.get("OPS_STATE_DIR")
        os.environ["OPS_STATE_DIR"] = self.tmp.name

    def tearDown(self):
        if self._env is None:
            os.environ.pop("OPS_STATE_DIR", None)
        else:
            os.environ["OPS_STATE_DIR"] = self._env
        self.tmp.cleanup()

    def row(self, **kw):
        base = dict(run_id="r1", experiment="e", task="contradiction", rung="2k", metric="f1", value=0.7,
                    eval_size=500, ckpt_path="/c", git_commit="abc", eval_job_id="J")
        base.update(kw)
        return base

    def test_add_computes_se_and_flags(self):
        r = results.add(**self.row(eval_size=488))
        self.assertAlmostEqual(r["se"], (0.7 * 0.3 / 488) ** 0.5, places=5)
        self.assertIn("eval_size<500", r["flags"])
        self.assertEqual(len(results.load()), 1)

    def test_validation(self):
        with self.assertRaises(ValueError) as cm:
            results.add(**self.row(n=500))
        self.assertIn("eval_size", str(cm.exception))
        with self.assertRaises(ValueError):
            results.add(**self.row(eval_size=None))
        with self.assertRaises(ValueError):
            results.add(**self.row(eval_size=0))
        with self.assertRaises(ValueError):
            results.add(**self.row(value="abc"))
        with self.assertRaises(ValueError):
            results.add(**self.row(cot_label="cot"))
        r = results.add(backfill=True, **self.row(eval_size=None, git_commit=None))
        self.assertEqual(sorted(r["flags"]), ["backfill", "eval_size_missing", "provenance_incomplete"])
        self.assertNotIn("se", r)

    def test_non_proportion_has_no_se(self):
        self.assertNotIn("se", results.add(**self.row(metric="ce", value=2.3)))

    def test_check_duplicate_and_drift(self):
        prior = [results.normalize(self.row(value=0.90))]
        warns = results.check(results.normalize(self.row(value=0.70)), prior=prior)
        self.assertTrue(any(w.startswith("duplicate") for w in warns))
        self.assertTrue(any("explain before quoting" in w for w in warns))
        # within noise -> only the duplicate warning
        warns = results.check(results.normalize(self.row(value=0.71)), prior=[results.normalize(self.row(value=0.70))])
        self.assertEqual([w.split(":")[0] for w in warns], ["duplicate"])

    def test_check_experiment_best(self):
        prior = [results.normalize(self.row(run_id="a", value=0.60))]
        worse = results.check(results.normalize(self.row(run_id="b", value=0.30)), prior=prior)
        self.assertFalse(any("deviation" in w for w in worse))  # a worse arm is normal
        better = results.check(results.normalize(self.row(run_id="b", value=0.95)), prior=prior)
        self.assertTrue(any("experiment best" in w and "explain before quoting" in w for w in better))

    def test_check_abs_tolerance_without_se(self):
        prior = [results.normalize(self.row(metric="ce", value=1.0))]
        warns = results.check(results.normalize(self.row(metric="ce", value=1.2)), prior=prior)
        self.assertTrue(any("0.1 abs" in w for w in warns))

    def test_check_provenance_and_exact_zero(self):
        warns = results.check(results.normalize(self.row(value=0.0, ckpt_path=None, eval_job_id=None)), prior=[])
        self.assertTrue(any("provenance gap" in w and "ckpt_path" in w for w in warns))
        self.assertTrue(any("bug signature" in w for w in warns))
        self.assertTrue(any("bug signature" in w for w in results.check(results.normalize(self.row(value=1.0)), prior=[])))

    def test_rows_from_json_shapes(self):
        multirung = {"model_path": "/weka/x/ckpts/fs35-outlier-kv17-s16M", "variant": "dense",
                     "outlier_8k": 0.18, "outlier_32k": 0.015, "ruler": {}, "_meta": {"a": 1}}
        rows = results.rows_from_json(multirung, {"eval_size": 600})
        self.assertEqual({(r["task"], r["rung"]) for r in rows}, {("outlier", "8k"), ("outlier", "32k")})
        self.assertEqual(rows[0]["run_id"], "fs35-outlier-kv17-s16M")
        s8 = {"task": "contradiction", "metric_name": "set_f1", "metric_value": 0.95, "rung_tokens": 2048,
              "eval_size": 488, "cot_label": "no-cot", "arm": "full", "model": "m",
              "provenance": {"ckpt_path": "/data/ckpts/run-a/step100", "git_commit": "g", "eval_backend": "native"}}
        (r,) = results.rows_from_json([s8])
        self.assertEqual((r["run_id"], r["metric"], r["rung"], r["git_commit"]), ("run-a", "set_f1", 2048, "g"))
        self.assertEqual(len(results.rows_from_json([self.row(), self.row(rung="8k")])), 2)
        with self.assertRaises(ValueError):
            results.rows_from_json({"foo": 1})

    def test_cli_add_json_and_show(self):
        from opslib.cli import main

        f = Path(self.tmp.name) / "r.json"
        f.write_text(json.dumps([self.row(), self.row(rung="8k", value=0.5), self.row(rung="8k", value=0.6, run_id="r2")]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["result", "add", "--json", str(f)]), 0)
            self.assertEqual(main(["result", "show", "--best"]), 0)
        self.assertEqual(len(results.load()), 3)
        self.assertIn("(2 rows)", out.getvalue())  # best per task/rung


if __name__ == "__main__":
    unittest.main()
