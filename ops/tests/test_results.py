"""Tests for opslib.results (tmp state dirs only)."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from opslib import registry, results


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

    def test_rows_from_multirung_meta(self):
        # native drivers record eval sizes + provenance inside _meta (never as top-level dicts)
        multirung = {"model_path": "/w/ckpts/ds64-nq-dense-u16M/step90", "nq_8k": 0.5, "nq_16k": 0.4,
                     "_meta": {"ladder_version": "v2", "eval_size": {"nq_8k": 500}, "eval_job_id": "01J",
                               "git_commit": "abc", "cot_label": "no-cot"}}
        rows = {r["rung"]: r for r in results.rows_from_json(multirung)}
        self.assertEqual(rows["8k"]["eval_size"], 500)
        self.assertNotIn("eval_size", rows["16k"])  # no size recorded -> stays missing, never guessed
        self.assertEqual((rows["8k"]["eval_job_id"], rows["8k"]["git_commit"], rows["8k"]["cot_label"]),
                         ("01J", "abc", "no-cot"))
        self.assertEqual(rows["8k"]["run_id"], "ds64-nq-dense-u16M")
        # explicit defaults (CLI --eval-size) win over _meta
        self.assertEqual(results.rows_from_json(multirung, {"eval_size": 600})[0]["eval_size"], 600)

    def test_rows_from_ctc_eval_outcome(self):
        # ctc/src/ctc/eval/runner.py EvalOutcome.to_dict(): the results_io successor
        outcome = {"task": "contradiction", "rung": "8k", "attn": "full", "backend": "vllm", "eval_size": 500,
                   "metrics": {"f1": 0.8, "precision": 0.9}, "standard_errors": {"f1": 0.018},
                   "primary_metric": "f1", "primary_value": 0.8, "parse_rate": 1.0, "cot_label": "no-cot",
                   "provenance": {"ckpt": "/data/ckpts/run-b/step50", "git_commit": "g1"}}
        (r,) = results.rows_from_json(outcome)
        self.assertEqual((r["run_id"], r["task"], r["rung"], r["metric"], r["value"]), ("run-b", "contradiction", "8k", "f1", 0.8))
        self.assertEqual((r["eval_size"], r["se"], r["cot_label"], r["eval_backend"]), (500, 0.018, "no-cot", "vllm"))
        self.assertEqual((r["ckpt_path"], r["git_commit"]), ("/data/ckpts/run-b/step50", "g1"))
        results.normalize(r)  # a valid contract row as-is

    def test_train_run_from_registered_eval_pass(self):
        registry.register(run_id="train-a", experiment="exp-a", cluster="beaker")
        registry.register(run_id="train-a__ev-nq-yarn2-202610121200", experiment="exp-a", cluster="beaker",
                          kind="eval", parent_run="train-a")
        r = results.add(**self.row(run_id="train-a__ev-nq-yarn2-202610121200", experiment=None))
        self.assertEqual((r["train_run"], r["experiment"]), ("train-a", "exp-a"))
        # a non-eval run id is its own train_run; an explicit train_run wins
        self.assertEqual(results.normalize(self.row(run_id="train-a"))["train_run"], "train-a")
        self.assertEqual(results.normalize(self.row(run_id="x", train_run="t"))["train_run"], "t")

    def test_load_defaults_train_run_for_old_rows(self):
        registry.register(run_id="ev-1", experiment="e", cluster="beaker", kind="eval", parent_run="train-b")
        path = Path(self.tmp.name) / "results.jsonl"
        old = {k: v for k, v in self.row(run_id="legacy-train").items()}
        path.write_text(json.dumps(old) + "\n" + json.dumps(self.row(run_id="ev-1")) + "\n")
        rows = results.load()
        self.assertEqual([r["train_run"] for r in rows], ["legacy-train", "train-b"])
        self.assertNotIn("train_run", path.read_text())  # filled on read, never rewritten

    def test_check_passes_of_same_train_run(self):
        prior = [results.normalize(self.row(run_id="t__ev-a", train_run="t", value=0.90))]
        warns = results.check(results.normalize(self.row(run_id="t__ev-b", train_run="t", value=0.70)), prior=prior)
        self.assertFalse(any(w.startswith("duplicate") for w in warns))  # a new pass is not a duplicate
        self.assertTrue(any("prior pass of t" in w and "explain before quoting" in w for w in warns))
        other = results.check(results.normalize(self.row(run_id="u__ev-a", train_run="u", value=0.70)), prior=prior)
        self.assertFalse(any("deviation" in w for w in other))  # worse than another run's best: normal

    def test_rows_from_multirung_eval_run_id(self):
        multirung = {"model_path": "/w/ckpts/run-c/step10", "nq_8k": 0.5,
                     "_meta": {"eval_run_id": "run-c__ev-nq-docchunk-dense-202610121200", "train_run": "run-c",
                               "eval_size": {"nq_8k": 500}}}
        (r,) = results.rows_from_json(multirung)
        self.assertEqual((r["run_id"], r["train_run"]), ("run-c__ev-nq-docchunk-dense-202610121200", "run-c"))
        # a YaRN serving copy (same weights) still scores the trained run
        (y,) = results.rows_from_json({"model_path": "/w/ckpts/run-c/step10_yarn2", "nq_256k": 0.4})
        self.assertEqual((y["run_id"], y["train_run"]), ("run-c", "run-c"))

    def test_cli_show_groups_by_train_run(self):
        from opslib.cli import main

        for rid, tr, v in (("t2__ev-x", "t2", 0.5), ("t1__ev-x", "t1", 0.6), ("t1__ev-y", "t1", 0.8)):
            results.add(**self.row(run_id=rid, train_run=tr, value=v))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["result", "show"])
        body = out.getvalue().splitlines()
        self.assertTrue(body[0].startswith("train_run"))
        self.assertEqual([l.split()[0] for l in body[1:4]], ["t1", "t1", "t2"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["result", "show", "--best"])
        self.assertIn("t1__ev-y", out.getvalue())  # the best pass is named
        self.assertIn("(1 rows)", out.getvalue())

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
