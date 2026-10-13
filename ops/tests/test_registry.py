"""Tests for opslib.registry (tmp state dirs only)."""
import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from opslib import registry


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "runs.jsonl"
        self._env = os.environ.get("OPS_STATE_DIR")
        os.environ["OPS_STATE_DIR"] = self.tmp.name

    def tearDown(self):
        if self._env is None:
            os.environ.pop("OPS_STATE_DIR", None)
        else:
            os.environ["OPS_STATE_DIR"] = self._env
        self.tmp.cleanup()

    def test_env_state_dir_resolved_at_call_time(self):
        self.assertEqual(registry.runs_path(), self.path)

    def test_register_and_fold(self):
        registry.register(run_id="r1", experiment="e", cluster="beaker", job_id="J1", launched_at="2026-09-01T00:00:00Z")
        registry.update("r1", final_state="done", state_source="x.json")
        registry.register(run_id="r1", experiment="e", cluster="beaker", job_id="J2")  # relaunch
        run = registry.load_runs()["r1"]
        self.assertEqual(run["job_id"], "J2")
        self.assertEqual(run["final_state"], "done")
        self.assertEqual(run["n_events"], 3)
        self.assertEqual(len(self.path.read_text().splitlines()), 3)

    def test_validation(self):
        with self.assertRaises(ValueError):
            registry.register(run_id="r", experiment="e")  # no cluster
        with self.assertRaises(ValueError):
            registry.register(run_id="r", experiment="e", cluster="mars")
        with self.assertRaises(ValueError):
            registry.register(run_id="r", experiment="e", cluster="slurm", kind="party")
        with self.assertRaises(ValueError):
            registry.register(run_id="r", experiment="e", cluster="slurm", bogus_field=1)
        registry.register(run_id="r", experiment="e", cluster="slurm")
        with self.assertRaises(ValueError):
            registry.update("r", final_state="finished")
        with self.assertRaises(KeyError):
            registry.update("unknown", final_state="done")
        self.assertFalse(self.path.read_text().count("\n") > 1)

    def test_defaults(self):
        os.environ.pop("CLAUDE_SESSION_ID", None)
        row = registry.register(run_id="r", experiment="e", cluster="local")
        self.assertEqual(row["launched_by"], "manual")
        self.assertEqual(row["kind"], "train")
        os.environ["CLAUDE_SESSION_ID"] = "sess-1"
        try:
            self.assertEqual(registry.register(run_id="r2", experiment="e", cluster="local")["launched_by"], "sess-1")
        finally:
            os.environ.pop("CLAUDE_SESSION_ID")

    def test_concurrent_appends_do_not_interleave(self):
        def worker(i):
            for j in range(25):
                registry.register(run_id=f"r{i}-{j}", experiment="e", cluster="slurm", notes="x" * 2000)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 100)
        for line in lines:
            json.loads(line)

    def test_find_filters(self):
        registry.register(run_id="old", experiment="a", cluster="slurm", launched_at="2020-01-01T00:00:00Z")
        registry.register(run_id="new", experiment="a", cluster="beaker")
        registry.register(run_id="other", experiment="b", cluster="beaker")
        self.assertEqual([r["run_id"] for r in registry.find(experiment="a", since="7d")], ["new"])
        self.assertEqual({r["run_id"] for r in registry.find(cluster="beaker")}, {"new", "other"})

    def test_to_iso_utc_formats(self):
        self.assertEqual(registry.to_iso_utc("2026-08-12T09:47:26-07:00"), "2026-08-12T16:47:26Z")
        self.assertEqual(registry.to_iso_utc("2026-09-14T21:23:22Z"), "2026-09-14T21:23:22Z")
        self.assertEqual(registry.to_iso_utc("2026-09-08 21:46"), "2026-09-09T04:46:00Z")  # naive = Pacific
        self.assertEqual(registry.to_iso_utc("2026-08-25"), "2026-08-25T07:00:00Z")
        self.assertIsNone(registry.to_iso_utc("yesterday"))

    def test_coerce_and_malformed_lines(self):
        self.assertEqual(registry.coerce("expected_minutes", "90"), 90)
        self.assertIs(registry.coerce("dirty", "true"), True)
        self.assertIsNone(registry.coerce("job_id", "null"))
        self.assertEqual(registry.coerce("job_id", "3492749"), "3492749")
        self.path.write_text('{"event":"launch","run_id":"a","experiment":"e","cluster":"slurm"}\nnot json\n')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(list(registry.load_runs()), ["a"])

    def test_cli_roundtrip(self):
        from opslib.cli import main

        with contextlib.redirect_stdout(io.StringIO()):
            main(["register", "--run-id", "c1", "--experiment", "e", "--cluster", "slurm", "--expected-minutes", "5"])
            main(["update", "c1", "final_state=failed", "notes=OOM"])
        run = registry.load_runs()["c1"]
        self.assertEqual((run["final_state"], run["notes"], run["expected_minutes"]), ("failed", "OOM", 5.0))


if __name__ == "__main__":
    unittest.main()
