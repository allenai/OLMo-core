"""Tests for opslib.guard (stall-watch wrapper). Each child runs < 2 s; whole file < 5 s."""
import io
import os
import sys
import tempfile
import time
import unittest

from opslib import guard

PY = sys.executable


def run_guard(code, **kw):
    out = io.BytesIO()
    kw.setdefault("sample", 0.1)
    kw.setdefault("poll", 0.1)
    kw.setdefault("pyspy", False)
    t0 = time.time()
    rc = guard.guard([PY, "-c", code], out=out, **kw)
    return rc, out.getvalue().decode(), time.time() - t0


class TestGuard(unittest.TestCase):
    def test_passthrough_and_exit_code(self):
        rc, out, _ = run_guard("import sys; print('hello'); sys.exit(3)", first_output=5, stall=5)
        self.assertEqual(rc, 3)
        self.assertIn("hello", out)
        self.assertNotIn("STALL", out)

    def test_stall_after_output_reports_and_keeps_waiting(self):
        rc, out, dt = run_guard("import time; print(1, flush=True); time.sleep(1.2); print('done')",
                                first_output=5, stall=0.4)
        self.assertEqual(rc, 0)
        self.assertIn("[ops guard] STALL: no new output", out)
        self.assertIn("VERDICT: IDLE: sleeping", out)
        self.assertIn("done", out)  # kept waiting, child finished normally
        self.assertLess(out.index("STALL"), out.index("done"))

    def test_first_output_timeout_with_kill(self):
        rc, out, dt = run_guard("import time; time.sleep(30)", first_output=0.3, stall=5, kill=True,
                                name="unit")
        self.assertEqual(rc, 124)
        self.assertIn("[ops guard] STALL unit: no output since start", out)
        self.assertIn("--kill", out)
        self.assertLess(dt, 8)

    def test_child_env_unbuffered_faulthandler(self):
        rc, out, _ = run_guard("import os; print(os.environ['PYTHONUNBUFFERED'], os.environ['PYTHONFAULTHANDLER'])",
                               first_output=5, stall=5)
        self.assertIn("1 1", out)

    def test_log_tee(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "x.log")
            rc, out, _ = run_guard("print('to-log')", first_output=5, stall=5, log=log)
            with open(log) as fh:
                self.assertIn("to-log", fh.read())

    def test_command_not_found(self):
        out = io.BytesIO()
        self.assertEqual(guard.guard(["no-such-binary-xyz"], out=out), 127)

    def test_start_warnings(self):
        mounts = ["/accounts", "/net/horton/data"]
        w = guard.start_warnings(["/scratch/users/prasann/conda/envs/x/bin/python", "a.py"],
                                 log="/accounts/projects/x/run.log", cwd="/accounts/projects/x", host="horton",
                                 mounts=mounts)
        self.assertEqual(len(w), 3)
        self.assertTrue(any("interpreter" in x for x in w))
        self.assertEqual(guard.start_warnings(["/scratch/users/prasann/conda/envs/x/bin/python"],
                                              log="/accounts/x.log", cwd="/accounts", host="radagast"), [])
        self.assertEqual(guard.start_warnings(["/data/prasann/conda/envs/x/bin/python"], log="/data/prasann/x.log",
                                              cwd="/data/prasann", host="horton", mounts=mounts), [])

    def test_is_nfs(self):
        self.assertTrue(guard.is_nfs("/net/horton/data/prasann/x", mounts=[]))
        self.assertTrue(guard.is_nfs("/somewhere/x", mounts=["/somewhere"]))
        self.assertFalse(guard.is_nfs("/data/prasann/x", mounts=["/accounts"]))

    def test_proc_tree_and_diagnose(self):
        import subprocess
        p = subprocess.Popen([PY, "-c", "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(3)']); time.sleep(3)"])
        try:
            time.sleep(0.5)
            tree = guard.proc_tree(p.pid)
            self.assertGreaterEqual(len(tree), 2)
            ev = guard.diagnose(p.pid, 400, sample=0.1, pyspy=False)
            self.assertEqual(len(ev["procs"]), len(tree))
            self.assertTrue(ev["verdict"].startswith("IDLE"))
        finally:
            p.kill()
            p.wait()


if __name__ == "__main__":
    unittest.main()
