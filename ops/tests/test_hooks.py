"""Tests for hooks/bash_guard.py and hooks/agent_guard.py.

Command shapes are taken (lightly shortened) from Prasann's real Claude Code transcripts. Running the
guard over all 14,147 transcript Bash calls (2026-10-12) denied 718 polling loops, 312 foreground
sleeps >= 30 s, 133 login-node test runs and 2 NFS-interpreter fallbacks inside srun, and nothing else.
"""
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HOOKS = Path(__file__).resolve().parent.parent / "hooks"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HOOKS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bg = _load("bash_guard")
ag = _load("agent_guard")

LOGIN, NODE = "radagast", "horton"
NFS_PY = "/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python"
DATA_PY = "/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python"

# (command, host, background, expected) with expected in {"deny", "advise", "allow"} and an optional
# substring the deny/advice must contain.
CASES = [
    # ---- (i) tests on the login node
    (f"{NFS_PY} -m pytest -q src/test/nn/attention/document_chunked_test.py 2>&1 | tail -15", LOGIN, False, "deny", "login node"),
    ("cd /accounts/projects/berkeleynlp/prasann/projects/OLMo-core; PY=" + NFS_PY + "; timeout 900 $PY -m pytest -q "
     "src/test/nn/attention/kv_route_test.py 2>&1 | tail -1", LOGIN, False, "deny", "login node"),
    ("pytest -v src/test/nn/moe", LOGIN, False, "deny", "login node"),
    ("nohup " + NFS_PY + " -m pytest ctc/tests -q > /tmp/x.log 2>&1 & echo relaunched", LOGIN, False, "deny", "login node"),
    ("make checks", LOGIN, False, "deny", "login node"),
    ("cd repo && make type-check", LOGIN, False, "deny", "login node"),
    ("python -m unittest discover -s src/test", LOGIN, False, "deny", "login node"),
    ("pytest -v src/test/nn/moe", NODE, False, "allow", None),  # on a compute node: fine
    ("srun --partition=berkeleynlp --qos=preemptive_high_sewonm --nodelist=horton --cpus-per-task=16 --mem=64G "
     f"{DATA_PY} -m pytest -q src/test/nn", LOGIN, False, "allow", None),
    ("sbatch --parsable --job-name=flopdrop-gputests --partition=jsteinhardt --output=/data/prasann/t.log "
     "--wrap \"set -e; cd /data/prasann/x && " + DATA_PY + " -m pytest -q src/test\"", LOGIN, False, "advise", "register"),
    ("cd ops && python3 -m unittest discover -s tests -t .", LOGIN, False, "allow", None),
    ("python3 -m unittest discover -s ops/tests", LOGIN, False, "allow", None),
    ("python3 -m unittest tests.test_hooks -v", LOGIN, False, "allow", None),
    ("pytest --version", LOGIN, False, "allow", None),
    ("pip install pytest", LOGIN, False, "allow", None),
    ("grep -rn pytest src/test | head", LOGIN, False, "allow", None),
    ("git log --oneline --grep pytest", LOGIN, False, "allow", None),
    # heredoc bodies are data, not commands
    ("cat > $SP/run_tests.sh <<'EOF'\nROOT=$1\ncd $ROOT\n" + NFS_PY + " -m pytest -v src/test/nn\nwhile true; do sleep 60; done\nEOF\nchmod +x $SP/run_tests.sh",
     LOGIN, False, "allow", None),
    ("git commit -q -m \"$(cat <<'EOF'\nrun pytest; sleep 300; torchrun x\nEOF\n)\"", LOGIN, False, "allow", None),
    ("python - <<'PY'\nimport time\nwhile True:\n    time.sleep(60)\nPY", LOGIN, False, "allow", None),

    # ---- (ii) NFS interpreter in torch/vLLM launches or jobs
    (f"srun -p berkeleynlp -w horton --gres=gpu:1 {NFS_PY} train.py", LOGIN, False, "deny", "NFS interpreter"),
    ("nohup srun -p berkeleynlp -w horton --gres=gpu:1 -c 8 -t 01:00:00 bash -c '\nset -u\n"
     "PY=/data/prasann/ctc_vllm_venv/bin/python\n[ -x $PY ] || PY=" + NFS_PY + "\n$PY eval.py\n' &",
     LOGIN, False, "deny", "NFS interpreter"),
    (f"{NFS_PY} -m torch.distributed.run --nproc-per-node=8 train.py", NODE, False, "deny", "NFS interpreter"),
    ("sbatch -w sneetches -p jsteinhardt --output=/data/prasann/joblogs/x_%j.log --wrap '\nENV=/data/prasann/conda/envs/corpus-reasoning-olmo\n"
     "[ -d \"$ENV\" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo\nexport PATH=\"$ENV/bin:$PATH\"\n"
     "python -m torch.distributed.run --nproc-per-node=1 x.py'", LOGIN, False, "advise", "register"),
    (f"{NFS_PY} debug/ds64/launch_ds64.py --tasks contradiction dry_run 2>&1 | tail -40", LOGIN, False, "allow", None),
    (f"{NFS_PY} -m black --check src/olmo_core/nn/attention/chunked_mask.py", LOGIN, False, "allow", None),
    (f"timeout 600 srun -p jsteinhardt -w sneetches --gres=gpu:1 -c 8 bash -c 'PY=/data/prasann/ctc_vllm_venv/bin/python; "
     "$PY -m pip show vllm'", LOGIN, False, "allow", None),

    # ---- (iii) /net paths inside jobs
    ("srun -p berkeleynlp -w mooney --gres=gpu:1 " + DATA_PY + " eval.py --ckpt /net/sneetches/data/prasann/ckpts/x",
     LOGIN, False, "deny", "/net/"),
    ("sbatch -w mooney --output=/net/horton/data/prasann/joblogs/x.log run.sbatch", LOGIN, False, "deny", "/net/"),
    ("timeout 90 srun --partition=jsteinhardt --nodelist=mooney --mem=2G bash -c 'mkdir -p /data/prasann/ck; "
     "rsync -a /net/sneetches/data/prasann/ck/ /data/prasann/ck/'", LOGIN, False, "advise", "NFS speed"),
    ("srun --jobid=3813965 --overlap --ntasks=1 bash -c 'tail /net/horton/data/prasann/x.log'", LOGIN, False, "allow", None),
    ("tail -20 /net/cubbins/data/prasann/logs/steppart_scale_3521980.log", LOGIN, False, "allow", None),
    ("rsync -a /net/cubbins/data/prasann/hf_cache/hub/x /net/sneetches/data/prasann/hf_cache/hub/", LOGIN, False, "allow", None),
    ("cd /net/sneetches/data/prasann/devloss_grid/data && " + NFS_PY + " -c \"import json\"", LOGIN, False, "allow", None),

    # ---- (iv) bare torchrun
    ("torchrun --nproc-per-node=8 src/scripts/train/x.py", NODE, False, "deny", "bare `torchrun`"),
    ("srun -p berkeleynlp -w horton --gres=gpu:8 torchrun --nproc-per-node=8 x.py", LOGIN, False, "deny", "bare `torchrun`"),
    ("cd /data/x && export PATH=/data/prasann/conda/envs/e/bin:$PATH && torchrun --standalone x.py", NODE, False, "allow", None),
    ("ENV=/data/prasann/conda/envs/e; export PATH=\"$ENV/bin:$PATH\"; torchrun --standalone x.py", NODE, False, "allow", None),
    ("cd /data/x && source ~/.bashrc && torchrun --standalone x.py", NODE, False, "deny", "torchrun"),
    (f"{DATA_PY} -m torch.distributed.run --nproc-per-node=8 x.py", NODE, False, "allow", None),
    ("/data/prasann/conda/envs/e/bin/torchrun --nproc-per-node=8 x.py", NODE, False, "allow", None),
    ("gantry run -w ai2/flex2 -b ai2/oe-other --priority urgent -- torchrun --nproc-per-node=8 x.py", LOGIN, False, "advise", "register"),
    ("which torchrun; grep -n torchrun src/scripts/*.sbatch", LOGIN, False, "allow", None),
    ("ssh lambda 'cd /accounts/projects/sewonm/prasann && torchrun --nproc-per-node=8 x.py'", LOGIN, False, "allow", None),

    # ---- (v) polling loops / long sleeps
    ("timeout 580 bash -c 'while ! squeue -u prasann -h -t R | grep -q router-diff; do sleep 20; done'; squeue -u prasann",
     LOGIN, False, "deny", "polling loop"),
    ("until grep -q \"=== DONE ===\" /net/cubbins/data/prasann/logs/jointsweep_3523680.log 2>/dev/null; do sleep 30; done; echo PACKED_DONE",
     LOGIN, False, "deny", "polling loop"),
    ("EXP=01M2GWEZE4MKQRZEDRQRRSDYQG\nuntil beaker experiment get $EXP --format json 2>/dev/null | python3 -c \"\nimport json,sys\n"
     "sys.exit(0)\n\"; do\n  sleep 20\ndone\necho JOB_FINALIZED", LOGIN, False, "deny", "polling loop"),
    ("for i in $(seq 1 9); do pgrep -f collect_ds64.py >/dev/null || break; sleep 55; done", LOGIN, False, "deny", "polling loop"),
    ("end=$((SECONDS+540)); while [ $SECONDS -lt $end ]; do grep -q row /x.out && break; sleep 60; done", LOGIN, False, "deny", "polling loop"),
    ("sleep 240; tail -20 /net/cubbins/data/prasann/logs/steppart_scale_3521980.log", LOGIN, False, "deny", "sleep 240"),
    ("sleep 180; timeout 240 beaker job logs 01M2K33JAQA5020KY44823DWGS 2>&1 | tail -12", LOGIN, False, "deny", "sleep 180"),
    ("sleep 2m && squeue -u prasann", LOGIN, False, "deny", "sleep 2m"),
    ("until [ -f /x/done.json ]; do sleep 60; done", LOGIN, True, "advise", "polling loop"),  # background: allowed
    ("sleep 600; echo hi", LOGIN, True, "advise", "sleep 600"),
    ("kill 724728 724733; sleep 1; ps -eo pid,ppid,etime,args | grep pytest | grep -v grep", LOGIN, False, "allow", None),
    ("PID=$(pgrep -f orchestrate_ds64.py | head -1); [ -n \"$PID\" ] && kill \"$PID\"; sleep 6", LOGIN, False, "allow", None),
    ("for i in 1 2 3; do curl -sf http://x && break; sleep 5; done", LOGIN, False, "allow", None),
    ("for f in a b c; do echo $f; done; sleep 5", LOGIN, False, "allow", None),
    ("while read -r l; do echo \"$l\"; done < file.txt; sleep 3", LOGIN, False, "allow", None),
    ("sbatch --wrap 'while true; do date; sleep 60; done'", LOGIN, False, "advise", "register"),  # inside a job: ok
    ("python3 -c \"import time\nwhile True: time.sleep(60)\"", LOGIN, False, "allow", None),
    ("echo 'while true; do sleep 60; done'", LOGIN, False, "allow", None),

    # ---- advice
    ("squeue -u prasann -o '%.10i %.16j %.8T %.8M %N %r'", LOGIN, False, "advise", "ops status"),
    ("sacct -j 3598122 --format=State -n | head -1", LOGIN, False, "advise", "ops status"),
    ("beaker experiment get 01M2HF346WAH6K7201Q3YF1Y4B --format json | python3 -c 'import json'", LOGIN, False, "advise", "ops status"),
    ("timeout 200 beaker job logs 01M2K3676JT9GXSZPCKB5YJV71 2>&1 | tail -5", LOGIN, False, "advise", "ops status"),
    ("TASK=nq CONFIGS=l0.2 sbatch --nodelist=mcfuzz --job-name=router-smoke debug/learned_router/run_router_local.sbatch",
     LOGIN, False, "advise", "register"),
    ("python src/scripts/train/OLMo2-1B.py launch olmo2-1b-test ai2/jupiter-cirrascale-2 --launch.num_nodes=4",
     LOGIN, False, "advise", "register"),
    ("ls -la; git status --short | head", LOGIN, False, "allow", None),
    ("echo 'unbalanced", LOGIN, False, "allow", None),  # parse failure: fail open
]


class TestBashGuard(unittest.TestCase):
    def test_matrix(self):
        for cmd, host, background, expected, needle in CASES:
            with self.subTest(cmd=cmd[:90]):
                f = bg.check(cmd, host, background, "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core")
                got = "deny" if f.deny else ("advise" if f.advise else "allow")
                self.assertEqual(got, expected, f"deny={f.deny} advise={f.advise}")
                if needle:
                    self.assertIn(needle, " ".join(f.deny if expected == "deny" else f.advise))

    def test_sbatch_script_scan_is_advisory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "run_local.sbatch"
            p.write_text("#!/bin/bash\n#SBATCH --output=/accounts/projects/berkeleynlp/prasann/.x.log\n"
                         "PY=" + NFS_PY + "\ntorchrun --nproc-per-node=8 train.py\n")
            f = bg.check(f"sbatch {p.name}", LOGIN, False, d)
            self.assertEqual(f.deny, [])
            txt = " ".join(f.advise)
            self.assertIn("looks risky", txt)
            self.assertIn("on NFS", txt)
            self.assertIn("torchrun", txt)

    def _hook(self, script, payload, env=None):
        return subprocess.run([sys.executable, str(HOOKS / script)], input=payload, capture_output=True, text=True,
                              timeout=10, env=dict(os.environ, **(env or {})))

    def test_main_deny_exit2(self):
        r = self._hook("bash_guard.py", json.dumps({"tool_name": "Bash", "tool_input": {"command": "sleep 120"}}))
        self.assertEqual(r.returncode, 2)
        self.assertIn("sleep 120", r.stderr)

    def test_main_advise_json(self):
        r = self._hook("bash_guard.py", json.dumps({"tool_name": "Bash", "tool_input": {"command": "squeue -u prasann"}}))
        self.assertEqual(r.returncode, 0)
        out = json.loads(r.stdout)
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"], "PreToolUse")
        self.assertIn("ops status", out["hookSpecificOutput"]["additionalContext"])

    def test_main_hostname_override_and_fail_open(self):
        r = self._hook("bash_guard.py", json.dumps({"tool_name": "Bash", "tool_input": {"command": "pytest src"}}),
                       env={"OPS_HOOK_HOSTNAME": "radagast"})
        self.assertEqual(r.returncode, 2)
        r = self._hook("bash_guard.py", "this is not json")
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        r = self._hook("bash_guard.py", json.dumps({"tool_name": "Read", "tool_input": {"file_path": "/x"}}))
        self.assertEqual((r.returncode, r.stdout), (0, ""))


class TestAgentGuard(unittest.TestCase):
    def test_check(self):
        deny = [{"prompt": "x", "subagent_type": "general-purpose"},
                {"prompt": "x", "subagent_type": "Explore", "model": ""},
                {"prompt": "x", "model": "fable"},
                {"prompt": "x", "model": "claude-fable-5"},
                {"prompt": "x", "subagent_type": "Plan", "model": "inherit"},
                {}]
        allow = [{"prompt": "x", "model": "sonnet"}, {"prompt": "x", "model": "opus"},
                 {"prompt": "x", "model": "haiku"}, {"prompt": "x", "model": "claude-opus-5-5"},
                 {"prompt": "x", "model": "claude-sonnet-4-5-20250929"},
                 {"prompt": "x", "subagent_type": "fork"},
                 {"prompt": "x", "subagent_type": "claude-code-guide"},
                 {"prompt": "x", "subagent_type": "statusline-setup"}]
        for ti in deny:
            self.assertIsNotNone(ag.check(ti), ti)
        for ti in allow:
            self.assertIsNone(ag.check(ti), ti)

    def test_main(self):
        def hook(payload):
            return subprocess.run([sys.executable, str(HOOKS / "agent_guard.py")], input=payload,
                                  capture_output=True, text=True, timeout=10)
        r = hook(json.dumps({"tool_name": "Agent", "tool_input": {"prompt": "x", "subagent_type": "general-purpose"}}))
        self.assertEqual(r.returncode, 2)
        self.assertIn('model: "opus"|"sonnet"', r.stderr)
        r = hook(json.dumps({"tool_name": "Agent", "tool_input": {"prompt": "x", "model": "opus"}}))
        self.assertEqual(r.returncode, 0)
        self.assertEqual(hook("{broken").returncode, 0)


if __name__ == "__main__":
    unittest.main()
