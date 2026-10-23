"""Run pytest on Beaker through the shared 1-GPU bench launcher (``--script`` gets ``--out``).

    python research/ffn_compute/flop_scaling/beaker_bench_launch.py --script research/token_routing/online_router/run_pytest.py \
        --extra "src/test/nn/online_token_router_test.py"
"""
import json
import subprocess
import sys

args = sys.argv[1:]
out = args[args.index("--out") + 1]
tests = [a for i, a in enumerate(args) if a != "--out" and (i == 0 or args[i - 1] != "--out")]
p = subprocess.run([sys.executable, "-m", "pytest", "-v", "-rs", *tests], capture_output=True, text=True)
print(p.stdout[-20000:], p.stderr[-5000:], flush=True)
json.dump({"returncode": p.returncode, "tail": p.stdout[-4000:]}, open(out, "w"))
sys.exit(p.returncode)
