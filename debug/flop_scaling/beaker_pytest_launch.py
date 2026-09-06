"""Run a pytest file on Beaker (1 GPU, jupiter, training image) -- for GPU tests when the local
slurm pool is saturated. Runs the PUSHED commit (gantry clones it; allow_dirty ships nothing).

    python debug/flop_scaling/beaker_pytest_launch.py src/test/nn/block_skip_gdn_test.py [-k expr]
"""
import argparse
from datetime import datetime

from olmo_core.internal.common import build_launch_config
from olmo_core.launch.beaker import OLMoCoreBeakerImage

ap = argparse.ArgumentParser()
ap.add_argument("tests", nargs="+")
ap.add_argument("-k", default=None)
ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
ap.add_argument("--priority", default="urgent")
a = ap.parse_args()
name = f"pytest-{a.tests[0].split('/')[-1].replace('.py', '').replace('_', '-')}-{datetime.now().strftime('%m%d%H%M')}"
cmd = ["python", "-m", "pytest", "-x", "-v", "-p", "no:cacheprovider", *a.tests] + (["-k", a.k] if a.k else [])
lc = build_launch_config(name=name, cmd=cmd, cluster=a.cluster, beaker_image=OLMoCoreBeakerImage.stable,
                         workspace="ai2/flex2", budget="ai2/oe-other", num_nodes=1, num_gpus=1)
lc.priority = a.priority
lc.allow_dirty = True  # the tree carries unrelated local edits; gantry clones the PUSHED commit
lc.step_timeout = None
lc.step_soft_timeout = None
wl = lc.launch(follow=False)
wl_id = getattr(wl, "id", None) or getattr(getattr(wl, "experiment", None), "id", None)
print(f"SUBMITTED id={wl_id} name={name}")
