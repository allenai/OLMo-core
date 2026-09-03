"""Run bench_ffn_speed.py on Beaker through the SAME launch path/image as the training jobs
(OLMoCoreBeakerImage.stable, pip install -e ., flash/fla kernels as trained), 1 GPU.
    python debug/flop_scaling/beaker_bench_launch.py --cluster ai2/jupiter-cirrascale-2 [--extra "--models q35-4B"]"""
import argparse, sys
from datetime import datetime
from olmo_core.internal.common import build_launch_config
from olmo_core.launch.beaker import OLMoCoreBeakerImage

ap = argparse.ArgumentParser()
ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
ap.add_argument("--extra", default="")
ap.add_argument("--priority", default="urgent")
a = ap.parse_args()
name = f"fs35-ffnspeed-olmo-{datetime.now().strftime('%m%d%H%M')}"
cmd = ["python", "debug/flop_scaling/bench_ffn_speed.py", "--out", "/results/ffn_speed.json"] + (a.extra.split() if a.extra else [])
lc = build_launch_config(name=name, cmd=cmd, cluster=a.cluster, beaker_image=OLMoCoreBeakerImage.stable,
                         workspace="ai2/flex2", budget="ai2/oe-other", num_nodes=1, num_gpus=1)
lc.priority = a.priority; lc.allow_dirty = True
lc.step_timeout = None; lc.step_soft_timeout = None
wl = lc.launch(follow=False)
wl_id = getattr(wl, "id", None) or getattr(getattr(wl, "experiment", None), "id", None)
print(f"SUBMITTED id={wl_id} name={name}")
