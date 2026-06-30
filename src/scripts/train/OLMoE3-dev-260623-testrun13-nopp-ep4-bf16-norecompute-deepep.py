import os
import runpy
from pathlib import Path

os.environ.setdefault("OLMOE3_TESTRUN_VARIANT", "testrun13-nopp-ep4-bf16-norecompute-deepep")
os.environ.setdefault("OLMOE3_TESTRUN_MAX_DURATION", str(int(10e9)))
os.environ.setdefault("OLMOE3_TESTRUN_EP_DIM", "4")
os.environ.setdefault("OLMOE3_TESTRUN_PP_DIM", "1")
os.environ.setdefault("OLMOE3_TESTRUN_EP_BACKEND", "deepep_v2")
os.environ.setdefault("OLMOE3_TESTRUN_DEEPEP_PATH", "/workspace/DeepEP")
os.environ.setdefault("OLMOE3_TESTRUN_EP_CAPACITY_FACTOR", "1.25")
os.environ.setdefault("EP_REUSE_NCCL_COMM", "0")
os.environ.setdefault("OLMOE3_TESTRUN_USE_FP8", "0")
os.environ.setdefault("OLMOE3_TESTRUN_PER_LAYER_RECOMPUTE", "0")
os.environ.setdefault("OLMOE3_TESTRUN_MICRO_BSZ", "2")

runpy.run_path(
    str(Path(__file__).with_name("OLMoE3-dev-260623-testrun.py")),
    run_name="__main__",
)
