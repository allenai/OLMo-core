import os
import runpy
from pathlib import Path

os.environ.setdefault("OLMOE3_TESTRUN_VARIANT", "testrun5-ep2-pp2-fp8-norecompute")
os.environ.setdefault("OLMOE3_TESTRUN_MAX_DURATION", str(int(10e9)))
os.environ.setdefault("OLMOE3_TESTRUN_EP_DIM", "2")
os.environ.setdefault("OLMOE3_TESTRUN_PP_DIM", "2")
os.environ.setdefault("OLMOE3_TESTRUN_USE_FP8", "1")
os.environ.setdefault("OLMOE3_TESTRUN_PER_LAYER_RECOMPUTE", "0")
os.environ.setdefault("OLMOE3_TESTRUN_MICRO_BSZ", "2")

runpy.run_path(
    str(Path(__file__).with_name("OLMoE3-dev-260623-testrun.py")),
    run_name="__main__",
)
