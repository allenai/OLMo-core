import os
import runpy
from pathlib import Path

os.environ.setdefault("OLMOE3_TESTRUN_VARIANT", "testrun13-nopp-ep4-tma-ibgda-bf16-norecompute")
os.environ.setdefault("OLMOE3_TESTRUN_MAX_DURATION", str(int(10e9)))
os.environ.setdefault("OLMOE3_TESTRUN_NUM_LAYERS", "4")
os.environ.setdefault("OLMOE3_TESTRUN_EP_DIM", "4")
os.environ.setdefault("OLMOE3_TESTRUN_PP_DIM", "1")
os.environ.setdefault("OLMOE3_TESTRUN_ROWWISE_BACKEND", "tma_ibgda")
os.environ.setdefault("OLMOE3_TESTRUN_USE_COMPILE", "1")
os.environ.setdefault("OLMOE3_TESTRUN_USE_FP8", "0")
os.environ.setdefault("OLMOE3_TESTRUN_PER_LAYER_RECOMPUTE", "0")
os.environ.setdefault("OLMOE3_TESTRUN_PRODUCTION_RUN", "0")
os.environ.setdefault("OLMOE3_TESTRUN_MICRO_BSZ", "2")
# os.environ.setdefault("OLMOE3_TESTRUN_DATA_MIX", "OLMoE-mix-0824-dev")
os.environ.setdefault("OLMOE3_TESTRUN_DATA_NUM_WORKERS", "0")

runpy.run_path(
    str(Path(__file__).with_name("OLMoE3-dev-260623-testrun.py")),
    run_name="__main__",
)
