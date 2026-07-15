import os
import runpy
from pathlib import Path


os.environ.setdefault(
    "OLMOE3_TESTRUN_VARIANT",
    "testrun16-nopp-ep32-fp8-recompute-deepep",
)
os.environ.setdefault("OLMOE3_TESTRUN_MAX_DURATION", "8192")
os.environ.setdefault("OLMOE3_TESTRUN_EP_DIM", "32")
os.environ.setdefault("OLMOE3_TESTRUN_PP_DIM", "1")
os.environ.setdefault("OLMOE3_TESTRUN_EP_BACKEND", "deepep_v2")
os.environ.setdefault("OLMOE3_TESTRUN_DEEPEP_PATH", "/workspace/DeepEP")
os.environ.setdefault("OLMOE3_TESTRUN_EP_CAPACITY_FACTOR", "1.25")
os.environ.setdefault("EP_REUSE_NCCL_COMM", "1")
os.environ.setdefault("OLMOE3_TESTRUN_USE_FP8", "1")
os.environ.setdefault("OLMOE3_TESTRUN_PER_LAYER_RECOMPUTE", "1")
os.environ.setdefault("OLMOE3_TESTRUN_USE_COMPILE", "1")
os.environ.setdefault("OLMOE3_TESTRUN_NUM_LAYERS", "2")
os.environ.setdefault("OLMOE3_TESTRUN_SEQUENCE_LENGTH", "256")
os.environ.setdefault("OLMOE3_TESTRUN_MICRO_BSZ", "1")
os.environ.setdefault("OLMOE3_TESTRUN_GLOBAL_BATCH_SIZE_SEQ", "32")
os.environ.setdefault("OLMOE3_TESTRUN_DATA_MIX", "OLMoE-mix-0824-dev")
os.environ.setdefault("OLMOE3_TESTRUN_DATA_NUM_WORKERS", "0")
os.environ.setdefault("OLMOE3_TESTRUN_PRODUCTION_RUN", "0")

runpy.run_path(
    str(Path(__file__).with_name("OLMoE3-dev-260623-testrun.py")),
    run_name="__main__",
)
