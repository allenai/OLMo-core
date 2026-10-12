"""Env probe for the chunked-vllm validation gate (run inside horton allocation)."""

import os
import sys

print("python:", sys.executable)
try:
    import transformers

    print("transformers:", transformers.__version__, transformers.__file__)
    print("has Qwen3_5ForCausalLM:", hasattr(transformers, "Qwen3_5ForCausalLM"))
except Exception as e:  # noqa: BLE001
    print("transformers import failed:", e)
try:
    import vllm

    print("vllm:", vllm.__version__)
except Exception as e:  # noqa: BLE001
    print("vllm import failed:", e)
try:
    import olmo_core

    print("olmo_core:", olmo_core.__file__)
except Exception as e:  # noqa: BLE001
    print("olmo_core import failed:", e)
print("tf_qwen35_target exists:", os.path.isdir("/scratch/users/prasann/tf_qwen35_target"))
if os.path.isdir("/scratch/users/prasann/tf_qwen35_target"):
    print(sorted(os.listdir("/scratch/users/prasann/tf_qwen35_target"))[:10])
