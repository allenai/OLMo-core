#!/usr/bin/env python3
"""
Print the Triton kernel emitted by torch.compile for
olmo_core.nn.functional.cross_entropy_loss.cross_entropy_with_z_loss
"""

import os
from pathlib import Path

import torch

# ----------------------------------------------------------------------
# 1.  Enable Inductor tracing so the generated code is written to disk
# ----------------------------------------------------------------------
os.environ["TORCH_COMPILE_DEBUG"] = "1"  # master switch
TRACE_DIR = Path("./inductor_trace")  # pick any writable dir
os.environ["TORCHINDUCTOR_TRACE_DIR"] = str(TRACE_DIR)  # where dumps will live
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"  # disable caching so we can see the kernel

from torch._inductor import config as inductor_cfg  # noqa: E402

inductor_cfg.trace.enabled = True  # redundant, but explicit
inductor_cfg.trace.debug_dir = str(TRACE_DIR)
inductor_cfg.trace.output_code = True  # write output_code.py

# ----------------------------------------------------------------------
# 2.  Import the function we want to compile
# ----------------------------------------------------------------------
from olmo_core.nn.functional.cross_entropy_loss import (  # noqa: F401, E402
    cross_entropy_loss,
    cross_entropy_with_z_loss,
)

# ----------------------------------------------------------------------
# 3.  Dummy inputs
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
N, V = 32, 50257  # batch   & vocab size
logits = torch.randn(N, V, device=device, dtype=torch.float16, requires_grad=False)
labels = torch.randint(0, V, (N,), device=device, dtype=torch.int64)

# ----------------------------------------------------------------------
# 4.  Compile & run once (this triggers kernel generation)
# ----------------------------------------------------------------------
compiled_fn = torch.compile(cross_entropy_loss, fullgraph=True, backend="inductor")

_ = compiled_fn(logits, labels, reduction="sum", compute_z_loss=True)  # forward
torch.cuda.synchronize(device)


# ----------------------------------------------------------------------
# 5.  Find the Triton kernel and FX graph and print it
# ----------------------------------------------------------------------
def first_file(root: Path, pattern: str):
    for f in root.glob(pattern):
        return f
    return None


kernel_file = first_file(TRACE_DIR, "**/output_code.py")
fx_graph_file = first_file(TRACE_DIR, "**/fx_graph_transformed.py") or first_file(
    TRACE_DIR, "**/fx_graph_runnable.py"
)

if kernel_file is None:
    raise RuntimeError("No Triton kernel found – compilation may have fallen back to CPU.")


print(f"\n=== Triton kernel ({kernel_file}) ===\n")
print(kernel_file.read_text())

if fx_graph_file:
    print(f"\n=== FX graph ({fx_graph_file}) ===\n")
    print(fx_graph_file.read_text())
else:
    print("\n(No FX graph file found!)")
