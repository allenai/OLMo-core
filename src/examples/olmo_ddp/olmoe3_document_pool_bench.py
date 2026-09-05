"""One-GPU document-pool correctness and latency screening; no model integration."""

import json
import os
import statistics
from pathlib import Path

import torch
from torch._inductor.utils import run_and_get_code

from olmo_core.ops.emo_document_pool import document_pool_keep_mask
from olmo_core.ops.moe import doc_sum_scatter, pool_keep_mask_inverse_scatter


def reference(scores, segments, pools):
    """Current fastest reference, including scatter, broadcast and inverse-scatter sort."""
    return pool_keep_mask_inverse_scatter(doc_sum_scatter(scores, segments), pools)


def timing(function, args):
    """Bracket candidate with the compiled reference; exclude setup/checking/compilation."""
    for _ in range(10):
        function(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(30):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        function(*args)
        end.record()
        samples.append((start, end))
    torch.cuda.synchronize()
    values = [start.elapsed_time(end) for start, end in samples]
    return {"median_ms": statistics.median(values), "mean_ms": statistics.mean(values)}


def main():
    """Test real MB4 shape across document lengths; random/tied/edge pools included."""
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(310)
    destination = Path(os.environ.get("RESULTS_DIR", "/results")) / "document-pool"
    destination.mkdir(parents=True, exist_ok=True)
    old = torch.compile(reference, fullgraph=True, dynamic=False)
    new = torch.compile(document_pool_keep_mask, fullgraph=True, dynamic=False)
    summary = {"source_commit": os.environ.get("GIT_REF"), "torch": torch.__version__, "cases": []}
    for length in (1, 16, 128, 1024, 8192):
        segments = (torch.arange(8192, device="cuda") // length).expand(4, -1).contiguous()
        raw_pools = torch.randint(16, 513, (4, 8192), device="cuda")
        raw_pools[0, ::3] = 512
        raw_pools[1, ::3] = 16
        pools = raw_pools.gather(1, segments)
        for tied in (False, True):
            scores = (
                torch.randint(-2, 3, (4, 8192, 512), device="cuda").float()
                if tied
                else torch.rand(4, 8192, 512, device="cuda")
            )
            args = (scores, segments, pools)
            expected = old(*args)
            actual, code = run_and_get_code(lambda: new(*args))
            mismatch = int((actual != expected).sum())
            row = {"document_length": length, "tied": tied, "mismatches": mismatch}
            print("DOCUMENT_POOL_PARITY", json.dumps(row), flush=True)
            if length == 1 and not tied:
                for index, source in enumerate(code):
                    (destination / f"candidate-generated-{index}.txt").write_text(source)
            assert mismatch == 0, row
            assert torch.equal(actual.sum(-1), pools)
            if not tied:
                row["reference_before"] = timing(old, args)
                row["candidate"] = timing(new, args)
                row["reference_after"] = timing(old, args)
            summary["cases"].append(row)
            (destination / "summary.json").write_text(json.dumps(summary, indent=2))
            print("DOCUMENT_POOL_RESULT", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
