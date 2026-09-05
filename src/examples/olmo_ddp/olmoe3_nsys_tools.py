"""Pinned standalone Nsight installation and read-only trace validation helpers."""

import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

NSYS_VERSION = "2026.4.1"
NSYS_PACKAGE = "nsight-systems-2026.4.1_2026.4.1.191-1_amd64.deb"
NSYS_SHA256 = "8aeaf8c73401ccafb0b9bbe59981a6fcc97a038388462b15ef48ff75458aba19"
NSYS_URL = f"https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/{NSYS_PACKAGE}"
OLD_NSYS = Path("/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/nsys")


@dataclass(frozen=True)
class NsysSettings:
    """One shared selection for wrapper injection, callbacks, provenance, and collection."""

    version: str
    ranks: tuple[int, ...]
    trace: str
    start: int
    end: int

    @classmethod
    def from_env(cls, environ=None, world_size=64):
        """Keep historical defaults unless an explicit qualified setup is requested."""
        env = os.environ if environ is None else environ
        value = env.get("OLMOE3_NSYS_RANKS", "all")
        ranks = tuple(range(world_size)) if value == "all" else tuple(map(int, value.split(",")))
        if (
            not ranks
            or len(ranks) != len(set(ranks))
            or any(r < 0 or r >= world_size for r in ranks)
        ):
            raise ValueError(f"Invalid Nsight ranks for {world_size} workers: {ranks}")
        version = env.get("OLMOE3_NSYS_VERSION", "installed")
        if version not in ("installed", NSYS_VERSION):
            raise ValueError(f"Unqualified Nsight version: {version}")
        start = int(env.get("OLMOE3_NSYS_START", "71"))
        end = int(env.get("OLMOE3_NSYS_END", "73"))
        if not 31 < start <= end:
            raise ValueError(f"Capture must follow timing warmup: {start}:{end}")
        trace = env.get("OLMOE3_NSYS_TRACE", "cuda,nvtx,osrt")
        if trace not in ("cuda,nvtx", "cuda,nvtx,osrt", "cuda-sw,nvtx", "cuda-sw,nvtx,osrt"):
            raise ValueError(f"Unexpected Nsight trace selection: {trace}")
        return cls(version, ranks, trace, start, end)

    def clean_windows(self, steps):
        """Exclude capture and a post-capture settling period from throughput."""
        if steps < self.end:
            raise ValueError(f"Run ends before capture: {steps} < {self.end}")
        return [[a, b] for a, b in ((31, self.start - 1), (self.end + 8, steps)) if a <= b]


def install_nsys():
    """Extract a verified NVIDIA package privately; never change the CUDA/driver stack."""
    root = Path(tempfile.mkdtemp(prefix="olmoe3-nsys-"))
    package = root / NSYS_PACKAGE
    subprocess.run(["curl", "-fL", "--retry", "3", NSYS_URL, "-o", str(package)], check=True)
    with package.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != NSYS_SHA256:
        raise RuntimeError(f"Unexpected Nsight package checksum: {digest}")
    subprocess.run(["dpkg-deb", "--extract", str(package), str(root / "extracted")], check=True)
    candidates = list(
        (root / "extracted/opt/nvidia").glob(f"nsight-systems/{NSYS_VERSION}/bin/nsys")
    )
    if not candidates:
        candidates = list((root / "extracted/opt/nvidia").glob("nsight-systems*/bin/nsys"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one standalone Nsight CLI, got {candidates}")
    binary = candidates[0]
    version = subprocess.check_output([str(binary), "--version"], text=True).strip()
    if NSYS_VERSION not in version:
        raise RuntimeError(f"Wrong profiler version: {version}")
    print(
        "STANDALONE_NSYS",
        json.dumps({"binary": str(binary), "version": version, "sha256": digest}),
        flush=True,
    )
    return binary


def summarize_sqlite(path):
    """Require real CUDA activity, not just a successful command or empty report file."""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        counts = {}
        for name in (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "CUPTI_ACTIVITY_KIND_DRIVER",
            "NVTX_EVENTS",
        ):
            counts[name] = (
                conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                if name in tables
                else 0
            )
        kernel_tables = [
            name
            for name in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
            if name in tables
        ]
        kernels = []
        for table in kernel_tables:
            columns = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
            column = next((c for c in ("demangledName", "shortName", "name") if c in columns), None)
            if column and "StringIds" in tables:
                kernels.extend(
                    {"name": row[0], "count": row[1], "duration_ms_sum": row[2]}
                    for row in conn.execute(
                        f'SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6 FROM "{table}" k '
                        f'JOIN StringIds s ON k."{column}"=s.id GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC'
                    )
                )
        return {
            "tables": sorted(tables),
            "counts": counts,
            "kernel_count": sum(counts[name] for name in kernel_tables),
            "memcpy_count": counts["CUPTI_ACTIVITY_KIND_MEMCPY"],
            "nccl_kernel_count": sum(k["count"] for k in kernels if "nccl" in k["name"].lower()),
            "top_kernels_nonadditive": sorted(kernels, key=lambda k: -k["duration_ms_sum"])[:20],
        }


def validate_report(binary, report):
    """Export and validate a newly created report, retaining raw files beside it."""
    database = report.with_suffix(".sqlite")
    subprocess.run(
        [str(binary), "export", "--type=sqlite", "--output", str(database), str(report)],
        check=True,
        timeout=120,
    )
    summary = summarize_sqlite(database)
    summary["valid_cuda_trace"] = summary["kernel_count"] > 0 and summary["memcpy_count"] > 0
    return summary
