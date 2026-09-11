"""Fail-closed verification of the running hero's numerical kernel environment."""

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

EXPECTED = {
    "torch": "2.11.0+cu130",
    "flash-linear-attention": "0.5.2",
    "fla-core": "0.5.2",
    "kernel-fun": "0.2.0.dev0",
    "nvidia-cutlass-dsl": "4.5.3",
    "triton": "3.6.0",
    "nvidia-nccl-cu13": "2.28.9",
    "flash-attn": "2.8.2",
    "flash-attn-4": "4.0.0b16",
    "transformer-engine": "2.9.0",
    "transformer-engine-torch": "2.9.0",
    "transformer-engine-cu13": "2.9.0",
    "torchao": "0.15.0",
    "cuda-python": "13.3.1",
    "cuda-bindings": "13.3.1",
    "grouped-gemm": "0.3.0",
}
FILES = {
    (
        "fla-core",
        "fla/ops/kda/gate.py",
    ): "04fc6aa1857eb216ddecc354480285014c25dded74fa646491e9c17e7e11ef35",
    (
        "kernel-fun",
        "kernel_fun/kda/chain.py",
    ): "50bf76c2486792bf262fd36a0844c4ad8818d16b8b6a2e78b9d4fd3d6e43e052",
    (
        "kernel-fun",
        "kernel_fun/kda/autograd.py",
    ): "970efa1da05429f4ecc635ad8361b9176e0dc91790cd59c71aa05b494d70bfbc",
}


def verify_runtime():
    """Check versions, source hashes, and the exact import that failed before any GPU work."""
    actual = {name: metadata.version(name) for name in EXPECTED}
    assert actual == EXPECTED, {k: (actual[k], v) for k, v in EXPECTED.items() if actual[k] != v}
    direct = json.loads(metadata.distribution("kernel-fun").read_text("direct_url.json"))
    assert direct["vcs_info"]["commit_id"] == "7a6983baf2beb4ec4d7fe914ec9f6670438af99b"
    for (package, relative), digest in FILES.items():
        path = Path(metadata.distribution(package).locate_file(relative))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, str(path)
    import torch
    from fla.ops.kda.gate import kda_gate_chunk_cumsum

    assert callable(kda_gate_chunk_cumsum)
    assert torch.version.cuda == "13.0"
    print("DECAY_HERO_RUNTIME_VERIFIED", json.dumps(actual, sort_keys=True), flush=True)


if __name__ == "__main__":
    verify_runtime()
