import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from olmo_core.testing import run_distributed_test


def fingerprint(tensor):
    flat = tensor.flatten()
    indices = [
        i * (flat.numel() - 1) // max(1, min(128, flat.numel()) - 1)
        for i in range(min(128, flat.numel()))
    ]
    return [
        flat.numel(),
        str(flat.dtype),
        hashlib.sha256(flat[indices].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
    ]


def check_exchange():
    import olmoe3_hero_reshard_audit as audit

    rank = dist.get_rank()
    audit.dist = SimpleNamespace(
        get_rank=lambda: rank,
        get_world_size=lambda: 128,
        batch_isend_irecv=dist.batch_isend_irecv,
        P2POp=dist.P2POp,
        isend=dist.isend,
        irecv=dist.irecv,
    )
    original = torch.arange(600, dtype=torch.float32) - 250
    original[2] = -0.0
    local = original.chunk(2)[rank]
    state = SimpleNamespace(
        placements=[SimpleNamespace(is_shard=lambda dim: dim == 0)], to_local=lambda: local
    )
    saved = dict(
        rank=0,
        gpus=64,
        step=300411,
        tokens=5040060235776,
        loss_history=[1.0],
        norm_history=[2.0],
        tensors={"weight.main": fingerprint(original)},
    )
    actual = {**saved, "gpus": 128, "rank": rank, "tensors": {"weight.main": fingerprint(local)}}
    trainer = SimpleNamespace(
        train_module=SimpleNamespace(optim=SimpleNamespace(states={"weight.main": state}))
    )
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        (path / "resume_audit").mkdir()
        (path / "resume_audit/rank0.json").write_text(json.dumps(saved))
        proof = audit.verify_doubled_dp(trainer, path, actual)
        assert proof["resharded_optimizer_tensors_verified"] == 1


def test_resharded_optimizer_sample_exchange():
    run_distributed_test(check_exchange, backend="gloo", start_method="spawn")
