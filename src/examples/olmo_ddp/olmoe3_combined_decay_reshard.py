"""Audit a 128 -> 64 DP restore against both old shards' saved sample hashes."""

import hashlib
import json
from pathlib import Path

import torch


def sample_hash(tensor):
    """Use the original hero audit's exact evenly spaced byte-sampling rule."""
    tensor = tensor.detach().reshape(-1)
    count = min(128, tensor.numel())
    if count == 0:
        raise ValueError("Cannot audit an empty optimizer shard")
    indices = [i * (tensor.numel() - 1) // max(1, count - 1) for i in range(count)]
    data = tensor[torch.tensor(indices, device=tensor.device)].contiguous()
    return [
        tensor.numel(),
        str(tensor.dtype),
        hashlib.sha256(data.view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
    ]


def verify_halved_dp(trainer, path, actual):
    """Check replicated state and each old half of every enlarged optimizer shard."""
    rank = actual["rank"]
    assert actual["gpus"] == 64 and 0 <= rank < 64
    records = []
    for half in (0, 1):
        old_rank = 2 * rank + half
        saved = json.loads((Path(path) / "resume_audit" / f"rank{old_rank}.json").read_text())
        assert saved["gpus"] == 128 and saved["rank"] == old_rank
        for key in ("step", "tokens", "loss_history", "norm_history"):
            assert saved[key] == actual[key], key
        assert saved["tensors"].keys() == actual["tensors"].keys()
        for name, expected in saved["tensors"].items():
            observed = actual["tensors"][name]
            if expected[0] == observed[0]:
                assert expected == observed, name
                continue
            tensor = trainer.train_module.optim.states[name]
            assert len(tensor.placements) == 1 and tensor.placements[0].is_shard(0), name
            local = tensor.to_local().detach().reshape(-1)
            assert local.numel() == 2 * expected[0], name
            width = expected[0]
            assert sample_hash(local[half * width : (half + 1) * width]) == expected, name
            records.append((old_rank, name))
    assert records, "Expected optimizer resharding, but no sharded tensors were checked"
    return dict(
        source_gpus=128,
        gpus=64,
        source_ranks=[2 * rank, 2 * rank + 1],
        old_optimizer_shards_verified=len(records),
        rng_policy="native-reseed-on-world-size-change",
    )
