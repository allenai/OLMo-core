"""Independent source checks for legacy DCP checkpoints without hero resume audits."""

import math

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader


def verify_source_weights(trainer, source):
    """Check every parameter's saved extent and independently reload six small tensors."""
    # EP1 models are replicated; optimizer shards are resharded separately 12->8.
    tm = trainer.train_module
    # Core's metadata uses its own _StorageInfo, without the newer PyTorch
    # transform_descriptors field. Use the matching reader even for local files.
    reader = RemoteFileSystemReader(source / "model_and_optim", thread_count=1)
    metadata = reader.read_metadata()
    saved = metadata.state_dict_metadata
    selected = {}
    for name, parameter in tm.model.named_parameters():
        matches = [k for k in saved if k.endswith(name + ".main")]
        assert len(matches) == 1, ("missing/ambiguous native parameter", name, matches)
        (key,) = matches
        assert math.prod(saved[key].size) == parameter.numel(), (
            name,
            saved[key].size,
            parameter.shape,
        )
        if (
            len(selected) < 6
            and parameter.numel() <= 1_048_576
            and any(s in name for s in ("q_norm", "conv1d", "router"))
        ):
            selected[key] = parameter
    assert len(selected) == 6, "Expected multiple independently verifiable small tensors"
    result = [None]
    if dist.get_rank() == 0:
        try:
            state = {
                k: torch.empty(saved[k].size, dtype=saved[k].properties.dtype) for k in selected
            }
            dcp.load(state, storage_reader=reader, no_dist=True)
            for key, parameter in selected.items():
                expected = state[key].reshape(parameter.shape).to(parameter.dtype)
                assert torch.equal(expected, parameter.detach().cpu()), key
            result[0] = {"passed": True, "keys": list(selected)}
        # DCP wraps read errors in a BaseException subclass, not Exception. Send
        # the failure to the other ranks instead of abandoning their broadcast.
        except (Exception, CheckpointException) as exc:  # noqa: BLE001
            result[0] = {"passed": False, "error": repr(exc)}
    dist.broadcast_object_list(result, src=0, group=trainer.bookkeeping_pg)
    assert result[0]["passed"], result[0]
