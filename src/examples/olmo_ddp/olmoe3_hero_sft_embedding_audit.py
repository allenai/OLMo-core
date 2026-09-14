"""Read only embedding weights/moments from the LC and completed SFT smoke saves."""

import torch
import torch.distributed.checkpoint as dcp
from olmoe3_hero_sft_plan import MOUNT, runs
from olmoe3_lr_sweep_watch import log

from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader


def load_embedding(root, suffix):
    paths = list(root.glob("*/.metadata"))
    assert len(paths) == 1, paths
    reader = RemoteFileSystemReader(str(paths[0].parent))
    metadata = reader.read_metadata()
    keys = [
        key for key in metadata.state_dict_metadata if key.endswith("embeddings.weight." + suffix)
    ]
    assert len(keys) == 1, keys
    key = keys[0]
    meta = metadata.state_dict_metadata[key]
    state = {key: torch.empty(meta.size, dtype=meta.properties.dtype)}
    dcp.load(state, storage_reader=reader)
    return state[key].float()


if __name__ == "__main__":
    assert MOUNT.is_mount()
    torch.set_num_threads(8)
    for run in runs(True):
        original = load_embedding(run.source, "main")
        moments = load_embedding(run.root / "step2", "exp_avg")
        updated = load_embedding(run.root / "step2", "main")
        row_norm = moments.norm(dim=1)
        ids = row_norm.topk(20).indices.tolist()
        selected = sorted(set(ids + [0, 100257, 100264, 100265, 100277]))
        log(
            "SFT_EMBEDDING_AUDIT",
            arm=run.arm,
            total_moment_norm=float(moments.norm()),
            top_row_ids=ids,
            rows=[
                {
                    "id": idx,
                    "source_weight_norm": float(original[idx].norm()),
                    "updated_weight_norm": float(updated[idx].norm()),
                    "moment_norm": float(row_norm[idx]),
                }
                for idx in selected
            ],
        )
