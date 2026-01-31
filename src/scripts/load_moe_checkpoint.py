import contextlib
import logging
from dataclasses import replace
from functools import cached_property
from typing import Any, Dict, Generator, Optional, Tuple, Union, Iterable, Sequence

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
import torch.distributed.checkpoint as dist_cp
from olmo_core.distributed.checkpoint import _prepare_env_for_save, RemoteFileSystemWriter, RemoteFileSystemReader
from olmo_core.aliases import PathOrStr


def load_state_dict_direct(
    dir: PathOrStr,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
    thread_count: Optional[int] = None,
):
    from olmo_core.io import normalize_path

    # sd_to_load = self.optim.state_dict()

    dir = normalize_path(dir)
    reader = RemoteFileSystemReader(
        dir, 
        thread_count=thread_count, 
        pre_download=pre_download, work_dir=work_dir
    )

    metadata = reader.read_metadata()
    # example: 'module.blocks.0.attention.w_q.weight.main'
    model_sd_meta = {k: v for k, v in metadata.state_dict_metadata.items() if k.endswith('main')}
    sd_to_load = {}
    for k in model_sd_meta.keys():
        sd_to_load[k] = torch.empty(model_sd_meta[k].size, dtype=torch.float32)

    dist_cp.state_dict_loader.load(
        sd_to_load,
        checkpoint_id=dir,
        storage_reader=reader,
        process_group=process_group,
        # planner=FlatLoadPlanner(),
    )

    return sd_to_load


if __name__ == "__main__":
    main_sd = load_state_dict_direct(
        dir='/workspace/checkpoint/OLMoE3-abl-260102-018a_1024d1024a_12L768M768S_32E4K1S_abl/step10000/model_and_optim',
        process_group=None, pre_download=True, work_dir='/workspace/tmp'
    )

    torch.save(main_sd, '/workspace/tmp/step10000_model_main.pt')
    