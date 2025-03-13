import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import torch
import torch.distributed as dist
from huggingface_hub import repo_exists
from torch.distributed.tensor import DTensor, distribute_tensor
from transformers import AutoModelForCausalLM

from olmo_core.aliases import PathOrStr
from olmo_core.distributed.utils import barrier, get_fs_local_rank, get_full_tensor
from olmo_core.doc_utils import beta_feature
from olmo_core.io import copy_dir, file_exists, is_url, upload
from olmo_core.nn.hf.config import get_hf_config
from olmo_core.nn.hf.key_mapping import get_key_mapping_from_hf, get_key_mapping_to_hf
from olmo_core.nn.transformer.model import Transformer

try:
    from accelerate import init_empty_weights
except ImportError:

    @contextmanager
    def init_empty_weights(include_buffers: bool = False) -> Generator[None, None, None]:
        log.warning("accelerate not installed, will initialize weights.")
        yield None


log = logging.getLogger(__name__)


@beta_feature
def load_hf_model(
    dir: PathOrStr,
    model_state_dict: Dict[str, Any],
    n_layers: int,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    work_dir: Optional[PathOrStr] = None,
):
    work_dir = f"{work_dir}/hf-tmp" if work_dir is not None else None

    if is_url(dir):
        log.warning(
            "Load path provided is a remote Hugging Face directory. This may not be suitable for unshared file systems."
        )
        assert work_dir is not None
        assert (
            file_exists(f"{dir}/generation_config.json")
            or file_exists(f"{dir}/model.safetensors.index.json")
            or file_exists(f"{dir}/pytorch_model.bin")
        )
        model_name_or_path = dir
        model_id = None

        # Download model to local FS
        if get_fs_local_rank() == 0:
            copy_dir(dir, work_dir)
        barrier(group=process_group)
    elif Path(dir).is_dir():
        assert (
            file_exists(f"{dir}/generation_config.json")
            or file_exists(f"{dir}/model.safetensors.index.json")
            or file_exists(f"{dir}/pytorch_model.bin")
        )
        model_name_or_path = dir
        model_id = None
    elif repo_exists(str(dir)):
        log.warning(
            "Load path provided is a Hugging Face model id. This may not be suitable for unshared file systems."
        )
        model_name_or_path = dir
        model_id = str(model_name_or_path)
    else:
        raise NotImplementedError

    # Warm up the HF local cache by downloading the model on just local rank 0
    if get_fs_local_rank() == 0:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        del hf_model
    barrier(group=process_group)

    hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    log.info(f"Loaded hf model: {hf_model}")

    if n_layers != len(hf_model.model.layers):
        raise RuntimeError(
            f"Trying to load a HF model with {len(hf_model.model.layers)} layers into a model with {n_layers} layers."
        )

    hf_state_dict: Dict[str, Any] = hf_model.state_dict()

    key_mapping = get_key_mapping_from_hf(hf_model.config, model_id=model_id)

    unupdated_keys = set(model_state_dict.keys())

    for hf_key, hf_state in hf_state_dict.items():
        olmo_core_key = key_mapping[hf_key]
        olmo_core_state = model_state_dict[olmo_core_key]

        # Initialize DTensor state from the global HF state tensors
        if isinstance(olmo_core_state, DTensor):
            olmo_core_state = distribute_tensor(
                hf_state, olmo_core_state.device_mesh, olmo_core_state.placements
            )
        else:
            olmo_core_state = hf_state

        model_state_dict[olmo_core_key] = olmo_core_state
        unupdated_keys.remove(olmo_core_key)

    if len(unupdated_keys) > 0:
        raise RuntimeError(
            f"Some OLMo core state keys were not set when loading HF model: {unupdated_keys}"
        )


@beta_feature
def save_hf_model(
    dir: PathOrStr,
    model_state_dict: Dict[str, Any],
    model: Transformer,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    work_dir: Optional[PathOrStr] = None,
    save_overwrite: bool = False,
):
    hf_config = get_hf_config(model)
    key_mapping = get_key_mapping_to_hf(hf_config)

    hf_state_dict = {}
    for key, value in model_state_dict.items():
        # One OLMo core key can correspond to many HF keys. We assume that the value needs to
        # be split equally across dim 0 in this case.
        mapped_keys = key_mapping[key]

        if isinstance(value, torch.Tensor):
            value = get_full_tensor(value)

            values = torch.chunk(value, chunks=len(mapped_keys))
            for mapped_key, v in zip(mapped_keys, values):
                hf_state_dict[mapped_key] = v
        else:
            if len(mapped_keys) > 1:
                raise RuntimeError(f"Key {key} of non-tensor state is mapped one-to-many")
            assert len(mapped_keys) == 1

            hf_state_dict[mapped_keys[0]] = value

    with init_empty_weights():
        log.info("Initializing HF model with empty weights...")
        model = AutoModelForCausalLM.from_config(hf_config)

    model.load_state_dict(hf_state_dict, assign=True)

    if get_fs_local_rank(process_group) == 0:
        if is_url(dir):
            assert work_dir is not None
            model.save_pretrained(work_dir)

            target = f"{dir}"
            upload(work_dir, target, save_overwrite=save_overwrite)
        else:
            target = Path(dir)
            if target.is_dir() and not save_overwrite:
                raise FileExistsError(target)
            target.parent.mkdir(exist_ok=True, parents=True)
            model.save_pretrained(target)
