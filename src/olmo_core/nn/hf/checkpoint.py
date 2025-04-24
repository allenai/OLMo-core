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
from olmo_core.nn.hf.convert import convert_state_from_hf, convert_state_to_hf
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

    converted_state_dict: Dict[str, torch.Tensor] = convert_state_from_hf(
        hf_model.config, hf_model.state_dict(), model_id=model_id
    )

    for key, state in converted_state_dict.items():
        olmo_core_state = model_state_dict[key]
        if isinstance(olmo_core_state, DTensor):
            olmo_core_state = distribute_tensor(
                state, olmo_core_state.device_mesh, olmo_core_state.placements
            )
        else:
            olmo_core_state = state

        model_state_dict[key] = olmo_core_state


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
    import pdb; pdb.set_trace()

    model_state_dict = {key: get_full_tensor(state) for key, state in model_state_dict.items()}
    hf_state_dict: Dict[str, torch.Tensor] = convert_state_to_hf(hf_config, model_state_dict)

    # model.save_pretrained fails says `tensor.reshape()` should be used instead of `tensor.view()`
    # if we do not make the state contiguous. Unfortunately this is bad for perf.
    hf_state_dict = {key: state.contiguous() for key, state in hf_state_dict.items()}

    with init_empty_weights():
        log.info("Initializing HF model with empty weights...")
        model = AutoModelForCausalLM.from_config(hf_config)

    for key in hf_state_dict:
        if "mlp.gate.weight" in key:
            import pdb; pdb.set_trace()
            hf_state_dict[key] = hf_state_dict[key].unflatten(0, (4, 4096))
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
