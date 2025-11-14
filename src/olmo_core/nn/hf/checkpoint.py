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
from olmo_core.config import DType
from olmo_core.distributed.utils import barrier, get_fs_local_rank, get_full_tensor
from olmo_core.doc_utils import beta_feature
from olmo_core.io import clear_directory, copy_dir, file_exists, is_url
from olmo_core.nn.hf.config import get_hf_config
from olmo_core.nn.hf.convert import convert_state_from_hf, convert_state_to_hf
from olmo_core.nn.transformer.model import Transformer

try:
    from accelerate import init_empty_weights  # type: ignore
except ImportError:

    @contextmanager
    def init_empty_weights(include_buffers: bool = False) -> Generator[None, None, None]:
        del include_buffers
        log.warning("accelerate not installed, will initialize weights.")
        yield None


log = logging.getLogger(__name__)


@beta_feature
def load_hf_model(
    model_name_or_path: PathOrStr,
    model_state_dict: Dict[str, Any],
    *,
    revision: str = "main",
    model_id: Optional[str] = None,
    num_embeddings: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    work_dir: Optional[PathOrStr] = None,
):
    """
    Loads an OLMo Core model state dict using a model in Hugging Face transformers format.

    :param model_name_or_path: The name of a model in HF Hub or the path to a model saved in HF format.
    :param model_state_dict: The OLMo Core model state dict in which to load HF state.
    :param revision: If ``model_name_or_path`` is the id of a model in HF Hub, then this is the revision
        (branch) of that model. Defaults to "main".
    :param model_id: Deprecated, model-specific mappings are now determined by the model architecture,
        in :mod:`olmo_core.nn.hf.convert`
    :param num_embeddings: The number of embeddings in the OLMo Core model being loaded into,
        defaults to the number of embeddings in the HF model.
    :param process_group: The process group to use for distributed communication.
    :param work_dir: A local directory that can be used for holding temporary state. Required when
        downloading a model from a cloud directory.
    """
    del model_id

    work_dir = f"{work_dir}/hf-tmp" if work_dir is not None else None

    if is_url(model_name_or_path):
        log.warning(
            "Model id or path provided is a remote Hugging Face directory. This may not be suitable for unshared file systems."
        )
        assert work_dir is not None
        assert (
            file_exists(f"{model_name_or_path}/generation_config.json")
            or file_exists(f"{model_name_or_path}/model.safetensors.index.json")
            or file_exists(f"{model_name_or_path}/pytorch_model.bin")
        )

        # Download model to local FS
        if get_fs_local_rank() == 0:
            copy_dir(model_name_or_path, work_dir)
        barrier(group=process_group)
    elif Path(model_name_or_path).is_dir():
        assert (
            file_exists(f"{model_name_or_path}/generation_config.json")
            or file_exists(f"{model_name_or_path}/model.safetensors.index.json")
            or file_exists(f"{model_name_or_path}/pytorch_model.bin")
        )
    elif repo_exists(str(model_name_or_path)):
        log.warning(
            "Model id or path provided is a Hugging Face model id. This may not be suitable for unshared file systems."
        )
    else:
        raise NotImplementedError

    # Warm up the HF local cache by downloading the model on just local rank 0
    if get_fs_local_rank() == 0:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, revision=revision)
        del hf_model
    barrier(group=process_group)

    hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, revision=revision)
    log.info(f"Loaded hf model: {hf_model}")
    hf_model.resize_token_embeddings(num_embeddings)

    converted_state_dict: Dict[str, torch.Tensor] = convert_state_from_hf(
        hf_model.config,
        hf_model.state_dict(),
        model_type=getattr(hf_model.config, "model_type", None),
    )

    for key in sorted(converted_state_dict.keys()):
        state = converted_state_dict[key]
        olmo_core_state = model_state_dict[key]
        if isinstance(olmo_core_state, DTensor):
            olmo_core_state = distribute_tensor(
                state, olmo_core_state.device_mesh, olmo_core_state.placements
            )
        else:
            olmo_core_state = state

        model_state_dict[key] = olmo_core_state

    if work_dir:
        clear_directory(work_dir)


@beta_feature
def save_hf_model(
    save_dir: PathOrStr,
    model_state_dict: Dict[str, Any],
    model: Transformer,
    *,
    dtype: Optional[DType] = None,
    vocab_size: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    work_dir: Optional[PathOrStr] = None,
    save_overwrite: bool = False,
):
    """
    Saves an OLMo Core model state dict in Hugging Face transformers format.

    :param save_dir: Directory in which to save model.
    :param model_state_dict: The OLMo Core model state dict being saved in HF format.
    :param dtype: The torch dtype that model weights should be saved as.
    :param vocab_size: The size of the vocab, defaults to the number of embeddings in the OLMo Core model.
    :param process_group: The process group to use for distributed communication.
    :param work_dir: A local directory that can be used for holding temporary state. Required when
        downloading a model from a cloud directory.
    :param save_overwrite: Overwrite existing files in ``save_dir``.
    """

    hf_config = get_hf_config(model)

    model_state_dict = {key: get_full_tensor(state) for key, state in model_state_dict.items()}
    if dtype is not None:
        model_state_dict = {
            key: state.to(dtype=dtype.as_pt()) for key, state in model_state_dict.items()
        }

    hf_state_dict: Dict[str, torch.Tensor] = convert_state_to_hf(hf_config, model_state_dict)

    # model.save_pretrained fails says `tensor.reshape()` should be used instead of `tensor.view()`
    # if we do not make the state contiguous. Unfortunately this is bad for perf.
    hf_state_dict = {key: state.contiguous() for key, state in hf_state_dict.items()}

    with init_empty_weights():
        log.info("Initializing HF model with empty weights...")
        hf_model = AutoModelForCausalLM.from_config(hf_config)
        del hf_config

    hf_model.load_state_dict(hf_state_dict, assign=True)

    hf_model.config.vocab_size = vocab_size or model.vocab_size
    hf_model.resize_token_embeddings(hf_model.config.vocab_size)

    if get_fs_local_rank(process_group) == 0:
        if is_url(save_dir):
            assert work_dir is not None
            hf_model.save_pretrained(work_dir)

            copy_dir(work_dir, save_dir, save_overwrite=save_overwrite)
        else:
            target = Path(save_dir)
            if target.is_dir() and not save_overwrite:
                raise FileExistsError(target)
            target.parent.mkdir(exist_ok=True, parents=True)
            hf_model.save_pretrained(target)
