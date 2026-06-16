from __future__ import annotations

import inspect
import os
import threading
import weakref
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union, cast

import torch

from olmo_core.distributed.utils import get_rank

if TYPE_CHECKING:
    from olmo_core.nn.ddp.block import OLMoDDPTransformerBlock


EP_NO_SYNC_SAVED_ACTIVATIONS_DEBUG_ENV_VAR = "OLMO_EP_NO_SYNC_SAVED_ACTIVATIONS_DEBUG"
_DEBUG_STATE = threading.local()


def _debug_activation_enabled() -> bool:
    return os.getenv(EP_NO_SYNC_SAVED_ACTIVATIONS_DEBUG_ENV_VAR, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_train_global_arg(key: str, default=None):
    from olmo_core.train.globals import get_global_arg

    return get_global_arg(key, default=default)


def _set_train_global_arg(key: str, value) -> None:
    from olmo_core.train.globals import set_global_arg

    set_global_arg(key, value)


@torch.compiler.disable(reason="Activation debug hooks use Python containers and stack inspection")
def record_named_saved_activation(tensor: torch.Tensor, name: str) -> None:
    recorder = getattr(_DEBUG_STATE, "record_named_saved_activation", None)
    if recorder is not None:
        recorder(tensor, name, name)


def maybe_dump_ep_no_sync_saved_activations(
    block: OLMoDDPTransformerBlock,
    x: torch.Tensor,
    *,
    loss_div_factor: Optional[Union[torch.Tensor, float]],
    forward_kwargs: Dict[str, object],
    no_sync_forward: Callable[..., torch.Tensor],
) -> Optional[torch.Tensor]:
    activation_dump_key = f"ep_no_sync_saved_activations_dumped_block_{block.block_idx}"
    if not (
        _get_train_global_arg("dry_run_done", default=False)
        and block.block_idx == 3
        and not _get_train_global_arg(activation_dump_key, default=False)
        and _debug_activation_enabled()
    ):
        return None

    saved_activations_by_storage: Dict[Tuple[str, int, int], Dict[str, object]] = {}
    saved_tensor_weakrefs: List[weakref.ReferenceType[torch.Tensor]] = []
    param_storage_keys = {
        (str(param.device), param.untyped_storage().data_ptr(), param.untyped_storage().nbytes())
        for param in block.parameters()
    }
    mem_before_gib = torch.cuda.memory_allocated() / (1024**3)

    def _saved_activation_name(tensor: torch.Tensor) -> Optional[str]:
        name = getattr(tensor, "_olmo_saved_activation_name", None)
        if isinstance(name, str) and name:
            return name
        return None

    def _format_names(record: Dict[str, object]) -> str:
        names = cast(set[str], record.get("names", set()))
        if not names:
            return " names=<unnamed>"
        return " names=" + "|".join(sorted(names))

    def _saved_tensor_source() -> str:
        for frame in inspect.stack(context=0)[2:]:
            filename = frame.filename
            if "/olmo_core/" not in filename:
                continue
            if filename.endswith("activation_debug.py"):
                continue
            try:
                rel = filename.split("/olmo_core/", 1)[1]
            except IndexError:
                rel = filename
            return f"{rel}:{frame.function}:{frame.lineno}"
        return "<unknown>"

    def _format_sources(record: Dict[str, object]) -> str:
        sources = cast(set[str], record.get("sources", set()))
        if not sources:
            return " sources=<unknown>"
        return " sources=" + "|".join(sorted(sources))

    def _record_saved_tensor(
        tensor: torch.Tensor,
        name_override: Optional[str] = None,
        source_override: Optional[str] = None,
    ) -> torch.Tensor:
        storage = tensor.untyped_storage()
        storage_nbytes = storage.nbytes()
        key = (str(tensor.device), storage.data_ptr(), storage_nbytes)
        tensor_name = name_override or _saved_activation_name(tensor)
        tensor_source = source_override or _saved_tensor_source()
        record = saved_activations_by_storage.get(key)
        if record is None:
            saved_activations_by_storage[key] = {
                "kind": "param" if key in param_storage_keys else "activation",
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "storage_nbytes": storage_nbytes,
                "save_count": 1,
                "names": {tensor_name} if tensor_name is not None else set(),
                "sources": {tensor_source},
            }
        else:
            record["save_count"] = cast(int, record["save_count"]) + 1
            if tensor_name is not None:
                record_names = record["names"]
                assert isinstance(record_names, set)
                record_names.add(tensor_name)
            record_sources = record["sources"]
            assert isinstance(record_sources, set)
            record_sources.add(tensor_source)
        saved_tensor_weakrefs.append(weakref.ref(tensor))
        return tensor

    _record_saved_tensor = torch.compiler.disable(
        reason="Activation debug hook uses Python containers and stack inspection"
    )(_record_saved_tensor)

    old_recorder = getattr(_DEBUG_STATE, "record_named_saved_activation", None)
    _DEBUG_STATE.record_named_saved_activation = _record_saved_tensor
    try:
        with torch.autograd.graph.saved_tensors_hooks(_record_saved_tensor, lambda tensor: tensor):
            out = no_sync_forward(
                x, loss_div_factor=loss_div_factor, **forward_kwargs
            )
    finally:
        if old_recorder is None:
            try:
                delattr(_DEBUG_STATE, "record_named_saved_activation")
            except AttributeError:
                pass
        else:
            _DEBUG_STATE.record_named_saved_activation = old_recorder

    mem_after_gib = torch.cuda.memory_allocated() / (1024**3)
    total_unique_storage_nbytes_pack = sum(
        cast(int, record["storage_nbytes"]) for record in saved_activations_by_storage.values()
    )
    total_param_storage_nbytes_pack = sum(
        cast(int, record["storage_nbytes"])
        for record in saved_activations_by_storage.values()
        if record["kind"] == "param"
    )
    total_activation_storage_nbytes_pack = (
        total_unique_storage_nbytes_pack - total_param_storage_nbytes_pack
    )
    saved_activation_summary_pack = ",\n".join(
        f"kind={record['kind']} "
        f"shape={record['shape']} dtype={record['dtype']} "
        f"storage_nbytes={record['storage_nbytes']} "
        f"storage_mib={cast(int, record['storage_nbytes']) / (1024**2):.2f} "
        f"refs={record['save_count']}"
        f"{_format_names(record)}"
        f"{_format_sources(record)}"
        for record in saved_activations_by_storage.values()
    )

    # Live view: current storage sizes after forward finishes. This reflects
    # storage-discard checkpointing (e.g., resize_(0)).
    live_saved_activations_by_storage: Dict[Tuple[str, int, int], Dict[str, object]] = {}
    for tensor_ref in saved_tensor_weakrefs:
        tensor = tensor_ref()
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        storage_nbytes = storage.nbytes()
        key = (str(tensor.device), storage.data_ptr(), storage_nbytes)
        record = live_saved_activations_by_storage.get(key)
        tensor_name = _saved_activation_name(tensor)
        tensor_sources = cast(
            set[str],
            saved_activations_by_storage.get(key, {}).get("sources", set()),
        )
        if record is None:
            live_saved_activations_by_storage[key] = {
                "kind": "param" if key in param_storage_keys else "activation",
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "storage_nbytes": storage_nbytes,
                "live_ref_count": 1,
                "names": {tensor_name} if tensor_name is not None else set(),
                "sources": set(tensor_sources),
            }
        else:
            record["live_ref_count"] = cast(int, record["live_ref_count"]) + 1
            if tensor_name is not None:
                record_names = record["names"]
                assert isinstance(record_names, set)
                record_names.add(tensor_name)
            record_sources = record["sources"]
            assert isinstance(record_sources, set)
            record_sources.update(tensor_sources)

    total_unique_storage_nbytes_live = sum(
        cast(int, record["storage_nbytes"])
        for record in live_saved_activations_by_storage.values()
    )
    total_param_storage_nbytes_live = sum(
        cast(int, record["storage_nbytes"])
        for record in live_saved_activations_by_storage.values()
        if record["kind"] == "param"
    )
    total_activation_storage_nbytes_live = (
        total_unique_storage_nbytes_live - total_param_storage_nbytes_live
    )
    saved_activation_summary_live = ",\n".join(
        f"kind={record['kind']} "
        f"shape={record['shape']} dtype={record['dtype']} "
        f"storage_nbytes={record['storage_nbytes']} "
        f"storage_mib={cast(int, record['storage_nbytes']) / (1024**2):.2f} "
        f"live_refs={record['live_ref_count']}"
        f"{_format_names(record)}"
        f"{_format_sources(record)}"
        for record in live_saved_activations_by_storage.values()
    )
    print(
        f"[EP no-sync saved activations] rank={get_rank()}\n"
        f"block={block.block_idx} mem_before={mem_before_gib:.2f} GiB\n"
        f"mem_after={mem_after_gib:.2f} GiB\n"
        f"delta={mem_after_gib - mem_before_gib:.2f} GiB\n"
        f"unique_saved_storage_nbytes={total_unique_storage_nbytes_live}\n"
        f"unique_saved_storage_mib={total_unique_storage_nbytes_live / (1024**2):.2f}\n"
        f"param_saved_storage_nbytes={total_param_storage_nbytes_live}\n"
        f"param_saved_storage_mib={total_param_storage_nbytes_live / (1024**2):.2f}\n"
        f"activation_saved_storage_nbytes={total_activation_storage_nbytes_live}\n"
        f"activation_saved_storage_mib={total_activation_storage_nbytes_live / (1024**2):.2f}\n"
        f"saved_at_pack_unique_storage_nbytes={total_unique_storage_nbytes_pack}\n"
        f"saved_at_pack_unique_storage_mib={total_unique_storage_nbytes_pack / (1024**2):.2f}\n"
        f"saved_at_pack_param_storage_nbytes={total_param_storage_nbytes_pack}\n"
        f"saved_at_pack_param_storage_mib={total_param_storage_nbytes_pack / (1024**2):.2f}\n"
        f"saved_at_pack_activation_storage_nbytes={total_activation_storage_nbytes_pack}\n"
        f"saved_at_pack_activation_storage_mib={total_activation_storage_nbytes_pack / (1024**2):.2f}\n"
        f"saved_live=[\n{saved_activation_summary_live}\n]\n"
        f"saved_at_pack=[\n{saved_activation_summary_pack}\n]"
    )
    _set_train_global_arg(activation_dump_key, True)
    return out
