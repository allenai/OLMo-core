import json
import logging
import struct
import sys
import tempfile
from functools import cached_property, reduce
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import safetensors as sft
import safetensors.torch as sft_torch
import torch
from cached_path import cached_path
from pydantic import BaseModel

from ..exceptions import OLMoUserError
from ..io import (
    PathOrStr,
    clear_directory,
    dir_is_empty,
    file_exists,
    get_bytes_range,
    is_url,
    upload,
)
from ..utils import wait_for
from .sharded_flat_parameter import ShardedFlatParameter
from .utils import barrier, get_rank, scatter_object

log = logging.getLogger(__name__)


class TensorStorageMetadata(BaseModel):
    flattened_offsets_per_file: Dict[str, Tuple[int, int]]
    """
    Maps file name to the offsets within the full flattened tensor that the shard in the file
    corresponds to.
    """

    shape: Tuple[int, ...]
    """
    The shape of the full (unflattened) tensor.
    """


class StorageMetadata(BaseModel):
    tensors: Dict[str, TensorStorageMetadata]


class TensorSavePlan(BaseModel):
    flattened_offsets_per_rank: Dict[int, Tuple[int, int]]
    """
    Maps global process rank to the offsets within the full flattened tensor that the shard for the
    rank corresponds to. Some ranks may be omitted.
    """


class SavePlan(BaseModel):
    tensors: Dict[str, TensorSavePlan]


class SafeTensorsLoader:
    """
    A wrapper around ``safetensors`` loading functionality for PyTorch that works with remote
    files as well without having to download the whole file.

    This should be used a context manager.
    """

    def __init__(self, path: PathOrStr):
        self.path = path
        self.safe_open: Optional[sft.safe_open] = None

    @cached_property
    def header_length(self) -> int:
        return struct.unpack("<Q", get_bytes_range(self.path, 0, 8))[0]

    @cached_property
    def header(self) -> Dict[str, Any]:
        return json.loads(get_bytes_range(self.path, 8, self.header_length))

    def get_shape(self, key: str) -> Tuple[int, ...]:
        return self.header[key]["shape"]

    def get_dtype(self, key: str) -> torch.dtype:
        return sft_torch._getdtype(self.header[key]["dtype"])

    def get_numel(self, key: str) -> int:
        return reduce(lambda x, y: x * y, self.get_shape(key), 1)

    def get_flat_slice(self, key: str, start_idx: int = 0, end_idx: Optional[int] = None) -> torch.Tensor:
        if self.safe_open is not None:
            return self.safe_open.get_slice(key)[start_idx:end_idx]  # type: ignore
        elif is_url(self.path):
            # Validate indices. Can only work with positive indices.
            if start_idx < 0:
                start_idx = self.get_numel(key) + start_idx
            elif start_idx > self.get_numel(key):
                raise IndexError(f"slice start index ({start_idx}) out of range")

            if end_idx is None:
                end_idx = self.get_numel(key)
            elif end_idx < 0:
                end_idx = self.get_numel(key) + end_idx
            elif end_idx > self.get_numel(key):
                raise IndexError(f"slice end index ({end_idx}) out of range")

            dtype = self.get_dtype(key)
            bytes_per_item = sft_torch._SIZE[dtype]
            num_bytes = bytes_per_item * (end_idx - start_idx)

            # Transform `start_idx` into a byte offset.
            offset_start = self.header[key]["data_offsets"][0]
            offset_start += bytes_per_item * start_idx
            # At this point `offset_start` is an offset into the byte-buffer part
            # of the file, not the file itself. We have to offset further by the header size byte
            # and the number of bytes in the header itself.
            offset_start += 8 + self.header_length

            # Load the tensor.
            array_bytes = get_bytes_range(self.path, offset_start, num_bytes)
            tensor = torch.frombuffer(bytearray(array_bytes), dtype=dtype)
            if sys.byteorder == "big":
                tensor = torch.from_numpy(tensor.numpy().byteswap(inplace=False))
            return tensor
        else:
            raise OLMoUserError(
                f"{self.__class__.__name__} is meant to be used as a context manager, did you forget to call __enter__?"
            )

    def __enter__(self):
        if not is_url(self.path):
            self.safe_open = sft.safe_open(self.path, framework="pt", device="cpu")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.safe_open is not None:
            self.safe_open.__exit__(exc_type, exc_val, exc_tb)  # type: ignore
            self.safe_open = None


class SafeTensorsMultiFileLoader:
    """
    A wrapper around :class:`SafeTensorsLoader` that should be used when working with multiple ``safetensors``
    files at once to avoid unnecessary IO.
    """

    def __init__(self):
        self.loaders: Dict[str, SafeTensorsLoader] = {}

    def open(self, path: PathOrStr) -> SafeTensorsLoader:
        if (loader := self.loaders.get(str(path))) is not None:
            return loader
        loader = SafeTensorsLoader(path)
        self.loaders[str(path)] = loader
        return loader


class Checkpointer:
    METADATA_FILENAME = "metadata.json"

    def _filename_for_rank(self, rank: int) -> str:
        return f"rank_{rank}.safetensors"

    def _get_global_save_plan_and_metadata(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[SavePlan, StorageMetadata]:
        tensors_save_plan = {}
        tensors_metadata = {}
        for key in state_dict.keys():
            tensor = state_dict[key]

            flattened_offsets_per_rank = {}
            full_shape: Tuple[int, ...]

            if isinstance(tensor, ShardedFlatParameter):
                for rank, offset in enumerate(tensor.sharding_spec.unsharded_flattened_offsets):
                    flattened_offsets_per_rank[rank] = offset
                full_shape = tensor.unsharded_shape
            else:
                flattened_offsets_per_rank = {0: (0, tensor.numel())}
                full_shape = tuple(tensor.shape)

            tensors_save_plan[key] = TensorSavePlan(flattened_offsets_per_rank=flattened_offsets_per_rank)
            tensors_metadata[key] = TensorStorageMetadata(
                flattened_offsets_per_file={
                    self._filename_for_rank(rank): offsets for rank, offsets in flattened_offsets_per_rank.items()
                },
                shape=full_shape,
            )

        tensors_save_plan = scatter_object(tensors_save_plan)
        tensors_metadata = scatter_object(tensors_metadata)
        return SavePlan(tensors=tensors_save_plan), StorageMetadata(tensors=tensors_metadata)

    @torch.no_grad()
    def save(self, dir: PathOrStr, state_dict: Dict[str, torch.Tensor], save_overwrite: bool = False):
        """
        Save a state dict. The state dict can contain regular Tensors, Parameters, or :class:`ShardedFlatParameter`s.

        When calling this from a distributed context, all ranks must call this at the same time and the
        state dict must have the same keys and tensor types across each rank.
        """
        if str(dir).startswith("file://"):
            dir = str(dir).replace("file://", "", 1)

        local_rank = get_rank()

        local_dir: Path
        remote_dir: Optional[str] = None
        clean_up_local_dir = False
        if not is_url(dir):
            local_dir = Path(dir)
            if local_rank == 0:
                if save_overwrite and not dir_is_empty(local_dir):
                    clear_directory(local_dir)
                local_dir.mkdir(parents=True, exist_ok=True)

            barrier()

            # All ranks wait for rank 0 to create the directory. On NFS the directory might
            # not be available immediately. This also ensures all ranks share the filesystem.
            description = f"Waiting for '{local_dir}' to be created"
            try:
                wait_for(local_dir.exists, description)
            except TimeoutError as e:
                raise RuntimeError(
                    f"{description} timed out, please ensure each rank is saving to the same directory on a shared filesystem."
                ) from e
        else:
            local_dir = Path(tempfile.mkdtemp())
            remote_dir = str(dir).rstrip("/")
            clean_up_local_dir = True
            # NOTE: we do have the ability to clear bucket storage "folders" via `clear_directory`,
            # but that's super dangerous. All it takes is one person passing in the wrong folder
            # name and they could wipe out a ton of very important checkpoints.
            if local_rank == 0:
                if file_exists(f"{remote_dir}/{self.METADATA_FILENAME}"):
                    raise FileExistsError(
                        f"Remote checkpoint directory '{remote_dir}' already contains a checkpoint!"
                    )

        try:
            if not dir_is_empty(local_dir):
                raise FileExistsError(f"Checkpoint directory '{local_dir}' is not empty!")

            barrier()

            global_save_plan, metadata = self._get_global_save_plan_and_metadata(state_dict)

            # Construct local flat tensors state dict to save.
            local_state_dict: Dict[str, torch.Tensor] = {}
            for key in state_dict.keys():
                tensor_save_plan = global_save_plan.tensors[key]

                if (local_offsets := tensor_save_plan.flattened_offsets_per_rank.get(local_rank)) is not None:
                    local_flat_tensor = state_dict[key].data.detach().flatten()
                    assert local_offsets[1] - local_offsets[0] == local_flat_tensor.numel()
                    local_state_dict[key] = local_flat_tensor

            # Save safetensors file.
            local_sft_path = local_dir / self._filename_for_rank(local_rank)
            sft_torch.save_file(local_state_dict, local_sft_path)
            if remote_dir is not None:
                upload(
                    local_sft_path,
                    f"{remote_dir}/{self._filename_for_rank(local_rank)}",
                    save_overwrite=save_overwrite,
                )

            # Save metadata.
            if local_rank == 0:
                metadata_path = local_dir / self.METADATA_FILENAME
                with open(metadata_path, "w") as f:
                    json.dump(metadata.model_dump(), f)

                if remote_dir is not None:
                    upload(metadata_path, f"{remote_dir}/{self.METADATA_FILENAME}", save_overwrite=save_overwrite)

            barrier()
        finally:
            if clean_up_local_dir and local_dir.exists():
                clear_directory(local_dir)

    @torch.no_grad()
    def load(self, dir: PathOrStr, state_dict: Dict[str, torch.Tensor]):
        """
        Load a state dict in-place.
        """
        dir = str(dir).rstrip("/")
        if dir.startswith("file://"):
            dir = dir.replace("file://", "", 1)

        local_rank = get_rank()

        # Collect metadata from rank 0, scatter to other ranks.
        metadata: Optional[StorageMetadata] = None
        if local_rank == 0:
            with open(cached_path(f"{dir}/{self.METADATA_FILENAME}")) as f:
                metadata = StorageMetadata(**json.load(f))
        metadata = scatter_object(metadata)
        assert metadata is not None

        safetensors_mfl = SafeTensorsMultiFileLoader()

        # Load each tensor from the slices in each file.
        for key in state_dict.keys():
            tensor_storage_metadata = metadata.tensors[key]
            tensor = state_dict[key]

            flat_tensor: torch.Tensor
            offsets: Tuple[int, int]
            if isinstance(tensor, ShardedFlatParameter):
                if tensor.unsharded_shape != tensor_storage_metadata.shape:
                    raise ValueError(
                        f"Shape mismatched for '{key}', expected {tuple(tensor.unsharded_shape)}, found {tensor_storage_metadata.shape}"
                    )

                offsets = tensor.unsharded_flattened_offsets
                flat_tensor = tensor.detach()
            else:
                if tensor.shape != tensor_storage_metadata.shape:
                    raise ValueError(
                        f"Shape mismatched for '{key}', expected {tuple(tensor.shape)}, found {tensor_storage_metadata.shape}"
                    )

                offsets = (0, tensor.numel())
                flat_tensor = tensor.detach().cpu().flatten()

            for filename, offsets_in_file in tensor_storage_metadata.flattened_offsets_per_file.items():
                # Check for overlap in offsets, and if there is overlap, load the slice from disk.
                if (
                    offsets_in_file[0] <= offsets[0] < offsets_in_file[1]
                    or offsets_in_file[0] < offsets[1] <= offsets_in_file[1]
                ):
                    with safetensors_mfl.open(f"{dir}/{filename}") as loader:
                        if len((shape_in_file := loader.get_shape(key))) != 1:
                            raise ValueError(
                                f"Expected a 1D tensor at {key} in {filename}, found shape {shape_in_file}"
                            )

                        if (dtype := loader.get_dtype(key)) != flat_tensor.dtype:
                            raise ValueError(
                                f"Data type mismatch between tensor to load ({dtype}) and to load into ({flat_tensor.dtype})"
                            )

                        numel_in_file = loader.get_numel(key)

                        # Start and end index of the slice within `flat_tensor` that we're going to load
                        # from a slice of `flat_tensor_to_load`.
                        flat_tensor_start, flat_tensor_end = 0, flat_tensor.numel()
                        # Start and end index of the slice within `flat_tensor_to_load` that we're going
                        # to load into the slice of `flat_tensor`.
                        flat_tensor_to_load_start, flat_tensor_to_load_end = 0, numel_in_file
                        # There are 5 scenarios to consider in terms of where the tensors overlap.
                        # Suppose the original flat tensor has 6 elements: 'x x x x x x'
                        # -------------------------------------------
                        # (A) flat_tensor_slice_to_load: [x x x]x x x  (0, 3)
                        #     flat_tensor:                x x[x x x]x  (2, 5)
                        # -------------------------------------------
                        # (B) flat_tensor_slice_to_load:  x x[x x x]x  (2, 5)
                        #     flat_tensor:               [x x x]x x x  (0, 3)
                        # -------------------------------------------
                        # (C) flat_tensor_slice_to_load:  x[x x x x]x  (1, 5)
                        #     flat_tensor:                x x[x x]x x  (2, 4)
                        # -------------------------------------------
                        # (D) flat_tensor_slice_to_load:  x x[x x]x x  (2, 4)
                        #     flat_tensor:                x[x x x x]x  (1, 5)
                        # -------------------------------------------
                        # (E) flat_tensor_slice_to_load:  x x[x x]x x  (2, 4)
                        #     flat_tensor:                x x[x x]x x  (2, 4)
                        # -------------------------------------------
                        if offsets[0] <= offsets_in_file[0]:
                            # Scenarios (B), (D), (E)
                            flat_tensor_start = offsets_in_file[0] - offsets[0]
                        else:
                            # Scenarios (A), (C)
                            flat_tensor_to_load_start = offsets[0] - offsets_in_file[0]

                        if offsets[1] <= offsets_in_file[1]:
                            # Scenarios (B), (C), (E)
                            flat_tensor_to_load_end -= offsets_in_file[1] - offsets[1]
                        else:
                            # Scenarios (A), (D)
                            flat_tensor_end -= offsets[1] - offsets_in_file[1]

                        log.debug(
                            "Loading '%s'\n  offsets: %s\n  offsets in file: %s\n  load into: (%s, %s)\n  load from: (%s, %s)",
                            key,
                            offsets,
                            offsets_in_file,
                            flat_tensor_start,
                            flat_tensor_end,
                            flat_tensor_to_load_start,
                            flat_tensor_to_load_end,
                        )

                        # Load the slice.
                        flat_tensor_to_load = loader.get_flat_slice(
                            key, flat_tensor_to_load_start, flat_tensor_to_load_end
                        )
                        flat_tensor[flat_tensor_start:flat_tensor_end].copy_(flat_tensor_to_load)

                        del flat_tensor_to_load

            state_dict[key].copy_(flat_tensor.view(tensor.shape))
            del flat_tensor
