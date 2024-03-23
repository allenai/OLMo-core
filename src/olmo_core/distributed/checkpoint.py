import json
import logging
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from pydantic import BaseModel
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save_file

from ..io import PathOrStr, dir_is_empty, get_bytes_range
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
    def save(self, dir: PathOrStr, state_dict: Dict[str, torch.Tensor]):
        """
        Save a state dict. The state dict can contain regular Tensors, Parameters, or :class:`ShardedFlatParameter`s.

        When calling this from a distributed context, all ranks must call this at the same time and the
        state dict must have the same keys and tensor types across each rank.
        """
        # TODO: support remote directories.

        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        if not dir_is_empty(dir):
            raise FileExistsError(f"Checkpoint directory {dir} is not empty!")

        barrier()

        global_save_plan, metadata = self._get_global_save_plan_and_metadata(state_dict)

        local_rank = get_rank()
        local_state_dict: Dict[str, torch.Tensor] = {}
        for key in state_dict.keys():
            tensor_save_plan = global_save_plan.tensors[key]

            if (local_offsets := tensor_save_plan.flattened_offsets_per_rank.get(local_rank)) is not None:
                local_flat_tensor = state_dict[key].data.detach().flatten()
                assert local_offsets[1] - local_offsets[0] == local_flat_tensor.numel()
                local_state_dict[key] = local_flat_tensor

        safetensors_save_file(local_state_dict, dir / self._filename_for_rank(local_rank))

        # Save metadata.
        if local_rank == 0:
            with open(dir / self.METADATA_FILENAME, "w") as f:
                json.dump(metadata.model_dump(), f)

    @torch.no_grad()
    def load(self, dir: PathOrStr, state_dict: Dict[str, torch.Tensor]):
        """
        Load a state dict in-place.
        """
        dir = Path(dir)
        local_rank = get_rank()

        # Collect metadata from rank 0, scatter to other ranks.
        metadata: Optional[StorageMetadata] = None
        if local_rank == 0:
            with open(dir / self.METADATA_FILENAME, "r") as f:
                metadata = StorageMetadata(**json.load(f))
        metadata = scatter_object(metadata)
        assert metadata is not None

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
                    with safe_open(dir / filename, framework="pt", device="cpu") as f:  # type: ignore
                        flat_tensor_to_load = f.get_slice(key)
                        numel_in_file = flat_tensor_to_load.get_shape()[0]

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
                        flat_tensor[flat_tensor_start:flat_tensor_end].copy_(
                            flat_tensor_to_load[flat_tensor_to_load_start:flat_tensor_to_load_end]
                        )

            state_dict[key].copy_(flat_tensor.view(tensor.shape))


def get_safetensors_header(path: PathOrStr) -> Dict[str, Any]:
    length_of_header = struct.unpack("<Q", get_bytes_range(path, 0, 8))[0]
    return json.loads(get_bytes_range(path, 8, length_of_header))
