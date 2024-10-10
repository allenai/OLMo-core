import dataclasses
import io
import logging
import operator
import pickle
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.filesystem import WriteResult
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, StorageMeta
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.futures import Future

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoCheckpointError
from olmo_core.io import (
    get_bytes_range,
    init_client,
    is_url,
    normalize_path,
    resource_path,
    upload,
)
from olmo_core.utils import generate_uuid, get_default_thread_count

log = logging.getLogger(__name__)


@dataclass
class _StorageInfo:
    """This is the per entry storage info."""

    relative_path: str
    offset: int
    length: int


@dataclass
class _StoragePrefix:
    prefix: str


def _item_size(item: WriteItem) -> int:
    size = 1
    assert item.tensor_data is not None
    # can't use math.prod as PT needs to support older python
    for s in item.tensor_data.size:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _split_by_size_and_type(bins: int, items: List[WriteItem]) -> List[List[WriteItem]]:
    if bins == 1:
        return [items]

    bytes_w = [wi for wi in items if wi.type == WriteItemType.BYTE_IO]
    tensor_w = [wi for wi in items if wi.type != WriteItemType.BYTE_IO]

    buckets: List[List[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]

    tensor_w.sort(key=_item_size, reverse=True)

    for i, wi in enumerate(bytes_w):
        buckets[i % bins].append(wi)

    for wi in tensor_w:
        # TODO replace with headq
        idx = min(enumerate(bucket_sizes), key=operator.itemgetter(1))[0]
        buckets[idx].append(wi)
        bucket_sizes[idx] += _item_size(wi)

    return buckets


def _write_items(
    path: str, storage_key: str, items: List[WriteItem], planner: SavePlanner
) -> List[WriteResult]:
    results: List[WriteResult] = []

    tmp_path = Path(
        tempfile.mktemp(suffix=".distcp", dir=None if is_url(path) else Path(path).parent)
    )
    try:
        with tmp_path.open("wb") as tmp_file:
            for write_item in items:
                offset = tmp_file.tell()
                data = planner.resolve_data(write_item)
                if isinstance(data, torch.Tensor):
                    data = data.cpu()
                    if data.storage().size() != data.numel():
                        data = data.clone()
                    torch.save(data, tmp_file)
                else:
                    tmp_file.write(data.getbuffer())

                length = tmp_file.tell() - offset
                results.append(
                    WriteResult(
                        index=write_item.index,
                        size_in_bytes=length,
                        storage_data=_StorageInfo(storage_key, offset, length),
                    )
                )

        if is_url(path):
            upload(tmp_path, path, save_overwrite=True)
        else:
            tmp_path.rename(path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return results


def _narrow_tensor_by_index(
    tensor: torch.Tensor, offsets: Sequence[int], sizes: Sequence[int]
) -> torch.Tensor:
    """
    Narrow the tensor according to ``offsets`` and ``sizes``.
    """
    narrowed_tensor = tensor
    for idx, (offset, size) in enumerate(zip(offsets, sizes)):
        if size < tensor.size(idx):
            # Reshape to get shard for this rank and we don't want autograd
            # recording here for the narrow op and 'local_shard' should be a
            # leaf variable in the autograd graph.
            narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
    return narrowed_tensor


class RemoteFileSystemWriter(dist_cp.StorageWriter):
    """
    A :class:`~torch.distributed.checkpoint.StorageWriter` that can write directly to both cloud
    and local storage.
    """

    def __init__(
        self,
        path: PathOrStr,
        thread_count: Optional[int] = None,
    ) -> None:
        super().__init__()
        if thread_count is not None and thread_count <= 0:
            raise ValueError("thread count must be at least 1")
        self.path = normalize_path(path)
        self.thread_count = thread_count or get_default_thread_count()
        self.save_id = generate_uuid()

    def reset(self, checkpoint_id: Optional[PathOrStr] = None) -> None:
        if checkpoint_id:
            self.path = normalize_path(checkpoint_id)
        self.save_id = generate_uuid()

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        del is_coordinator

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        if not is_url(self.path):
            path = Path(self.path)
            path.mkdir(exist_ok=True, parents=True)
        return plan

    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        new_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_"))
            for i, plan in enumerate(plans)
        ]
        return new_plans

    def write_data(
        self,
        plan: dist_cp.SavePlan,
        planner: dist_cp.SavePlanner,
    ) -> Future[List[WriteResult]]:
        if is_url(self.path):
            # Create the global S3 client up front to work around a threading issue in boto.
            init_client(self.path)

        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0

        def gen_file_name() -> str:
            nonlocal file_count
            file_name = f"{storage_plan.prefix}{file_count}.distcp"
            file_count += 1
            return file_name

        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            futures = []
            for bucket in _split_by_size_and_type(self.thread_count, plan.items):
                file_name = gen_file_name()
                path = f"{self.path}/{file_name}"
                futures.append(executor.submit(_write_items, path, file_name, bucket, planner))

            results = []
            for f in as_completed(futures):
                try:
                    results += f.result()
                except BaseException:
                    # NOTE: we might get an error here that can't be pickled, which causes a different failure
                    # later when PyTorch tries to reduce that error across ranks. So here we just make
                    # sure we're raising a simple error type that can be pickled.
                    raise OLMoCheckpointError(f"Original error:\n{traceback.format_exc()}")

        fut: Future[List[WriteResult]] = Future()
        fut.set_result(results)
        return fut

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        storage_md = dict()
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md
        metadata.storage_meta = self.storage_meta()

        tmp_path = Path(
            tempfile.mktemp(
                suffix=".tmp",
                dir=None if is_url(self.metadata_path) else Path(self.metadata_path).parent,
            )
        )
        try:
            with tmp_path.open("wb") as tmp_file:
                pickle.dump(metadata, tmp_file)

            if is_url(self.metadata_path):
                upload(tmp_path, self.metadata_path, save_overwrite=True)
            else:
                tmp_path.rename(self.metadata_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def storage_meta(self) -> Optional[StorageMeta]:
        return StorageMeta(checkpoint_id=self.checkpoint_id, save_id=self.save_id)

    @property
    def metadata_path(self) -> str:
        return f"{self.path}/.metadata"

    @property
    def checkpoint_id(self) -> str:
        """
        return the checkpoint_id that will be used to save the checkpoint.
        """
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: PathOrStr) -> bool:
        del checkpoint_id
        return True


class RemoteFileSystemReader(dist_cp.StorageReader):
    """
    A :class:`~torch.distributed.checkpoint.StorageReader` based on :class:`~torch.distributed.checkpoint.FileSystemReader`
    that can read data directly from cloud storage as well as a local directory.
    """

    def __init__(self, path: PathOrStr, *, thread_count: Optional[int] = None):
        super().__init__()
        if thread_count is not None and thread_count <= 0:
            raise ValueError("thread count must be at least 1")
        self.path = normalize_path(path)
        self.thread_count = thread_count or get_default_thread_count()
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()
        self.load_id = generate_uuid()
        self._metadata: Optional[Metadata] = None

    def _get_bytes(self, relative_path: str, offset: int, length: int) -> bytes:
        return get_bytes_range(f"{self.path}/{relative_path}", offset, length)

    def _get_content_for_read(self, read_item: ReadItem) -> Tuple[ReadItem, bytes]:
        sinfo = self.storage_data[read_item.storage_index]
        content = self._get_bytes(sinfo.relative_path, sinfo.offset, sinfo.length)
        return (read_item, content)

    def reset(self, checkpoint_id: Optional[PathOrStr] = None) -> None:
        self.storage_data = dict()
        if checkpoint_id:
            self.path = normalize_path(checkpoint_id)
        self.load_id = generate_uuid()

    def read_data(self, plan: dist_cp.LoadPlan, planner: dist_cp.LoadPlanner) -> Future[None]:
        # Create the global S3 client up front to work around a threading issue in boto.
        if isinstance(self.path, str):
            init_client(self.path)

        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            read_item_content_futures = []
            for read_item in plan.items:
                read_item_content_futures.append(
                    executor.submit(self._get_content_for_read, read_item)
                )
            read_item_content_results = []
            for f in as_completed(read_item_content_futures):
                try:
                    read_item_content_results.append(f.result())
                except BaseException:
                    # NOTE: we might get an error here that can't be pickled, which causes a different failure
                    # later when PyTorch tries to reduce that error across ranks. So here we just make
                    # sure we're raising a simple error type that can be pickled.
                    raise OLMoCheckpointError(f"Original error:\n{traceback.format_exc()}")

        # Modified from `FileSystemReader.read_data()`
        for read_item, content in read_item_content_results:
            bytes = io.BytesIO(content)
            bytes.seek(0)
            if read_item.type == LoadItemType.BYTE_IO:
                planner.load_bytes(read_item, bytes)
            else:
                # NOTE: 'weights_only=False' needed to load torchao's float8 linear layer checkpoints
                tensor = cast(
                    torch.Tensor, torch.load(bytes, map_location="cpu", weights_only=False)
                )
                tensor = _narrow_tensor_by_index(
                    tensor, read_item.storage_offsets, read_item.lengths
                )
                target_tensor = planner.resolve_tensor(read_item).detach()

                assert (
                    target_tensor.size() == tensor.size()
                ), f"req {read_item.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                target_tensor.copy_(tensor)
                planner.commit_tensor(read_item, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        if self._metadata is None:
            with resource_path(self.path, ".metadata").open("rb") as metadata_file:
                metadata = pickle.load(metadata_file)

            if getattr(metadata, "storage_meta", None) is None:
                metadata.storage_meta = StorageMeta()
            metadata.storage_meta.load_id = self.load_id

            self._metadata = metadata

        return self._metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        del is_coordinator
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: dist_cp.LoadPlan) -> dist_cp.LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: List[dist_cp.LoadPlan]) -> List[dist_cp.LoadPlan]:
        return global_plan

    @property
    def checkpoint_id(self) -> str:
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: PathOrStr) -> bool:
        del checkpoint_id
        return True
