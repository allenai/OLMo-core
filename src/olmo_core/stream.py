from __future__ import annotations

from abc import ABC
from contextlib import contextmanager
from typing import Generator, Optional

import torch


class Stream(ABC):
    """
    This is generally just a thin wrapper around ``torch.cuda.Stream``, except we provide a
    no-op implementation for CPU as well.
    """

    @classmethod
    def default(cls, device: Optional[torch.device] = None) -> Stream:
        if (device is not None and device.type == "cuda") or (device is None and torch.cuda.is_available()):
            return CudaStream(torch.cuda.default_stream(device))
        else:
            return CpuStream()

    @classmethod
    def current(cls, device: Optional[torch.device] = None) -> Stream:
        if (device is not None and device.type == "cuda") or (device is None and torch.cuda.is_available()):
            return CudaStream(torch.cuda.current_stream(device))
        else:
            return CpuStream()

    @classmethod
    def new(cls, device: Optional[torch.device] = None) -> Stream:
        if (device is not None and device.type == "cuda") or (device is None and torch.cuda.is_available()):
            # TODO: mess with priority?
            return CudaStream(torch.cuda.Stream())
        else:
            return CpuStream()

    @contextmanager
    def __call__(self, wait_stream: Optional[Stream] = None) -> Generator[Stream, None, None]:
        if wait_stream is not None:
            self.wait_stream(wait_stream)
        with self:
            yield self

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb

    def wait_stream(self, other: Stream):
        del other

    def record_for(self, tensor: torch.Tensor):
        del tensor


class CudaStream(Stream):
    def __init__(self, base_stream: torch.cuda.Stream):
        self.base_stream = base_stream
        self.stream_context = None

    def __enter__(self) -> Stream:
        self.stream_context = torch.cuda.stream(self.base_stream)
        self.stream_context.__enter__()  # type: ignore
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.stream_context is not None
        self.stream_context.__exit__(exc_type, exc_val, exc_tb)

    def wait_stream(self, other: Stream):
        if isinstance(other, CudaStream):
            self.base_stream.wait_stream(other.base_stream)
        elif isinstance(other, torch.cuda.Stream):
            self.base_stream.wait_stream(other)
        elif not isinstance(other, Stream):
            raise ValueError(f"expected a Stream, got {type(other)}")

    def record_for(self, tensor: torch.Tensor):
        tensor.record_stream(self.base_stream)


class CpuStream(Stream):
    pass
