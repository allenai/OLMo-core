import pytest
import torch

from olmo_core.distributed.fsdp.stream import CudaStream, Stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.gpu
def test_cuda_stream():
    default_stream = Stream.default(torch.device("cuda"))
    assert isinstance(default_stream, CudaStream)
    assert isinstance(default_stream.base_stream, torch.cuda.Stream)

    current_stream = Stream.current(torch.device("cuda"))
    assert isinstance(current_stream, CudaStream)
    assert isinstance(current_stream.base_stream, torch.cuda.Stream)
