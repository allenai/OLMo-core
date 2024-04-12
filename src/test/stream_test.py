import torch

from olmo_core.stream import CudaStream, Stream

from .utils import requires_gpu


@requires_gpu
def test_cuda_stream():
    device = torch.device("cuda")

    default_stream = Stream.default(device)
    assert isinstance(default_stream, CudaStream)
    assert isinstance(default_stream.base_stream, torch.cuda.Stream)

    current_stream = Stream.current(device)
    assert isinstance(current_stream, CudaStream)
    assert isinstance(current_stream.base_stream, torch.cuda.Stream)

    other_stream = Stream.new(device)
    assert isinstance(other_stream, CudaStream)

    x = torch.empty((100, 100), device=device).normal_(0.0, 1.0)
    other_stream.wait_stream(default_stream)
    with other_stream:
        assert torch.cuda.current_stream(device) == other_stream.base_stream
        y = torch.sum(x)
    assert torch.cuda.current_stream(device) == default_stream.base_stream

    default_stream.wait_stream(other_stream)
    del x, y

    x = torch.empty((100, 100), device=device).normal_(0.0, 1.0)
    with other_stream(wait_stream=default_stream):
        assert torch.cuda.current_stream(device) == other_stream.base_stream
        y = torch.sum(x)
    assert torch.cuda.current_stream(device) == default_stream.base_stream

    default_stream.wait_stream(other_stream)
    del x, y
