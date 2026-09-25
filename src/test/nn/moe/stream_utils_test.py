import torch

from olmo_core.nn.moe.utils import run_on_stream_no_compile, wait_stream_no_compile
from olmo_core.testing import requires_gpu


@requires_gpu
def test_compiled_side_stream_backward_after_separate_compiled_loss():
    # The separate loss graph replaces Dynamo's stream registry. On Torch 2.13,
    # tracing the stream context itself makes the preceding graph's backward fail.
    torch.manual_seed(44)
    stream = torch.cuda.Stream()

    def compute(x, w):
        assert torch.compiler.is_compiling(), "Side-stream math must stay compiled"
        return torch.sin(x @ w)

    def forward(x, w):
        wait_stream_no_compile(stream, torch.cuda.current_stream())
        y = run_on_stream_no_compile(stream, compute, x, w)
        wait_stream_no_compile(torch.cuda.current_stream(), stream)
        return x + y

    compiled_forward = torch.compile(forward)
    compiled_loss = torch.compile(lambda x: x.square().mean())
    for _ in range(2):
        x = torch.randn(16, 32, device="cuda", requires_grad=True)
        w = torch.randn(32, 32, device="cuda", requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        wr = w.detach().clone().requires_grad_()
        ref = (xr + torch.sin(xr @ wr)).square().mean()
        ref.backward()
        loss = compiled_loss(compiled_forward(x, w))
        loss.backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(loss, ref, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(x.grad, xr.grad, rtol=1e-4, atol=2e-6)
        torch.testing.assert_close(w.grad, wr.grad, rtol=1e-4, atol=2e-6)
