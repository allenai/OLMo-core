from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist


class _StreamBarrierFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fwd_event, bwd_event, paired_fwd_event, paired_bwd_event):
        # Stash for backward
        ctx.bwd_event = bwd_event
        ctx.paired_fwd_event = paired_fwd_event

        # Record this barrier's forward event, then wait on the *paired* backward event
        s = torch.cuda.current_stream(x.device)
        s.record_event(fwd_event)
        s.wait_event(paired_bwd_event)
        return x

    @staticmethod
    def backward(ctx, grad):
        # Record this barrier's backward event, then wait on the *paired* forward event
        s = torch.cuda.current_stream(grad.device)
        s.record_event(ctx.bwd_event)
        s.wait_event(ctx.paired_fwd_event)
        # One gradient for each forward input (x, fwd_event, bwd_event, paired_fwd_event, paired_bwd_event)
        return grad, None, None, None, None


class StreamBarrier(torch.nn.Module):
    """
    A module that provides a (fwd_event, bwd_event) pair per CUDA device and
    can be 'paired' with another StreamBarrier. In forward:
      - record self.fwd_event  and wait on paired.bwd_event
    In backward:
      - record self.bwd_event  and wait on paired.fwd_event
    """

    def __init__(self):
        super().__init__()
        # Per-device event storage: {device_index: (fwd_event, bwd_event)}
        self.fwd_event = torch.cuda.Event()
        self.bwd_event = torch.cuda.Event()

        self._paired: Optional["StreamBarrier"] = None

    def pair_with(self, other: "StreamBarrier"):
        self._paired = other
        other._paired = self

    def forward(self, x: torch.Tensor):
        assert x.is_cuda, "StreamBarrier expects CUDA tensors."
        assert self._paired is not None, "StreamBarrier must be paired before use."

        return _StreamBarrierFn.apply(
            x, self.fwd_event, self.bwd_event, self._paired.fwd_event, self._paired.bwd_event
        )
