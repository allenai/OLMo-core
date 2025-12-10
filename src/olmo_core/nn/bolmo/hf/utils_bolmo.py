import math

import torch
import torch.nn.functional as F


def compute_boundary_mask(boundary_logprobs: torch.Tensor, boundary_threshold: str) -> torch.Tensor:
    if boundary_threshold.startswith("sample:"):
        _, temperature = boundary_threshold.split(":")
        temperature = float(temperature)

        if temperature == 0:
            return (boundary_logprobs > math.log(0.5))
        elif temperature == 1:
            return torch.bernoulli(torch.exp(boundary_logprobs)).to(torch.bool)
        else:
            raise NotImplementedError("Temperatures outside {0,1} are not implemented yet.")
    elif boundary_threshold.startswith("topk:"):
        _, topk = boundary_threshold.split(":")
        topk = int(topk)
        thresholds = torch.quantile(boundary_logprobs, dim=1, q=1 - (topk / boundary_logprobs.shape[1]))
        return (boundary_logprobs >= thresholds.unsqueeze(-1))
    elif boundary_threshold.startswith("topk_percent:"):
        _, topk_percent = boundary_threshold.split(":")
        topk_percent = float(topk_percent)
        assert 0 <= topk_percent <= 1
        thresholds = torch.quantile(boundary_logprobs, dim=1, q=1 - topk_percent)
        return (boundary_logprobs >= thresholds.unsqueeze(-1))
    else:
        raise ValueError(f"Unknown boundary threshold: {boundary_threshold}")


def _pad(tensors: list[torch.Tensor], multiple_of: int, direction: str, value):
    max_len = max(t.size(0) for t in tensors)
    if multiple_of > 1:
        # Round up max_len to the nearest multiple_of
        max_len = ((max_len + multiple_of - 1) // multiple_of) * multiple_of
    padded = []
    for t in tensors:
        if direction == "left":
            pad_shape = (max_len - t.size(0), 0)
        elif direction == "right":
            pad_shape = (0, max_len - t.size(0))
        else:
            raise ValueError(f"Unknown direction: {direction}. Must be 'left' or 'right'.")
        padded.append(F.pad(t, pad_shape, value=value))
    return torch.stack(padded, dim=0)

def pad_right(
    tensors: list[torch.Tensor],
    multiple_of: int = 128,
    value=0,
):
    return _pad(tensors, multiple_of, direction="right", value=value)

def pad_left(
    tensors: list[torch.Tensor],
    multiple_of: int = 128,
    value=0,
):
    return _pad(tensors, multiple_of, direction="left", value=value)

class MaskState:
    def __init__(self, mask):
        self.cpu_mask = mask.cpu()

        self.mask = mask
        self.inv_mask = ~mask
        self._all = self.cpu_mask.all().item()
        self._any = self.cpu_mask.any().item()

    def any(self):
        return self._any

    def all(self):
        return self._all

    def selective_get(self, x, inv=False):
        # try to avoid sync through nonzero on index
        if inv:
            if self.all():
                return x[[]]
            elif not self.any():
                return x
            else:
                return x[self.inv_mask]
        else:
            if self.all():
                return x
            elif not self.any():
                return x[[]]
            else:
                return x[self.mask]

    def selective_put(self, x, out, inv=False):
        # try to avoid sync through nonzero on index
        if inv:
            if self.all():
                return
            elif not self.any():
                out.copy_(x)
            else:
                out[self.inv_mask] = x
        else:
            if self.all():
                out.copy_(x)
            elif not self.any():
                return
            else:
                out[self.mask] = x
    
    def selective_add(self, x, out, inv=False):
        # try to avoid sync through nonzero on index
        if inv:
            if self.all():
                return
            elif not self.any():
                out.add_(x)
            else:
                out[self.inv_mask] += x
        else:
            if self.all():
                out.add_(x)
            elif not self.any():
                return
            else:
                out[self.mask] += x