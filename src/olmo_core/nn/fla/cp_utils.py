import torch
import torch.distributed as dist


def _a2a(x: torch.Tensor, stage: int, group: dist.ProcessGroup) -> torch.Tensor:
    """All-to-all with reshapes matching Ulysses (stage 1 sharding heads, stage 2 restoring)."""
    assert stage in [1, 2]
    x = x.contiguous()
    world_size = dist.get_world_size(group)
    t, h, d = x.shape
    if stage == 1:
        x = x.reshape(t, world_size, h // world_size, d).transpose(0, 1).contiguous()
    else:
        x = x.reshape(world_size, t // world_size, h, d).contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    if stage == 1:
        out = out.flatten(0, 1)
    else:
        out = out.transpose(0, 1).contiguous().flatten(1, 2)
    return out


class _All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, stages: tuple[int, int], group: dist.ProcessGroup):
        fwd_stage, bwd_stage = stages
        out = _a2a(x, fwd_stage, group)
        ctx.bwd_stage = bwd_stage
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_in = _a2a(grad_out, ctx.bwd_stage, ctx.group)
        return grad_in, None, None


def _qkvo_all2ll(x: torch.Tensor, *, is_qkv: bool, group: dist.ProcessGroup) -> torch.Tensor:
    stages = (1, 2) if is_qkv else (2, 1)
    return _All2All.apply(x, stages, group)
