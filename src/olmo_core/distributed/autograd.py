import torch

from olmo_core.distributed.utils import get_world_size


class _AllToAll(torch.autograd.Function):
    """
    Autograd-compatible all-to-all collective operation.

    This implements a differentiable all-to-all operation that can be used in models
    requiring gradient computation through distributed communication. Supports both
    equal splits (all2all) and unequal splits (all2all-v).

    The backward pass performs a reverse all-to-all with swapped split sizes to
    correctly propagate gradients.
    """

    @staticmethod
    def forward(
        ctx,
        group: torch.distributed.ProcessGroup,
        iinput: torch.Tensor,
        output_split_sizes: list[int] | None,
        input_split_sizes: list[int] | None,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = get_world_size(group)
        if world_size == 1:
            return iinput

        iinput = iinput.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(iinput)
        else:
            # Unequal split (all2all-v)
            output = iinput.new_empty(
                size=[sum(output_split_sizes)] + list(iinput.size()[1:]),
                dtype=iinput.dtype,
                device=torch.cuda.current_device(),
            )
        torch.distributed.all_to_all_single(
            output,
            iinput,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return (
            None,  # group
            _AllToAll.apply(  # iinput (reverse all-to-all)
                ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes
            ),
            None,  # output_split_sizes
            None,  # input_split_sizes
        )


def all_to_all(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: list[int] | None = None,
    input_split_sizes: list[int] | None = None,
) -> torch.Tensor:
    return _AllToAll.apply(group, input_, output_split_sizes, input_split_sizes)  # type: ignore[return-value]
