import torch
import torch.distributed.nn.functional as dist_nn


def all_to_all_single(
    group: torch.distributed.ProcessGroup,
    input_: torch.Tensor,
    output_split_sizes: list[int] | None = None,
    input_split_sizes: list[int] | None = None,
) -> torch.Tensor:
    """
    Autograd-compatible all-to-all collective operation.

    Each process splits input tensor and then scatters the split list to all processes in a group.
    Then concatenates the received tensors from all the processes in the group and returns a
    single output tensor.

    :param group: The process group to use for the collective.
    :param input_: Input tensor to scatter.
    :param output_split_sizes: Output split sizes for dim 0. If None, dim 0 of output tensor
        must divide equally by world_size.
    :param input_split_sizes: Input split sizes for dim 0. If None, dim 0 of input tensor
        must divide equally by world_size.
    :returns: The gathered concatenated output tensor.
    """
    # Allocate output tensor
    if output_split_sizes is None:
        output = torch.empty_like(input_)
    else:
        output = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=input_.device,
        )

    # Note: all_to_all_single handles grad contiguity internally.
    return dist_nn.all_to_all_single(  # type: ignore[return-value]
        output,
        input_.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
