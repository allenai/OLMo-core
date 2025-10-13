import torch
from typing import Any, Optional

def _free_tensor_inplace(t: torch.Tensor) -> None:
    """
    Release GPU storage held by *t* without breaking autograd graphs that might
    still reference the `torch.Tensor` *object*.

    Setting `t.data` to an empty tensor:
    * frees the CUDA allocation,
    * preserves dtype / device so later checks do not error,
    * keeps the Python object alive (harmless zero‑sized placeholder).
    """
    if t is not None and t.numel():
        t.data = torch.empty((1,), device=t.device, dtype=t.dtype)


def generate_stage_to_rank_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, int]:
    """
    Compute the stage id to rank mapping for either a looped or V-style schedule.

    Most commonly num_stages == pp_size * 2, but this function can be used to
    compute the mapping for any number of stages per rank.
    """
    mapping = {}
    if style == "loop":
        for stage_index in range(num_stages):
            mapping[stage_index] = stage_index % pp_size
    elif style == "v":
        if num_stages % pp_size != 0:
            raise ValueError(
                f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size} for V schedules"
            )

        rank_index = 0
        for stage_index in range(num_stages):
            mapping[stage_index] = rank_index
            # dont change rank if we are on the border (to keep v shape)
            if (stage_index + 1) % pp_size == 0:
                continue
            if (stage_index // pp_size) % 2 == 0:
                rank_index += 1
            else:
                rank_index -= 1
    else:
        raise ValueError(f"Style {style} is not supported.")
    return mapping




def flatten_args(args):
    """
    Flatten the args into a list form.
    """
    flat_args = []

    def extract_tensor_args(a):
        nonlocal flat_args
        flat_args.append(a)
        return a

    torch.fx.node.map_aggregate( # type: ignore[attr-defined]
        args,
        extract_tensor_args,
    )

    return flat_args


def normalize_model_output_as_tuple(output: Any) -> tuple[Any]:

    if type(output) is list:
        # HACK: this is a hacky workaround for the fact that export creates
        # output in list format
        output = tuple(output)

    # Unify output form to tuple for easy correspondance with
    # `act_send_info`
    output_tuple = output if type(output) is tuple else (output,)
    return output_tuple



def stage_backward(
    stage_output: torch.Tensor,
    output_grads: Optional[torch.Tensor],
    input_values: list[torch.Tensor],
) -> tuple[Optional[torch.Tensor], ...]:


    # stage_output may be a composite datatype like dict. Extract all individual
    # tensor values here
    stage_output_tensors: list[torch.Tensor] = []
    output_grad_tensors: list[Optional[torch.Tensor]] = []

    def extract_tensors_with_grads(
        output_val,
        grad_val,
        # Don't delete me- see [Note: ref cycle]
        extract_tensors_with_grads,
    ):
        if isinstance(output_val, torch.Tensor):
            if not output_val.requires_grad and output_val.grad_fn is None:
                return
            assert isinstance(grad_val, (torch.Tensor, type(None))), (
                f"Expected Tensor or None gradient but got {type(grad_val)}"
            )
            stage_output_tensors.append(output_val)
            output_grad_tensors.append(grad_val)
        elif isinstance(output_val, (tuple, list)):
            if grad_val is None:
                return
            assert isinstance(grad_val, (tuple, list)), (
                f"grad_value expected to have type {type(output_val)} but got {type(grad_val)}"
            )
            assert len(output_val) == len(grad_val)
            for ov, gv in zip(output_val, grad_val):
                extract_tensors_with_grads(
                    ov,
                    gv,
                    extract_tensors_with_grads,
                )
        elif isinstance(output_val, dict):
            if grad_val is None:
                return
            assert isinstance(grad_val, dict)
            assert set(output_val.keys()) == set(grad_val.keys())
            for k in output_val.keys():
                extract_tensors_with_grads(
                    output_val[k], grad_val[k], extract_tensors_with_grads
                )
        else:
            # Output is a non-tensor type; just ignore it
            pass

    # Note: ref cycle
    # break a ref cycle that would keep tensors alive until GC runs
    # 1. extract_tensors_with_grads refers to a cell that holds refs to any vars defined in stage_backward
    #    and used in extract_tensors_with_grads
    # 2. extract_tensors_with_grads referred to both stage_output_tensors, output_grad_tensors,
    #    and to itself (extract_tensors_with_grads) since it makes a recursive call
    # 3. stage_output_tensors was kept alive by the above refcycle, and it holds activation tensors, which is bad
    # fix -> explicitly pass in the ref to the fn, so there is no gc cycle anymore
    extract_tensors_with_grads(
        stage_output, output_grads, extract_tensors_with_grads
    )

    torch.autograd.backward(
        stage_output_tensors,
        grad_tensors=output_grad_tensors,  # type: ignore[arg-type]
    )

    # Extract gradients wrt the input values
    grad_inputs: list[Optional[torch.Tensor]] = []
    for val in input_values:
        if isinstance(val, torch.Tensor):
            grad_inputs.append(val.grad)
        else:
            grad_inputs.append(None)

    # Alternative impl: `torch.autograd.grad`.
    # Note that `torch.autograd.grad` will not accumulate gradients into the
    # model's parameters.
    """
    inputs_with_grad = []
    for val in input_values:
        if isinstance(val, torch.Tensor) and val.requires_grad:
            inputs_with_grad.append(val)

    grad_inputs = torch.autograd.grad(
        stage_output_tensors, inputs_with_grad, output_grad_tensors,  # type: ignore[arg-type]
    )
    """



    return tuple(grad_inputs)

