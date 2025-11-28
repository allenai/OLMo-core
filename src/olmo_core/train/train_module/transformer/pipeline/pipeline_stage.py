


import torch
import torch.distributed as dist
from collections import Counter, defaultdict
from torch.utils._pytree import tree_map_only
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple, cast
import torch
from torch.fx.node import Argument
from torch.distributed._composable import replicate as _rep # type: ignore[attr-defined]
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FSDPModule, fully_shard
from olmo_core.nn.lm_head import LMOutputWithLoss

# from torch.distributed.pipelining._backward import stage_backward, stage_backward_input
from .helpers import stage_backward
import re
import nvtx
from .helpers import(
    _free_tensor_inplace,
    flatten_args,
    normalize_model_output_as_tuple,
)


def _make_tensor_from_meta(
    example: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a real tensor from a tensor.
    """
    return torch.empty(
        example.size(),
        dtype=example.dtype,
        layout=example.layout,
        device=device,
    )

class RecvInfo:
    """
    Represents a stage input.
    """

    def __init__(
        self,
        input_name: str,
        source: int,
        buffer: torch.Tensor,
    ):
        # Name of this input
        self.input_name = input_name
        # Stage index of the source of this input, -1 if first stage (no source)
        self.source = source
        # Buffer to receive the input into.
        self.buffer = buffer


    def __repr__(self):
        return f"RecvInfo(input={self.input_name}, source={self.source}, shape={self.buffer.size()})"



class CustomPipelineStage:
    """
    Drop-in replacement for :class:`PipelineStage` that adds a few features:
    
    1. keeps only a *small
    pool* (≤ pipeline depth) of receive/grad buffers and re-uses them across
    micro-batches.
    
    2. override `forward_maybe_with_nosync` and `backward_maybe_with_nosync` to
    support torch.distributed._composable.replicate (used to be just torch.nn.parallel.DistributedDataParallel)
    
    """

    def __init__(
        self,
        submodule: torch.nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        group: dist.ProcessGroup,
        **kwargs,
    ):
        if stage_index >= num_stages:
            raise ValueError(
                f"Stage index {stage_index} is out of range of {num_stages}"
            )

        # Whether this stage is using replicate (composable) DDP, affects how to control gradient sync in backward
        self.is_rddp: bool = kwargs.pop("is_rddp", False)

        # the model
        self.submod = submodule

        # current stage index
        self.stage_index = stage_index
        
        # total number of stages
        self.num_stages = num_stages
        
        # current device
        self.device = device
        
        # PP process group
        self.group = group

        # hidden states size, used for creating p2p buffers
        d_model = getattr(submodule, 'd_model', None)
        assert isinstance(d_model, int), "submodule must have d_model (int) attribute"
        self.d_model: int = d_model

        # dtype for receive/send buffers
        self.p2p_dtype: torch.dtype = torch.bfloat16

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(self.group)
        self.group_size = dist.get_world_size(self.group)
        if self.group_size > self.num_stages:
            raise RuntimeError(
                f"Pipeline group size {self.group_size} cannot be larger than number of stages {self.num_stages}"
            )

        # map microbatch ID to list of forward tensor args
        # used for: (1) after forward, need to store it before sending to next stage
        #           (2) before backward, need to the stage output for the autograd graph
        self.fwd_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {} # mb_idx -> (stage_input, stage_output)

        # map microbatch ID to list of backward grad tensor args
        # used for: after backward, need to store the stage input grads before sending to previous stage
        self.bwd_cache: dict[int, torch.Tensor] = {} # mb_idx -> stage_input_grads

        # Initialize has_backward to false; this will be set to true if loss
        # function is passed to pipeline schedule
        # self.has_backward = False

        # To be populated later by the Schedule
        # self.chunks: Optional[int] = None
        self.stage_index_to_group_rank: dict[int, int] = {
            i: i % self.group_size for i in range(self.num_stages)
        }

        # self.inputs: Optional[list[torch.Tensor]] = None
        self.inputs_meta: Optional[torch.Tensor] = None
        self.outputs_meta: Optional[torch.Tensor] = None

        self.received_activations: dict[int, torch.Tensor] = {}
        self.received_grads: dict[int, torch.Tensor] = {}

        # these are the buffers used in backwards send/recv, they are allocated later
        # self.outputs_grad: list[torch.Tensor] = []

        # store outputs of this stage for logging purposes, the outer schedule can access this to return
        # information to the train module
        # the outputs should be detached losses
        # None if not last stage
        self.stage_outputs: dict[int, Optional[LMOutputWithLoss]] = {}  # mb_idx -> output

        # runtime per step info (can change in each step())
        self._step_global_batch_size: Optional[int] = None
        self._step_micro_batch_size: Optional[int] = None
        self._step_seqlen: Optional[int] = None


    def get_stage_outputs_as_list(self) -> List[Optional[LMOutputWithLoss]]:
        """
        Get the stage outputs as a list, ordered by microbatch id.
        """
        num_chunks = len(self.stage_outputs)
        return [self.stage_outputs[i] for i in range(num_chunks)]

    @property
    def is_first(self):
        """
        Returns true if this stage is the first stage in the pipeline.
        """
        return self.stage_index == 0

    @property
    def is_last(self):
        """
        Returns true if this stage is the last stage in the pipeline.
        """
        return self.stage_index == self.num_stages - 1

    # def _check_chunk_id(self, chunk_id: int):
    #     if self.chunks is None:
    #         raise RuntimeError(
    #             "Attempted to access chunk_id before chunks have been configured."
    #         )
    #     if chunk_id >= self.chunks:
    #         raise RuntimeError(
    #             f"Chunk id {chunk_id} is out of range [0, {self.chunks})"
    #         )



    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        # print(f"{self.stage_index}.forward_one_chunk({fwd_chunk_id})")

        if self.is_first:
            # First stage doesn't need to receive anything
            composite_args = args
        else:
            # Receive activations for this chunk
            # Activations only come in args form
            stage_input = self._retrieve_recv_activations(fwd_chunk_id)
            stage_input.requires_grad_(True)
            composite_args = (stage_input, ) # make it a tuple

        composite_kwargs = kwargs or {}

        # self._validate_fwd_input(args, kwargs)

        # Compute forward
        # print(f'{self.stage_index}_{fwd_chunk_id}-F-Start')
        output: Union[torch.Tensor, LMOutputWithLoss] = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)
        # print(f'{self.stage_index}_{fwd_chunk_id}-F-End')

        if self.is_last:
            assert isinstance(output, LMOutputWithLoss), "Last stage output must be LMOutputWithLoss"
            logits, loss, ce_loss, z_loss = output # logits, loss, ce_loss, z_loss
            stage_output_with_graph = loss.unsqueeze(0) # this is the final loss to do backward on, it has the autograd graph attached, need to change shape from () to (1, ) for backward
            detached_lm_output = LMOutputWithLoss(
                logits=logits, # NOTE;WARN: logits are too large to keep. If using fused lm head, logits is None
                loss=loss.detach(),
                ce_loss=ce_loss.detach(),
                z_loss=z_loss.detach() if z_loss is not None else None,
            )
            self.stage_outputs[fwd_chunk_id] = detached_lm_output
        else:
            assert isinstance(output, torch.Tensor), "Non-last stage output must be torch.Tensor"
            stage_output_with_graph = output # this is the activation to send to next stage, it has the autograd graph attached
            self.stage_outputs[fwd_chunk_id] = None  # non-last stage has no output to return

        # Save activations and inputs for backward
        self.fwd_cache[fwd_chunk_id] = (
            composite_args[0],  # input_values
            stage_output_with_graph,  # stage_output
        )
        # print(f"{self.stage_index}.forward_one_chunk({fwd_chunk_id}) ret")
        
    

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        last_backward=False,
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.


        last_backward is controlled by the schedule and signals synchronization of gradients across DP groups
        after the last backward.
        """
        # print(f"{self.stage_index}.backward_one_chunk({bwd_chunk_id})")

        _ = torch.zeros(128, device=self.device) # to make nvtx ranges more visible

        # self._check_chunk_id(bwd_chunk_id)

        (
            input_values,
            stage_output,
        ) = self.fwd_cache.pop(bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": None,
                "input_values": [input_values], # need to be a list
            }
        else:
            # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads(bwd_chunk_id)
            # If an input to the pipeline requires gradient,
            # `torch.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": [input_values], # need to be a list
            }

        grads_of_stage_input: Tuple[Optional[torch.Tensor], ...] = ()

        # Custom backward function
        grads_of_stage_input, _ = self.backward_maybe_with_nosync(
            "full", bwd_kwargs, last_backward=last_backward
        )
        # with torch.no_grad():
        #     names = [n for n, p in self.submod.named_parameters()]
        #     debug_norm = torch.nn.utils.get_total_norm([p.grad for p in self.submod.parameters()])
        assert len(grads_of_stage_input) == 1, "Expect only one input to the stage"
        if not self.is_first:
            assert grads_of_stage_input[0] is not None, "Expect input grad to be not None"
            # check for nan
            # if torch.isnan(grads_of_stage_input[0]).any():
            #     raise RuntimeError("NaN detected in gradients of stage input")
            self.bwd_cache[bwd_chunk_id] = grads_of_stage_input[0] # only one input expected
        
        if self.is_last and not self.is_first:
            # Autograd dependencies:
            #    rest_of_autograd_graph -> stage_output -> loss
            # stage_output is no longer used in the last stage for backward and only needed
            # to return to the user in merge_output_chunks, therefore
            # this should be detached to release autograd graph context and free memory earlier
            # TODO: what?
            for t in stage_output:
                if not t._is_view():  # views are not detachable in-place
                    t.detach_()

        _ = torch.zeros(128, device=self.device) # to make nvtx ranges more visible


    def _get_recv_ops(self, source_stage_index, recv_buffer) -> List[dist.P2POp]:
        return self._get_p2p_ops(source_stage_index, recv_buffer, dist.irecv)

    def _get_send_ops(self, dest_stage_index, send_buffer) -> List[dist.P2POp]:
        return self._get_p2p_ops(dest_stage_index, send_buffer, dist.isend)

    def _get_p2p_ops(self, source_stage_index: int, recv_buffer: torch.Tensor, p2p_type: Callable) -> List[dist.P2POp]:
        ops: list[dist.P2POp] = []
        assert source_stage_index >= 0
        assert source_stage_index < self.num_stages
        peer_rank = self.stage_index_to_group_rank[source_stage_index]
        peer_global_rank = (
            peer_rank
            if self.group is None
            else dist.get_global_rank(self.group, peer_rank)
        )
        ops.append(
            dist.P2POp(p2p_type, recv_buffer, peer_global_rank, self.group)
        )

        return ops
    # ------------------- fwd ---------------------
    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> List[dist.P2POp]:
        if self.is_first:
            return []
        
        source_stage_index = self.stage_index - 1
        meta = self.inputs_meta
        assert meta is not None
        recv_buffer = _make_tensor_from_meta(meta, self.device)

        ops = self._get_recv_ops(source_stage_index, recv_buffer)

        self.received_activations[fwd_chunk_id] = recv_buffer

        return ops

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> List[dist.P2POp]:
        if self.is_last:
            return []
        
        dst_stage_index = self.stage_index + 1
        meta = self.outputs_meta
        assert meta is not None
        buffer = self.fwd_cache[fwd_chunk_id][1]  # stage_output

        # check buffer and meta have the same structure
        assert buffer.size() == meta.size()
        assert buffer.dtype == meta.dtype

        ops = self._get_send_ops(dst_stage_index, buffer.detach())
        return ops


    # ------------------- bwd ---------------------

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        if self.is_first:
            return []
        
        dest_stage_index = self.stage_index - 1
        assert dest_stage_index >= 0

        meta = self.inputs_meta # in backward, send grad has the same shape as stage input
        assert meta is not None
        send_buffer = self.bwd_cache.pop(bwd_chunk_id)

        assert send_buffer.size() == meta.size()
        assert send_buffer.dtype == meta.dtype

        ops = self._get_send_ops(dest_stage_index, send_buffer.detach())
        return ops
    

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> List[dist.P2POp]:
        
        # don't need to recv in backward if last stage or no backward
        if self.is_last:
            return []

        source_stage_index = self.stage_index + 1
        assert source_stage_index < self.num_stages


        meta = self.outputs_meta # in backward, recv grad has the same shape as stage output
        assert meta is not None
        recv_buffer = _make_tensor_from_meta(meta, self.device)
        
        ops = self._get_recv_ops(source_stage_index, recv_buffer)

        self.received_grads[bwd_chunk_id] = recv_buffer
        return ops
        
    
    def _retrieve_recv_activations(self, fwd_chunk_id: int):
        return self.received_activations.pop(fwd_chunk_id)

    def _retrieve_recv_grads(
        self,
        bwd_chunk_id: int,
    ):
        return self.received_grads.pop(bwd_chunk_id)
    
    
    def _prepare_forward_backward_meta(self, num_microbatches: int, args_mb, kwargs_mb=None):
        """
        Prepare the meta tensors (to record shapes and dtypes) for forward and backward.
        """
       

        assert self._step_micro_batch_size is not None, "Need to call prepare_step"
        assert self._step_seqlen is not None, "Need to call prepare_step"
        assert self.submod.d_model is not None
        example_p2p_tensor = torch.empty(
            (self._step_micro_batch_size, self._step_seqlen, self.d_model),  
            device="meta", 
            dtype=self.p2p_dtype
        )

        if self.is_first:
            # for the first stage, args_mb is the (input_id, ) tuple
            self.inputs_meta = args_mb[0]
        else:
            # for the non-first stage, args_mb is empty (), but we know
            # the input shapes is (mbsz, seqlen, hidden)
            self.inputs_meta = example_p2p_tensor.clone()

        if self.is_last:
            # for the last stage, no need to send activations
            self.outputs_meta = None
        else:
            # for the non-last stage, we know the output shapes is
            # (mbsz, seqlen, hidden)
            self.outputs_meta = example_p2p_tensor.clone()



    def prepare_step(self, global_batch_size: int, micro_batch_size: int, seqlen: int):
        self._step_global_batch_size = global_batch_size
        self._step_micro_batch_size = micro_batch_size
        self._step_seqlen = seqlen
    


    def forward_maybe_with_nosync(self, *args, **kwargs):
        # If submod is wrapped with DDP, we use the `no_sync` context manager to
        # avoid gradient all-reduce per microbatch
        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync():  # type: ignore[operator]
                out_val = self.submod(*args, **kwargs)
        elif self.is_rddp: # composable.replicate
            assert getattr(self.submod, 'set_requires_gradient_sync', None) is not None, "submod must have set_requires_gradient_sync method"
            assert isinstance(self.submod.set_requires_gradient_sync, Callable)

            self.submod.set_requires_gradient_sync(False) 
            out_val = self.submod(*args, **kwargs)
            self.submod.set_requires_gradient_sync(True) 
        else:
            out_val = self.submod(*args, **kwargs)
        return out_val

    def backward_maybe_with_nosync(
        self, backward_type, bwd_kwargs: Dict, last_backward=False
    ) -> Tuple[Tuple[Optional[torch.Tensor], ...], Optional[List[Dict[str, Any]]]]:
        """
        Whether using PP with FSDP or DDP, there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """

        def perform_backward(
            backward_type,
        ) -> Callable[
            [],
            Tuple[Tuple[Optional[torch.Tensor], ...], Optional[List[Dict[str, Any]]]],
        ]:
            if backward_type == "full":
                return lambda: (
                    stage_backward(
                        bwd_kwargs["stage_output"],
                        bwd_kwargs["output_grads"],
                        bwd_kwargs["input_values"],
                    ),
                    None, # param_groups
                )
            else:
                raise RuntimeError(f"Unknown backward type: {backward_type}")

        # If submod is wrapped by DDP
        if isinstance(self.submod, DistributedDataParallel):
            if last_backward:
                # Last chunk, prepare for gradient reduction
                # HACK: reaching into DDP implementation details here. Is there a better way?
                self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                result = perform_backward(backward_type)()
            else:
                with self.submod.no_sync():  # type: ignore[operator]
                    result = perform_backward(backward_type)()
        elif self.is_rddp: # composable.replicate
            # # The optimizer handles gradient synchronization for us, so always backward with no sync
            # self.submod.set_requires_gradient_sync(False) # type: ignore
            # result = perform_backward(backward_type)()
            # self.submod.set_requires_gradient_sync(True) # type: ignore

            if last_backward:
                state = _rep.state(self.submod)            # grab _ReplicateState
                ddp_impl = state._ddp                      # the hidden real DDP
                ddp_impl.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                result = perform_backward(backward_type)()
                pass
            else:
                self.submod.set_requires_gradient_sync(False) # type: ignore
                result = perform_backward(backward_type)()
                self.submod.set_requires_gradient_sync(True) # type: ignore
        
        # If submod is a FSDP module
        elif isinstance(self.submod, FSDPModule):
            self.submod.set_is_last_backward(False)
            self.submod.set_reshard_after_backward(False)
            self.submod.set_requires_gradient_sync(False)
            result = perform_backward(backward_type)()
            if last_backward:
                # Manually call post backward for FSDP
                def run_post_backward(fsdp_module: FSDPModule) -> None:
                    fsdp_module.set_is_last_backward(True)
                    fsdp_module.set_reshard_after_backward(True)
                    fsdp_module.set_requires_gradient_sync(True)
                    fsdp_state = fully_shard.state(fsdp_module)  # type: ignore[arg-type]
                    for state in fsdp_state._state_ctx.all_states:
                        if state._fsdp_param_group:
                            state._fsdp_param_group.post_backward()

                run_post_backward(self.submod)
        else:
            # Non-DP submodule, regular backward
            result = perform_backward(backward_type)()

        grads, param_groups = result
        return grads, param_groups

    def clear_runtime_states(self) -> None:
        pass
        # TODO: anything to clear?

    
    def clear_step_info(self):
        self._step_global_batch_size = None
        self._step_micro_batch_size = None
        self._step_seqlen = None

        self.stage_outputs = {}