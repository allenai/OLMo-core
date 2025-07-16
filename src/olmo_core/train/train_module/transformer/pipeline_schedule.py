# ---------------------------------------------------------------------------
# Memory-efficient pipeline stage and 1F1B schedule
# ---------------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.distributed as dist

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Up-stream helpers
from torch.distributed.pipelining.stage import (
    PipelineStage as _TorchPipelineStage,
    _RecvInfo,
    _RootArgPlaceholder,
    _make_tensor_from_meta,
    InputInfo,
    _normalize_model_output_as_tuple,
)
from torch.distributed.pipelining.schedules import (
    Schedule1F1B as _Schedule1F1B,
    ScheduleInterleaved1F1B as _ScheduleInterleaved1F1B,
)
from torch.distributed.pipelining.schedules import _batch_p2p
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.pipelining._backward import stage_backward, stage_backward_input, stage_backward_weight
import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

import torch
from torch.fx.node import Argument
from torch.distributed._composable import replicate as _rep
import nvtx


def friendly_debug_info(v: object) -> Argument:
    """
    Helper function to print out debug info in a friendly way.
    """
    if isinstance(v, torch.Tensor):
        return f"Tensor({v.shape}, grad={v.requires_grad}, dtype={v.dtype})"
    else:
        return str(v)


def map_debug_info(a: Argument) -> Argument:
    """
    Helper function to apply `friendly_debug_info` to items in `a`.
    `a` may be a list, tuple, or dict.
    """
    return torch.fx.node.map_aggregate(a, friendly_debug_info)

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

def flatten_args(args):
    """
    Flatten the args into a list form.
    """
    flat_args = []

    def extract_tensor_args(a):
        nonlocal flat_args
        flat_args.append(a)
        return a

    torch.fx.node.map_aggregate(
        args,
        extract_tensor_args,
    )

    return flat_args



# --------------------------------------------------------------------- Stage


class CustomPipelineStage(_TorchPipelineStage):
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
        *args,
        buffer_pool_size: Optional[int] = None,
        **kwargs,
    ):
        self.is_rddp: bool = kwargs.pop("is_rddp", False)
        super().__init__(*args, **kwargs)
        # At most «pipeline depth» distinct micro-batches live in a stage
        default_pool = min(self.num_stages, kwargs.get("num_microbatches", self.num_stages))
        self._pool_size: int = buffer_pool_size or default_pool
        
        self.output_chunks: dict = {} # override

    # ---------------- infrastructural overrides -----------------

    # override to get rid of `self.output_chunks`
    @nvtx.annotate("CustomPipelineStage.forward_one_chunk", color='blue')
    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage.
        As of Sept 2024:
        - `args` applies to the first stage only, other stages receives args
          through activation transmission.
        - `kwargs` can be passed to all stages via respective `step` calls.
        """

        if self.is_first:
            # First stage doesn't need to receive anything
            composite_args = args
        else:
            # Receive activations for this chunk
            # Activations only come in args form
            composite_args = self._retrieve_recv_activations(fwd_chunk_id)

        composite_kwargs = kwargs or {}

        self._validate_fwd_input(args, kwargs)

        # Compute forward
        try:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)

        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        # See [Note: pipeline model output type]
        output_tuple = _normalize_model_output_as_tuple(output)

        # Prepare for final output merge or reduction
        self.output_chunks[fwd_chunk_id] = output

        # Save activations and inputs for backward
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[fwd_chunk_id] = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

        logger.debug(
            "%s Forwarded chunk %s, outputs: %s",
            self.log_prefix,
            fwd_chunk_id,
            map_debug_info(output),
        )
        self._validate_fwd_outputs(output_tuple)

        # We return the original user-provied output, not normalized to tuple.
        # See [Note: pipeline model output type]
        return output
    
    # override to add nvtx
    @nvtx.annotate("CustomPipelineStage.backward_one_chunk", color='red')
    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is optional that `dw_runner` was provided to the PipelineStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.

        last_backward is controlled by the schedule and signals synchronization of gradients across DP groups
        after the last backward.
        """
        self._check_chunk_id(bwd_chunk_id)

        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
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
                "input_values": input_values,
            }

        grads_input: Tuple[Optional[torch.Tensor], ...] = ()

        # Custom backward function
        if self.dw_builder:
            # TODO: We may want to change our semantics so we are allowed to ignore
            # the 'dw_builder' and call full_backward directly when it is a full_backward op.
            grads_input, _ = self.backward_maybe_with_nosync(
                "full", bwd_kwargs, last_backward=last_backward
            )
            if full_backward:
                self.dw_builder()()
            else:
                self.dw_runner[bwd_chunk_id] = self.dw_builder()
        else:
            if full_backward:
                grads_input, _ = self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward
                )
            else:
                param_groups: List[Dict[str, Any]] | None = None
                # Skip the backward for the first stage since we will perform the weight update with
                # autograd.backward in backward_weight_one_chunk
                if not self.is_first:
                    if isinstance(bwd_kwargs["stage_output"], torch.Tensor):
                        bwd_kwargs["stage_output"] = (bwd_kwargs["stage_output"],)

                    # perform the partial backwards for the inputs with a custom backward function
                    # when the "stage_ouput" is a loss, then it is a tensor, otherwise it is a tuple of tensors
                    grads_input, param_groups = self.backward_maybe_with_nosync(
                        "input", bwd_kwargs, last_backward=last_backward
                    )

                # TODO: we dont need to save this, add to dw_runner?
                self.backward_state[bwd_chunk_id] = (
                    bwd_kwargs["input_values"],
                    param_groups,
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                )
                # Save a placeholder for the dw_runner
                self.dw_runner[bwd_chunk_id] = lambda: None

        self.bwd_cache[bwd_chunk_id] = grads_input

        if self.is_last and not self.is_first:
            # Autograd dependencies:
            #    rest_of_autograd_graph -> stage_output -> loss
            # stage_output is no longer used in the last stage for backward and only needed
            # to return to the user in merge_output_chunks, therefore
            # this should be detached to release autograd graph context and free memory earlier
            for t in stage_output:
                if not t._is_view():  # views are not detachable in-place
                    t.detach_()

        logger.debug("%s Backwarded chunk %s", self.log_prefix, bwd_chunk_id)
    
    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> List[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
        recv_infos: Tuple[InputInfo, ...] = self.args_recv_info[fwd_chunk_id]
        for info in recv_infos: # tag the recv info with the fwd chunk id so that we can track which chunk it belongs to
            if isinstance(info, _RecvInfo):
                assert getattr(info, "assigned_to_fwd_chunk_id", None) is None, \
                    f"Expected recv info to not be assigned to fwd chunk {fwd_chunk_id}, " \
                    f"but got {info.assigned_to_fwd_chunk_id}"
                info.assigned_to_fwd_chunk_id = fwd_chunk_id # type: ignore[attr-defined]
        return self._get_recv_ops(recv_infos)

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> List[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the gradients
        for this stage.
        """
        if not self.has_backward or self.is_last:
            return []

        recv_infos = self.grad_recv_info[bwd_chunk_id]
        for info in recv_infos: # tag the recv info with the bwd chunk id so that we can track which chunk it belongs to
            info.assigned_to_bwd_chunk_id = bwd_chunk_id # type: ignore[attr-defined]
        return self._get_recv_ops(recv_infos)
    
    def _retrieve_recv_activations(self, fwd_chunk_id: int):
        """
        Retrieve the activations received for the current stage during forward.
        """
        recv_infos = self.args_recv_info[fwd_chunk_id]
        
        for info in recv_infos:  # check tag
            if isinstance(info, _RecvInfo):
                assert info.assigned_to_fwd_chunk_id == fwd_chunk_id, \
                    f"Expected recv info to be assigned to fwd chunk {fwd_chunk_id}, " \
                    f"but got {info.assigned_to_fwd_chunk_id}" 
                info.assigned_to_fwd_chunk_id = None

        activations = self._map_tensor_from_recv_info(recv_infos)
        return activations

    def _retrieve_recv_grads(
        self,
        bwd_chunk_id: int,
    ):
        """
        Retrieve the gradients received for the current stage during backward.
        """
        recv_infos = self.grad_recv_info[bwd_chunk_id]  
        for info in recv_infos:  # check tag
            assert info.assigned_to_bwd_chunk_id == bwd_chunk_id, \
                f"Expected recv info to be assigned to bwd chunk {bwd_chunk_id}, " \
                f"but got {info.assigned_to_bwd_chunk_id}"

        grads = self._map_tensor_from_recv_info(recv_infos)
        return grads
    
    def _prepare_forward_infra(self, num_microbatches: int, args, kwargs=None):
        if self.inputs_meta is None:
            self._shape_inference(args, kwargs)

        assert self.inputs_meta is not None
        self._fwd_recv_pool: List[Tuple[_RecvInfo | _RootArgPlaceholder, ...]] = []
        for k in range(self._pool_size):
            if self.is_first:
                infos = tuple(_RootArgPlaceholder(m) for m in self.inputs_meta)
            else:
                infos = tuple(
                    _RecvInfo(
                        f"recv_fwd_pool{k}_{i}",
                        self.stage_index - 1,
                        _make_tensor_from_meta(meta, self.device),
                    )
                    for i, meta in enumerate(self.inputs_meta)
                )
                if self.has_backward:
                    for r in infos:
                        r.buffer.requires_grad_(True)
            self._fwd_recv_pool.append(infos)

        # Map every micro-batch to a pool entry (by modulo)
        self.args_recv_info = {
            i:  self._fwd_recv_pool[i % self._pool_size] for i in range(num_microbatches)
        }

        # Forward send info unchanged
        self.act_send_info = {
            i: ([self.stage_index + 1] if not self.is_last else [])
            for i in range(len(self.get_outputs_meta()))
        }
        return tuple()

    def _prepare_backward_infra(self, num_microbatches: int):
        self.chunks = num_microbatches
        if self.is_last:
            self.grad_recv_info = {i: tuple() for i in range(num_microbatches)}
            return
        self._bwd_recv_pool: List[Tuple[_RecvInfo, ...]] = []
        for k in range(self._pool_size):
            infos = tuple(
                _RecvInfo(
                    f"recv_grad_pool{k}_{i}",
                    self.stage_index + 1,
                    _make_tensor_from_meta(meta, self.device),
                )
                for i, meta in enumerate(self.get_outputs_meta())
            )
            self._bwd_recv_pool.append(infos)
        self.grad_recv_info = {
            i: self._bwd_recv_pool[i % self._pool_size] for i in range(num_microbatches)
        }

    # ---------------- reclamation helper -----------------

    def release_microbatch_buffers(self, mb_idx: int):
        """Free activation & grad buffers for *mb_idx*."""
        for store in (
            getattr(self, "args_recv_info", None),
            getattr(self, "grad_recv_info", None),
        ):
            if not store:
                continue
            entry = store[mb_idx]
            if entry is None:
                continue
            container = store
            for rec in entry:
                if isinstance(rec, _RecvInfo):
                    _free_tensor_inplace(rec.buffer)
            # Drop the tuple reference itself so the GC can reclaim it.
            container[mb_idx] = None

    def forward_maybe_with_nosync(self, *args, **kwargs):
        # If submod is wrapped with DDP, we use the `no_sync` context manager to
        # avoid gradient all-reduce per microbatch
        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync():  # type: ignore[operator]
                out_val = self.submod(*args, **kwargs)
        elif self.is_rddp: # composable.replicate
            self.submod.set_requires_gradient_sync(False) # type: ignore
            out_val = self.submod(*args, **kwargs)
            self.submod.set_requires_gradient_sync(True) # type: ignore
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
                    None,
                )
            elif backward_type == "input":
                return lambda: stage_backward_input(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                    self.submod.parameters(),
                )
            elif backward_type == "weight":
                return lambda: (
                    stage_backward_weight(
                        self.submod.parameters(), bwd_kwargs["param_groups"]
                    ),
                    None,
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

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> List[dist.P2POp]:
        """
        Get the activation send ops for current stage's forward.
        """
        output = self.output_chunks.pop(fwd_chunk_id)
        # Unify output form to tuple for easy correspondance with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)

        ops: List[dist.P2POp] = []

        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            for dst in dst_stages:
                if dst is None:
                    continue
                logger.debug(
                    "%s Sending tensor to Stage %s: %s",
                    self.log_prefix,
                    dst,
                    out.size(),
                )
                peer_rank = self.stage_index_to_group_rank[dst]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )  # TODO
                ops.append(dist.P2POp(dist.isend, out, peer_global_rank, self.group))

        return ops
    
    def clear_runtime_states(self) -> None:
        """
        Clear runtime states of the stage.
        """
        # map microbatch ID to list of forward tensor args
        self.fwd_cache.clear()
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks.clear()

        # Clear grad of input buffers in between schedule steps. This is because
        # `torch.autograd.backward()` will accumulate gradients into leaf
        # tensors by default. For gradients to pass back to previous stages, we
        # don't want such accumulation.
        for recv_tuple in self.args_recv_info.values():  # iterate over all chunks
            for a in recv_tuple:  # iterate over all input args
                if isinstance(a, _RecvInfo):
                    # Set to None is the newer and recommended way to clear grads, compared to `zero_()`.
                    # See https://github.com/pytorch/pytorch/pull/92731
                    a.buffer.grad = None
# --------------------------------------------------------------------- Schedule


class CustomSchedule1F1B(_Schedule1F1B):
    """
    Same algorithm as PyTorch’s Schedule1F1B, but calls
    ``release_microbatch_buffers`` as soon as the gradient send for a micro-batch
    completes.
    """

    # override to discard merged outputs
    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """

        # Clean per iteration
        self._stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        # if self._stage.is_last:
        #     return self._merge_outputs(self._stage.output_chunks)
        # else:
        #     return None
        
        return None # don't keep the outputs, we don't need them in this schedule
    
    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        assert isinstance(self._stage, CustomPipelineStage), \
            "CustomSchedule1F1B requires CustomPipelineStage as the stage type."
        
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        
        if not self._stage_initialized:
            self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        warmup = min(self._n_microbatches, self._num_stages - self._stage.stage_index)
        fwd_i, bwd_i = 0, 0
        send_w: Optional[dist.Work] = None
        fwd_sends = []
        # pending_release: Optional[int] = None

        # ---------------- warm-up ----------------
        for _ in range(warmup):
            if wrk := _batch_p2p(self._stage.get_fwd_recv_ops(fwd_i), "fwd_recv"):
                wrk.wait()
            
            out = self._stage.forward_one_chunk(fwd_i, arg_mbs[fwd_i], kwarg_mbs[fwd_i])

            if send_w is not None:
                send_w.wait()

            fwd_sends = self._stage.get_fwd_send_ops(fwd_i)
            if fwd_i != warmup - 1:
                send_w = _batch_p2p(fwd_sends, "fwd_send")

            self._maybe_compute_loss(self._stage, out, target_mbs, fwd_i)
            fwd_i += 1

        # ---------------- 1B1F ----------------
        while True:
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_i)
            if wrk := _batch_p2p(fwd_sends + bwd_recvs, "fwd_send_bwd_recv"):
                wrk.wait()

            loss = self._maybe_get_loss(self._stage, bwd_i)
            self._stage.backward_one_chunk(
                bwd_i, loss, last_backward=(bwd_i == self._n_microbatches - 1)
            )
            bwd_sends = self._stage.get_bwd_send_ops(bwd_i)
            bwd_i += 1

            if fwd_i == self._n_microbatches:
                break

            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_i)
            
            if wrk := _batch_p2p(bwd_sends + fwd_recvs, "bwd_send_fwd_recv"):
                wrk.wait()
                # self._stage.release_microbatch_buffers(bwd_i)

            out = self._stage.forward_one_chunk(fwd_i, arg_mbs[fwd_i], kwarg_mbs[fwd_i])
            
            self._maybe_compute_loss(self._stage, out, target_mbs, fwd_i)
            
            fwd_sends = self._stage.get_fwd_send_ops(fwd_i)

            fwd_i += 1


        send_w = _batch_p2p(bwd_sends, desc="bwd_send")
        
        # print("cooldown")

        # ---------------- cooldown ----------------
        while bwd_i < self._n_microbatches:
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_i)
            if wrk := _batch_p2p(bwd_recvs, "bwd_recv"):
                wrk.wait()

            loss = self._maybe_get_loss(self._stage, bwd_i)

            self._stage.backward_one_chunk(
                bwd_i, loss, last_backward=(bwd_i == self._n_microbatches - 1)
            )


            if send_w is not None:
                send_w.wait()
                # self._stage.release_microbatch_buffers(bwd_i - 1)

            bwd_sends = self._stage.get_bwd_send_ops(bwd_i)
            send_w = _batch_p2p(bwd_sends, "bwd_send")
            bwd_i += 1

        if send_w is not None:
            send_w.wait()


        self._update_losses(self._stage, losses)
        
        
class CustomScheduleInterleaved1F1B(_ScheduleInterleaved1F1B):
    
    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """
        # Clean per iteration
        for stage in self._stages:
            stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        # for stage in self._stages:
        #     if stage.is_last:
        #         return self._merge_outputs(stage.output_chunks)
        # Does not contain the last stage
        return None