import logging
import re
import time
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import nvtx
import torch
import torch.distributed as dist
from torch.distributed.pipelining.microbatch import TensorChunkSpec
from torch.fx.node import Argument
from torch.utils._pytree import tree_map_only

from olmo_core.nn.lm_head import LMOutputWithLoss

from .gpu_activation_offload import GPUActivationOffloader
from .helpers import generate_stage_to_rank_mapping
from .pipeline_stage import CustomPipelineStage

logger = logging.getLogger(__name__)


# Helper to parse an action string like 1F0 into a tuple of (stage_index, computation_type, microbatch_index)
_action_regex = re.compile(r"(\d+)(F|I|B|W|SEND_F|RECV_F|SEND_B|RECV_B)(\d*)")


class PipelineActionType(Enum):
    FORWARD = 1
    BACKWARD_INPUT = 2
    BACKWARD_WEIGHT = 3
    SEND_F = 6
    RECV_F = 7
    SEND_B = 8
    RECV_B = 9
    FULL_BACKWARD = 10
    FULL_BACKWARD_CONT = 11

    def __str__(self):
        str_map = {
            PipelineActionType.FORWARD: "F",
            PipelineActionType.BACKWARD_INPUT: "I",
            PipelineActionType.BACKWARD_WEIGHT: "W",
            PipelineActionType.SEND_F: "SEND_F",
            PipelineActionType.RECV_F: "RECV_F",
            PipelineActionType.SEND_B: "SEND_B",
            PipelineActionType.RECV_B: "RECV_B",
            PipelineActionType.FULL_BACKWARD: "B",
            PipelineActionType.FULL_BACKWARD_CONT: "B_",
        }
        return str_map[self]

    @staticmethod
    def from_str(action):
        if action == "F":
            return PipelineActionType.FORWARD
        elif action == "I":
            return PipelineActionType.BACKWARD_INPUT
        elif action == "W":
            return PipelineActionType.BACKWARD_WEIGHT
        elif action == "SEND_F":
            return PipelineActionType.SEND_F
        elif action == "RECV_F":
            return PipelineActionType.RECV_F
        elif action == "SEND_B":
            return PipelineActionType.SEND_B
        elif action == "RECV_B":
            return PipelineActionType.RECV_B
        elif action == "B":
            return PipelineActionType.FULL_BACKWARD
        elif action == "B_":
            return PipelineActionType.FULL_BACKWARD_CONT
        else:
            raise RuntimeError(f"Invalid computation type {action}")


FORWARD = PipelineActionType.FORWARD
BACKWARD_INPUT = PipelineActionType.BACKWARD_INPUT
BACKWARD_WEIGHT = PipelineActionType.BACKWARD_WEIGHT

SEND_F = PipelineActionType.SEND_F
RECV_F = PipelineActionType.RECV_F
SEND_B = PipelineActionType.SEND_B
RECV_B = PipelineActionType.RECV_B
FULL_BACKWARD = PipelineActionType.FULL_BACKWARD
FULL_BACKWARD_CONT = PipelineActionType.FULL_BACKWARD_CONT


class PipelineAction(NamedTuple):
    stage_index: int
    computation_type: PipelineActionType
    microbatch_index: Optional[int] = None
    need_offload: bool = False
    need_reload: bool = False

    def __repr__(self):
        repr = str(self.stage_index)
        repr += str(self.computation_type)
        if self.microbatch_index is not None:
            repr += str(self.microbatch_index)

        if self.need_offload:
            repr += "↗"
        else:
            # repr += " "
            pass

        if self.need_reload:
            repr = "↘" + repr
        else:
            # repr = " " + repr
            pass

        return repr

    @staticmethod
    def from_str(action_string: str):
        """
        Reverse of __repr__

        String should be formatted as [stage][action type][(microbatch)]
            e.g. `2F0`, `1UNSHARD`, `3SEND_F1`
        """
        action_string = action_string.strip()
        if match := _action_regex.match(action_string):
            stage_index, computation_type, microbatch_index = match.groups()
            return PipelineAction(
                int(stage_index),
                PipelineActionType.from_str(computation_type),
                int(microbatch_index) if len(microbatch_index) else None,
            )
        elif action_string == "":
            return None
        raise RuntimeError(
            f"Invalid action string: {action_string}, should be formatted as [stage][action type][(microbatch)] e.g. 2F0"
        )


class CustomScheduleInterleaved1F1B:
    def reset_n_microbatches(self, n_microbatches: int):
        self._n_microbatches = n_microbatches
        self.number_of_rounds = max(1, n_microbatches // self.pp_group_size)
        self.microbatches_per_round = n_microbatches // self.number_of_rounds
        if n_microbatches % self.number_of_rounds != 0:
            raise ValueError(
                "Interleaved 1F1B requires the number of microbatches to be a "
                f"multiple of the number of rounds ({self.number_of_rounds}), "
                f"but got {n_microbatches}."
            )
        self.configure_pipeline_order()

    def __init__(
        self,
        stages: list[CustomPipelineStage],
        n_microbatches: int,
        # loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[dict[str, Any], tuple[Any]]] = None,
    ):
        self.pp_group_size = stages[0].group_size

        # Chunking specification for positional inputs. (default: `None`)
        self._args_chunk_spec = args_chunk_spec
        # Chunking specification for keyword inputs. (default: `None`)
        self._kwargs_chunk_spec = kwargs_chunk_spec
        self._output_merge_spec = output_merge_spec
        """
        # args_chunk_spec and kwargs_chunk_spec specify how to chunk inputs.
        # They are used to convert batch to microbatches in `step(x)`.  See
        # `TensorChunkSpec` for helper methods for creating them.
        """

        logger.info("Using %s", self.__class__.__name__)

        # Self attributes
        self._stages = stages
        self._num_stages = stages[0].num_stages
        self.pp_group_size = stages[0].group_size
        self.rank = stages[0].group_rank
        # Set the pipeline stage states
        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size, self._num_stages
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank

        # Set the same has_backward flag for stage object
        # for stage in self._stages:
        #     stage.has_backward = self._has_backward

        self._stages_initialized = False

        # avoid putting a reference to 'self' inside the lambda, it creates a ref cycle
        # has_loss: bool = self._loss_fn is not None
        # self._should_compute_loss = lambda stage: stage.is_last and has_loss

        self.n_local_stages = 2
        assert len(stages) == 2, "Interleaved 1F1B requires exactly 2 stages per rank."
        self.rank = stages[0].group_rank
        # self.number_of_rounds = max(1, n_microbatches // self.pp_group_size)
        # self.microbatches_per_round = n_microbatches // self.number_of_rounds
        # if n_microbatches % self.number_of_rounds != 0:
        #     raise ValueError(
        #         "Interleaved 1F1B requires the number of microbatches to be a "
        #         f"multiple of the number of rounds ({self.number_of_rounds}), "
        #         f"but got {n_microbatches}."
        #     )

        self.reset_n_microbatches(n_microbatches)

        self.stage_index_to_group_rank: dict[int, int] = {}
        for stage_idx in range(self._num_stages):
            self.stage_index_to_group_rank[stage_idx] = stage_idx % self.pp_group_size

        target_pp_group_rank = (self.pp_group_size - 1) - self.rank  # rank in the pp group
        target_global_rank = torch.distributed.get_global_rank(
            stages[0].group, target_pp_group_rank
        )  # global rank
        self.gpu_activation_offloader = GPUActivationOffloader(
            target_device=torch.device(f"cuda:{target_global_rank}"),
        )
        self.use_gpu_activation_offload = False

    def configure_pipeline_order(self):
        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: dict[int, list[Optional[PipelineAction]]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        self.pipeline_order = pad_to_max_length(self.pipeline_order)
        self.pipeline_order = configure_offload(self.pipeline_order)

    def _initialize_stages(self, args_mb: tuple[Any, ...], kwargs_mb):
        # init the first collective in the PP group
        # if the P2P API is the first collective call in the ``group``
        # passed to ``dist.P2POp``, all ranks of the ``group`` must participate in
        # this API call
        dummy = torch.tensor(1.0).to(torch.cuda.current_device())
        dist.all_reduce(dummy, group=self._stages[0].group)
        assert dummy.item() == self.pp_group_size

        self._stages_initialized = True

    def step(
        self, *args, target: Optional[torch.Tensor] = None, forward_only: bool = False, **kwargs
    ) -> List[List[Optional[LMOutputWithLoss]]]:
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
        self._step_microbatches(args_split, kwargs_split, targets_split, forward_only=forward_only)

        outputs: List[List[Optional[LMOutputWithLoss]]] = [
            stage.get_stage_outputs_as_list() for stage in self._stages
        ]

        return outputs

    def _split_inputs(
        self,
        args,
        kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Splits a full-batch input into chunks (i.e. microbatches) and returns
        the chunks
        """
        if args is not None or kwargs is not None:
            from torch.distributed.pipelining.microbatch import (
                split_args_kwargs_into_chunks,
            )

            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self._n_microbatches,
                self._args_chunk_spec,
                self._kwargs_chunk_spec,
            )
            # TODO: implement our own version
            return args_split, kwargs_split
        else:
            # Empty inputs (e.g. when called on middle stages)
            # Return a list of empty tuples/dicts with matching length as chunks
            return [()] * self._n_microbatches, [{}] * self._n_microbatches

    def prepare_step(self, global_batch_size: int, seqlen: int):
        for stage in self._stages:
            micro_batch_size = global_batch_size // self._n_microbatches
            stage.prepare_step(global_batch_size, micro_batch_size, seqlen)

    def clear_step_info(self):
        for stage in self._stages:
            stage.clear_step_info()

    def _step_microbatches(
        self,
        arg_mbs: list,
        kwarg_mbs: list,
        target_mbs: Optional[list] = None,
        forward_only: bool = False,
    ):
        if not self._stages_initialized:
            self._initialize_stages(arg_mbs[0], kwarg_mbs[0])

        # the microbatch shape might change between training and eval, so ned to re-prepare the meta info
        for stage in self._stages:
            stage._prepare_forward_backward_meta(self._n_microbatches, arg_mbs[0], kwarg_mbs[0])

        stage_index_to_stage: dict[int, CustomPipelineStage] = {
            stage.stage_index: stage for stage in self._stages
        }

        # determine prev_rank and next_rank based on which ranks are next to
        # the stages in the pipeline_order
        all_prev_ranks: set[int] = set()
        all_next_ranks: set[int] = set()
        for stage_index in stage_index_to_stage.keys():
            if stage_index > 0:
                all_prev_ranks.add(self.stage_index_to_group_rank[stage_index - 1])
            if stage_index < self._num_stages - 1:
                all_next_ranks.add(self.stage_index_to_group_rank[stage_index + 1])

        # count either full_backward or backward_weight together, to determine when to sync DP grads
        backward_counter: Counter[int] = Counter()
        forward_counter: Counter[int] = Counter()
        handles = []
        past_first_backward = False
        # reload_event: Optional[torch.cuda.Event] = None
        for time_step, action in enumerate(self.pipeline_order[self.rank]):
            # print(f'{action}-Start')

            # do a 1-step lookahead prefetch if needed
            next_action = (
                self.pipeline_order[self.rank][time_step + 1]
                if time_step + 1 < len(self.pipeline_order[self.rank])
                else None
            )
            if (
                next_action is not None
                and next_action.computation_type == PipelineActionType.FULL_BACKWARD
                and next_action.need_reload
                and self.use_gpu_activation_offload
            ):
                debug_mem_before_reload = torch.cuda.memory_allocated() / (1024**3)
                _ = self.gpu_activation_offloader.async_reload(
                    f"{next_action.stage_index}F{next_action.microbatch_index}"
                )  # in saving, using "F" group for both F and B
                debug_mem_after_reload = torch.cuda.memory_allocated() / (1024**3)

            ops: list[dist.P2POp] = []
            if action is not None:
                computation_type = action.computation_type
                mb_index = action.microbatch_index
                stage_index = action.stage_index
                assert (
                    mb_index is not None
                ), "All currently supported action types require valid microbatch_index"
                if computation_type == PipelineActionType.FORWARD:
                    # perform forward computation
                    stage = stage_index_to_stage[stage_index]
                    offload_group = f"{action.stage_index}F{mb_index}"
                    forward_counter[stage_index] += 1
                    last_forward = forward_counter[stage_index] == self._n_microbatches
                    with nvtx.annotate(f"{action.stage_index}F{mb_index}", color="green"):
                        # use this context manager to capture all saved tensors in this block, it does not transfer anything at this point
                        with self.gpu_activation_offloader.get_offload_context(
                            group=offload_group,
                            enable=action.need_offload and self.use_gpu_activation_offload,
                        ):
                            stage.forward_one_chunk(
                                mb_index,
                                arg_mbs[mb_index],
                                kwarg_mbs[mb_index],
                                last_forward=last_forward,
                            )
                        # start D2D transfer
                        if self.use_gpu_activation_offload:
                            debug_mem_before_offload = torch.cuda.memory_allocated() / (1024**3)
                            _ = self.gpu_activation_offloader.async_offload(offload_group)
                            debug_mem_after_offload = torch.cuda.memory_allocated() / (1024**3)
                    # self._maybe_compute_loss(stage, output, target_mbs, mb_index)
                    ops.extend(stage.get_fwd_send_ops(mb_index))
                elif computation_type == PipelineActionType.FULL_BACKWARD:
                    if forward_only:
                        # in forward only (eval) mode, skip backward computation, but need to free fwd_cache wihch
                        # is supposed to be freed in backward pass
                        stage = stage_index_to_stage[stage_index]
                        stage.fwd_cache.pop(mb_index)
                    else:
                        # make sure the reload is done
                        if action.need_reload and self.use_gpu_activation_offload:
                            self.gpu_activation_offloader.wait_reload(
                                f"{action.stage_index}F{action.microbatch_index}"
                            )  # in saving, using "F" group for both F and B

                        # perform backward computation
                        stage = stage_index_to_stage[stage_index]

                        backward_counter[stage_index] += 1
                        last_backward = backward_counter[stage_index] == self._n_microbatches
                        # grad_scale_factor = (
                        #     self._n_microbatches if self.scale_grads else 1
                        # )
                        with nvtx.annotate(f"{action.stage_index}B{mb_index}", color="red"):
                            stage.backward_one_chunk(
                                mb_index,
                                loss=None,  # loss is retrieved inside the stage
                                last_backward=last_backward,
                            )
                        # if last_backward:
                        #     stage.scale_grads(grad_scale_factor)

                        if action.need_reload and self.use_gpu_activation_offload:
                            debug_mem_before_release = torch.cuda.memory_allocated() / (1024**3)
                            self.gpu_activation_offloader.manual_release_group(
                                f"{action.stage_index}F{action.microbatch_index}"
                            )
                            debug_mem_after_release = torch.cuda.memory_allocated() / (1024**3)
                        ops.extend(stage.get_bwd_send_ops(mb_index))
                        past_first_backward = True
                elif computation_type == PipelineActionType.FULL_BACKWARD_CONT:
                    # continuation of full backward, no computation
                    pass
                else:
                    raise ValueError(f"Unknown computation type {computation_type}")
            else:
                # No operation for this time step
                pass

            # Look at the neighboring ranks for this current timestep and determine whether
            # this current rank needs to do any recv communication
            for prev_rank in all_prev_ranks:
                prev_rank_ops = self.pipeline_order[prev_rank]
                if time_step < len(prev_rank_ops):
                    prev_rank_action = prev_rank_ops[time_step]
                else:
                    prev_rank_action = None  # no action from previous rank at this time step

                if prev_rank_action is not None:  # previous rank has an action at this time step
                    computation_type = prev_rank_action.computation_type
                    mb_index = prev_rank_action.microbatch_index
                    stage_index = prev_rank_action.stage_index
                    assert (
                        mb_index is not None
                    ), "All currently supported action types require valid microbatch_index"
                    # Only handle sends for the forward from a previous rank
                    if computation_type == PipelineActionType.FORWARD:
                        # If not the last stage, then receive fwd activations
                        if stage_index + 1 in stage_index_to_stage:
                            stage = stage_index_to_stage[stage_index + 1]
                            ops.extend(stage.get_fwd_recv_ops(mb_index))
                    elif computation_type == PipelineActionType.FULL_BACKWARD:
                        # Previous rank doing backward has no influence for the current rank forward recv
                        pass
                    elif computation_type == PipelineActionType.FULL_BACKWARD_CONT:
                        pass
                    else:
                        raise ValueError(f"Unknown computation type {computation_type}")

            # Now look at the next ranks for this current timestep and determine whether
            # this current rank needs to do any recv communication (for backward gradients)
            for next_rank in all_next_ranks:
                next_rank_ops = self.pipeline_order[next_rank]

                if time_step < len(next_rank_ops):
                    next_rank_action = next_rank_ops[time_step]
                else:
                    next_rank_action = None  # no action from next rank at this time step

                if next_rank_action is not None:
                    computation_type = next_rank_action.computation_type
                    mb_index = next_rank_action.microbatch_index
                    stage_index = next_rank_action.stage_index
                    assert (
                        mb_index is not None
                    ), "All currently supported action types require valid microbatch_index"
                    # Only handle receives for the backwards from a next rank
                    if computation_type == FORWARD:
                        # Next rank doing forward or weight update has no influence for the current rank backward recv
                        pass
                    elif computation_type == FULL_BACKWARD_CONT:
                        # If not the first stage, then receive bwd gradients
                        # if stage_index - 1 in stage_index_to_stage:
                        #     stage = stage_index_to_stage[stage_index - 1]
                        #     ops.extend(stage.get_bwd_recv_ops(mb_index))
                        pass
                    elif computation_type == FULL_BACKWARD:
                        if forward_only:
                            # don't need backward recv in forward only mode
                            pass
                        else:
                            # If not the first stage, then receive bwd gradients
                            if stage_index - 1 in stage_index_to_stage:
                                stage = stage_index_to_stage[stage_index - 1]
                                ops.extend(stage.get_bwd_recv_ops(mb_index))
                    else:
                        raise ValueError(f"Unknown computation type {computation_type}")

            # at the end of step N, wait for the N-1 communication to finish, because N+1 compute may use the data from N-1 communication (F-B-F)
            if handles:
                for handle in handles:
                    handle.wait()
                handles.clear()

            if ops:
                handles = dist.batch_isend_irecv(ops)

            # do the communication
            if handles:
                if forward_only:
                    # in forward only mode, just wait right away for simplicity
                    for handle in handles:
                        handle.wait()
                    handles.clear()
                else:
                    # in training mode, we can do 1-step lookahead to overlap communication with computation in some cases
                    if (
                        not past_first_backward
                    ):  # it's only safe collect the p2p handles at N+1 step in the stable 1F1B phase. Need to collect it immediately in the warmup phase
                        for handle in handles:
                            handle.wait()
                        handles.clear()

            # print(f'{action}-Done')

            pass  # time step done

        return

    def _calculate_single_rank_operations(self, rank) -> list[Optional[PipelineAction]]:
        def get_rank_warmup_ops(rank):
            # Warms up operations for last stage
            warmups_ops_last_stage = (self.n_local_stages - 1) * self.microbatches_per_round

            # warmups_ops_last_stage += 1 # fused overlap

            # Increment warmup operations by 2 for each hop away from the last stage
            multiply_factor = 2
            warmup_ops = warmups_ops_last_stage + multiply_factor * (
                (self.pp_group_size - 1) - rank
            )

            # We cannot have more warmup operations than there are number of microbatches, so cap it there
            return min(warmup_ops, self._n_microbatches * self.n_local_stages)

        warmup_ops = get_rank_warmup_ops(rank)
        microbatch_ops = self.n_local_stages * self._n_microbatches
        # fwd_bwd_ops should encompass the remaining forwards
        fwd_bwd_ops = microbatch_ops - warmup_ops
        # cooldown_ops should encompass the remaining backwards
        cooldown_ops = microbatch_ops - fwd_bwd_ops
        # total ops encompass both forward and backward ops
        total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops
        # warmup_ops + fwd_bwd_ops * 2 + cooldown_ops == microbatch_ops * 2

        # Calculates the stage index based on step and pp_group_size
        def forward_stage_index(step):
            # Get the local index from 0 to n_local_stages-1
            local_index = (step // self.microbatches_per_round) % self.n_local_stages
            return (local_index * self.pp_group_size) + rank

        def backward_stage_index(step):
            local_index = (
                self.n_local_stages
                - 1
                - ((step - warmup_ops) // self.microbatches_per_round) % self.n_local_stages
            )
            return (local_index * self.pp_group_size) + rank

        rank_ops = _get_interleaved_1f1b_rank_ops(
            self.n_local_stages,
            self.pp_group_size,
            warmup_ops,
            fwd_bwd_ops,
            cooldown_ops,
            rank,
            forward_stage_index,
            backward_stage_index,
        )

        return rank_ops


def configure_offload(rank_pipeline_order: dict[int, list[Optional[PipelineAction]]]):
    """
    This function configures activation offloading schedule to keep the number of held activations under a limit.
    """
    total_ranks = len(rank_pipeline_order)
    total_steps = len(rank_pipeline_order[0])

    def find_corresponding_backward(pp_order, fwd_action) -> Optional[int]:
        for time_step, action in enumerate(pp_order):
            if (
                action is not None
                and action.stage_index == fwd_action.stage_index
                and action.microbatch_index == fwd_action.microbatch_index
                and action.computation_type == PipelineActionType.FULL_BACKWARD
            ):
                return time_step
        return None

    for rank in range(total_ranks):
        allowed_offloads = total_ranks - 1 - rank * 2

        if allowed_offloads <= 0:
            continue

        held_activations = 0
        offloaded_activations = 0
        for time_step in range(total_steps):
            action = rank_pipeline_order[rank][time_step]

            # if next action is a backward that requires reload, then activation +1 when this forward is done
            next_action = None
            if time_step + 1 < total_steps:
                next_action = rank_pipeline_order[rank][time_step + 1]
            if (
                next_action is not None
                and next_action.computation_type == PipelineActionType.FULL_BACKWARD
                and next_action.need_reload
            ):
                held_activations += 1
                offloaded_activations -= 1

            # forward pass, activation +1
            if action is not None and action.computation_type == PipelineActionType.FORWARD:
                held_activations += 1

            # backward pass, activation -1
            if (
                action is not None
                and action.computation_type == PipelineActionType.FULL_BACKWARD_CONT
            ):
                held_activations -= 1

            if offloaded_activations < allowed_offloads:
                if action is not None and action.computation_type == PipelineActionType.FORWARD:
                    rank_pipeline_order[rank][time_step] = action._replace(need_offload=True)
                    offloaded_activations += 1
                    held_activations -= 1
                    # set the corresponding backward to need_reload
                    bwd_step = find_corresponding_backward(rank_pipeline_order[rank], action)
                    if bwd_step is not None:
                        bwd_action = rank_pipeline_order[rank][bwd_step]
                        assert bwd_action is not None
                        rank_pipeline_order[rank][bwd_step] = bwd_action._replace(need_reload=True)
        assert held_activations == 0
        assert offloaded_activations == 0
    return rank_pipeline_order


def pad_to_max_length(rank_pipeline_order: dict[int, list[Optional[PipelineAction]]]):
    """Pads all rank operation lists to the maximum length with None (no-op) actions."""
    max_length = max(len(ops) for ops in rank_pipeline_order.values())
    for rank, ops in rank_pipeline_order.items():
        if len(ops) < max_length:
            ops.extend([None] * (max_length - len(ops)))
    return rank_pipeline_order


def _get_interleaved_1f1b_rank_ops(
    n_local_stages,
    pp_group_size,
    warmup_ops,
    fwd_bwd_ops,
    cooldown_ops,
    rank,
    forward_stage_index,
    backward_stage_index,
    num_1f1b_microbatches=0,
):
    # All stages start with handling microbatch 0
    fwd_stage_mb_index: dict[int, int] = defaultdict(int)
    bwd_stage_mb_index: dict[int, int] = defaultdict(int)

    # Store the list of operations used for that rank
    # Pre-padding, rank starts with no-ops based on the warmup.
    rank_ops: list[Optional[PipelineAction]] = [None for _ in range(rank)]
    # These are used to calculate the number of slots to fill with no-ops, to account for the delay in warmup
    # when we want to wait for the backward to trickle back up and start 1f1b to align all ranks.
    # Formula:
    # pre-padding + warmup_ops + post_warmup_ops = earliest time step of first backward
    # post_warmup_ops = [earliest time step of first backward] - (warmup_ops + pre-padding)
    # earliest time step of first backward = [local_stages * group_size + 2 * (group_size - 1 - rank)]
    # warmup_ops = calculated above
    post_warmup_ops = 2 * (pp_group_size - 1 - rank)

    total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops

    for op in range(total_ops):
        # Warmup phase
        if op < warmup_ops:
            fwd_stage_index = forward_stage_index(op)
            # This will assign the current microbatch index and update it as well
            fwd_stage_mb_index[fwd_stage_index] = (
                mb_index := fwd_stage_mb_index[fwd_stage_index]
            ) + 1
            rank_ops.append(PipelineAction(fwd_stage_index, PipelineActionType.FORWARD, mb_index))
            if op == warmup_ops - 1:
                # This is the last step in the warmup phase, so we need to wait for the backward to trickle back up
                rank_ops.extend([None] * post_warmup_ops)
        # 1F1B Phase (forward and backward)
        elif warmup_ops <= op < warmup_ops + fwd_bwd_ops:
            # 1F
            fwd_stage_index = forward_stage_index(op)
            fwd_stage_mb_index[fwd_stage_index] = (
                fwd_mb_index := fwd_stage_mb_index[fwd_stage_index]
            ) + 1
            rank_ops.append(
                PipelineAction(fwd_stage_index, PipelineActionType.FORWARD, fwd_mb_index)
            )

            # 1B
            bwd_stage_index = backward_stage_index(op)
            bwd_stage_mb_index[bwd_stage_index] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
            ) + 1
            rank_ops.append(PipelineAction(bwd_stage_index, FULL_BACKWARD, bwd_mb_index))
            rank_ops.append(
                PipelineAction(bwd_stage_index, FULL_BACKWARD_CONT, bwd_mb_index)
            )  # Backward takes twice the time

        # Cooldown phase
        else:
            # During cooldown phase, we need steps to align with 1f1b happening in other ranks
            cooldown_idx = op - (warmup_ops + fwd_bwd_ops)
            if cooldown_idx < (pp_group_size - 1 - rank):
                rank_ops.append(None)

            bwd_stage_index = backward_stage_index(op)
            bwd_stage_mb_index[bwd_stage_index] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage_index]
            ) + 1
            rank_ops.append(PipelineAction(bwd_stage_index, FULL_BACKWARD, bwd_mb_index))
            rank_ops.append(
                PipelineAction(bwd_stage_index, FULL_BACKWARD_CONT, bwd_mb_index)
            )  # Backward takes twice the time

    return rank_ops
