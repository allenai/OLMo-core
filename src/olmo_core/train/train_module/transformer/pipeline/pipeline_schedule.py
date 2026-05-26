import logging
import os
import re
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Union

import nvtx
import torch
import torch.distributed as dist

from olmo_core.nn.lm_head import LMOutputWithLoss

from .gpu_activation_offload import GPUActivationOffloader
from .helpers import generate_stage_to_rank_mapping
from .p2p_transport import NCCLRMAPipelineP2PTransport
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
    placement_style = "loop"
    enable_activation_offload_schedule = True
    enable_p2p_overlap = True
    p2p_overlap_kinds = frozenset(("F", "B"))
    p2p_overlap_ops = frozenset(("F_SEND", "F_RECV", "B_SEND", "B_RECV"))
    max_p2p_overlap_steps: Optional[int] = 1
    prioritize_next_action_input_p2p = False

    @staticmethod
    def _action_advances_p2p_overlap(action: Optional[PipelineAction]) -> bool:
        return action is not None and action.computation_type in (
            PipelineActionType.FORWARD,
            PipelineActionType.FULL_BACKWARD,
        )

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
        args_chunk_spec: Optional[Any] = None,
        kwargs_chunk_spec: Optional[dict[str, Any]] = None,
        output_merge_spec: Optional[Union[dict[str, Any], tuple[Any]]] = None,
        forward_pull_ahead_extra_activations: int | Sequence[int] | Mapping[int, int] = 0,
    ):
        self.pp_group_size = stages[0].group_size

        # Chunking specification for positional inputs. (default: `None`)
        self._args_chunk_spec = args_chunk_spec
        # Chunking specification for keyword inputs. (default: `None`)
        self._kwargs_chunk_spec = kwargs_chunk_spec
        self._output_merge_spec = output_merge_spec
        self.forward_pull_ahead_extra_activations = forward_pull_ahead_extra_activations

        logger.info("Using %s", self.__class__.__name__)

        # Self attributes
        self._stages = stages
        self._num_stages = stages[0].num_stages
        self.pp_group_size = stages[0].group_size
        self.rank = stages[0].group_rank
        self.uses_separate_p2p_group = any(stage.p2p_group is not stage.group for stage in stages)
        self.p2p_backend = getattr(stages[0], "p2p_backend", "nccl")
        if any(getattr(stage, "p2p_backend", "nccl") != self.p2p_backend for stage in stages):
            raise RuntimeError("All local pipeline stages must use the same P2P backend")
        self.p2p_transport: Optional[NCCLRMAPipelineP2PTransport] = None
        if self.p2p_backend == "nccl_rma":
            self.p2p_transport = NCCLRMAPipelineP2PTransport(
                group=stages[0].p2p_group,
                device=stages[0].device,
                num_stages=stages[0].num_stages,
            )
            for stage in stages:
                stage.set_p2p_transport(self.p2p_transport)
        # Set the pipeline stage states
        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size, self._num_stages, style=self.placement_style
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

        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size, self._num_stages, style=self.placement_style
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank

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
        self.pipeline_order_source = "interleaved_1f1b_loop"
        self.pipeline_order: dict[int, list[Optional[PipelineAction]]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

        self.pipeline_order = pad_to_max_length(self.pipeline_order)
        if self.enable_activation_offload_schedule:
            self.pipeline_order = configure_offload(self.pipeline_order)

    def _initialize_stages(self, args_mb: tuple[Any, ...], kwargs_mb):
        # init the first collective in the PP group
        # if the P2P API is the first collective call in the ``group``
        # passed to ``dist.P2POp``, all ranks of the ``group`` must participate in
        # this API call
        dummy = torch.tensor(1.0).to(torch.cuda.current_device())
        dist.all_reduce(dummy, group=self._stages[0].group)
        assert dummy.item() == self.pp_group_size
        if self.uses_separate_p2p_group:
            dummy = torch.tensor(1.0).to(torch.cuda.current_device())
            dist.all_reduce(dummy, group=self._stages[0].p2p_group)
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
            if self._args_chunk_spec is not None or self._kwargs_chunk_spec is not None:
                raise NotImplementedError(
                    "Custom chunk specs are not supported by this pipeline schedule."
                )

            args = args or ()
            kwargs = kwargs or {}

            if len(args) > 1:
                raise ValueError(f"Expected zero or one positional tensor arg, got {len(args)}")

            input_ids: Optional[torch.Tensor] = None
            if len(args) == 1:
                input_ids = args[0]
                if not isinstance(input_ids, torch.Tensor) or input_ids.dim() == 0:
                    raise TypeError(
                        "args must be empty or a tuple containing one non-scalar tensor"
                    )
                if input_ids.size(0) < self._n_microbatches:
                    raise ValueError(
                        f"input batch size {input_ids.size(0)} is smaller than num_microbatches={self._n_microbatches}"
                    )
                input_id_chunks = list(torch.tensor_split(input_ids, self._n_microbatches, dim=0))
                args_split: List[tuple] = [(chunk,) for chunk in input_id_chunks]
            else:
                args_split = [()] * self._n_microbatches
            kwargs_split: List[dict] = [{} for _ in range(self._n_microbatches)]

            supported_keys = {
                "loss_div_factor",
                "labels",
                "ignore_index",
                "loss_reduction",
                "z_loss_multiplier",
                "return_logits",
            }
            unexpected_keys = set(kwargs) - supported_keys
            if unexpected_keys:
                raise ValueError(
                    f"Unsupported kwargs for pipeline splitting: {sorted(unexpected_keys)}"
                )

            for key, value in kwargs.items():
                if key == "labels":
                    if not isinstance(value, torch.Tensor) or value.dim() == 0:
                        raise TypeError("'labels' must be a non-scalar tensor")
                    if value.size(0) < self._n_microbatches:
                        raise ValueError(
                            f"'labels' batch size {value.size(0)} is smaller than num_microbatches={self._n_microbatches}"
                        )
                    if input_ids is not None and value.size(0) != input_ids.size(0):
                        raise ValueError(
                            f"'labels' batch size {value.size(0)} does not match input batch size {input_ids.size(0)}"
                        )
                    value_chunks = list(torch.tensor_split(value, self._n_microbatches, dim=0))
                    for i, chunk in enumerate(value_chunks):
                        kwargs_split[i][key] = chunk
                elif key == "loss_div_factor":
                    if isinstance(value, torch.Tensor):
                        if value.dim() != 0:
                            raise TypeError("'loss_div_factor' must be a scalar tensor")
                    elif not isinstance(value, (int, float)):
                        raise TypeError("'loss_div_factor' must be a scalar tensor, int, or float")
                    for kwarg_mb in kwargs_split:
                        kwarg_mb[key] = value
                elif key == "ignore_index":
                    if not isinstance(value, int):
                        raise TypeError("'ignore_index' must be an int")
                    for kwarg_mb in kwargs_split:
                        kwarg_mb[key] = value
                elif key == "loss_reduction":
                    if not isinstance(value, str):
                        raise TypeError("'loss_reduction' must be a str")
                    for kwarg_mb in kwargs_split:
                        kwarg_mb[key] = value
                elif key == "z_loss_multiplier":
                    if value is not None and not isinstance(value, (int, float)):
                        raise TypeError("'z_loss_multiplier' must be None, int, or float")
                    for kwarg_mb in kwargs_split:
                        kwarg_mb[key] = value
                elif key == "return_logits":
                    if not isinstance(value, bool):
                        raise TypeError("'return_logits' must be a bool")
                    for kwarg_mb in kwargs_split:
                        kwarg_mb[key] = value

            return args_split, kwargs_split
        else:
            # Empty inputs (e.g. when called on middle stages)
            # Return a list of empty tuples/dicts with matching length as chunks
            assert False, "this branch not used in this version"
            return [()] * self._n_microbatches, [{}] * self._n_microbatches

    def prepare_step(self, global_batch_size: int, seqlen: int):
        for stage in self._stages:
            micro_batch_size = global_batch_size // self._n_microbatches
            stage.prepare_step(global_batch_size, micro_batch_size, seqlen)

    def clear_step_info(self):
        for stage in self._stages:
            stage.clear_step_info()

    def _maybe_local_forward_handoff(
        self,
        stage: CustomPipelineStage,
        mb_index: int,
        stage_index_to_stage: dict[int, CustomPipelineStage],
    ) -> None:
        if not stage.has_local_forward_dst():
            return
        dst_stage = stage_index_to_stage[stage.stage_index + 1]
        dst_stage.set_local_forward_input(mb_index, stage.pop_forward_handoff_tensor(mb_index))

    def _maybe_local_backward_handoff(
        self,
        stage: CustomPipelineStage,
        mb_index: int,
        stage_index_to_stage: dict[int, CustomPipelineStage],
    ) -> None:
        if not stage.has_local_backward_dst():
            return
        dst_stage = stage_index_to_stage[stage.stage_index - 1]
        dst_stage.set_local_backward_grad(mb_index, stage.pop_backward_handoff_tensor(mb_index))

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
        if self.p2p_transport is not None:
            payload_meta: Optional[torch.Tensor] = None
            for stage in self._stages:
                for meta in (stage.outputs_meta, stage.inputs_meta):
                    if meta is not None and meta.dtype == stage.p2p_dtype and meta.dim() == 3:
                        payload_meta = meta
                        break
                if payload_meta is not None:
                    break
            if payload_meta is None:
                raise RuntimeError("Could not infer NCCL RMA P2P payload metadata")
            self.p2p_transport.prepare_step(
                num_microbatches=self._n_microbatches,
                payload_shape=tuple(payload_meta.size()),
                payload_dtype=payload_meta.dtype,
            )

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
        past_first_backward = False
        # Loop interleaved 1F1B launches one tick-wide P2P batch, but waits
        # recv handles only when the next local compute consumes them. 1F1B-V
        # overrides this method to keep its peer/key launch ordering.
        pending_p2p: dict[
            tuple[str, int, int, int],
            list[tuple[Any, Any, str, int]],
        ] = defaultdict(list)
        completed_p2p_overlap_compute_steps = 0

        def wait_all_p2p() -> None:
            for key in list(pending_p2p):
                wait_p2p_key(key)

        def wait_p2p_key(key: tuple[str, int, int, int]) -> None:
            for handle, _op, _op_kind, _launch_overlap_step in pending_p2p.pop(key, []):
                handle.wait()

        def wait_p2p_ops(op_kinds: set[str] | frozenset[str]) -> None:
            for key in list(pending_p2p):
                pending = pending_p2p.pop(key)
                still_pending: list[tuple[Any, Any, str, int]] = []
                for handle, op, op_kind, launch_overlap_step in pending:
                    if op_kind in op_kinds:
                        handle.wait()
                    else:
                        still_pending.append((handle, op, op_kind, launch_overlap_step))
                if still_pending:
                    pending_p2p[key] = still_pending

        def prune_completed_p2p() -> None:
            for key, entries in list(pending_p2p.items()):
                pending: list[tuple[Any, Any, str, int]] = []
                for handle, op, op_kind, launch_overlap_step in entries:
                    if handle.is_completed():
                        handle.wait()
                    else:
                        pending.append((handle, op, op_kind, launch_overlap_step))
                if pending:
                    pending_p2p[key] = pending
                else:
                    pending_p2p.pop(key, None)

        def wait_p2p_launched_at_or_before(overlap_compute_step: int) -> None:
            for key in list(pending_p2p):
                pending = pending_p2p.pop(key)
                still_pending: list[tuple[Any, Any, str, int]] = []
                for handle, op, op_kind, launch_overlap_step in pending:
                    if launch_overlap_step <= overlap_compute_step:
                        handle.wait()
                    else:
                        still_pending.append((handle, op, op_kind, launch_overlap_step))
                if still_pending:
                    pending_p2p[key] = still_pending

        def wait_for_action_inputs(action: PipelineAction) -> None:
            assert action.microbatch_index is not None
            stage = stage_index_to_stage[action.stage_index]
            mb_index = action.microbatch_index
            if action.computation_type == PipelineActionType.FORWARD:
                if not stage.is_first and not stage.has_local_forward_src():
                    wait_p2p_key(("F", action.stage_index - 1, action.stage_index, mb_index))
            elif action.computation_type == PipelineActionType.FULL_BACKWARD:
                if not forward_only and not stage.is_last and not stage.has_local_backward_src():
                    wait_p2p_key(("B", action.stage_index + 1, action.stage_index, mb_index))

        def launch_p2p_ops(keyed_ops: list[tuple[tuple[str, int, int, int], str, Any]]) -> None:
            if not keyed_ops:
                return
            with nvtx.annotate("P2P", color="blue"):
                if self.p2p_backend == "nccl_rma":
                    handles = [op.start() for _key, _op_kind, op in keyed_ops]
                else:
                    handles = dist.batch_isend_irecv([op for _key, _op_kind, op in keyed_ops])
            if len(handles) == 1 and len(keyed_ops) > 1:
                handles_for_ops = handles * len(keyed_ops)
            elif len(handles) == len(keyed_ops):
                handles_for_ops = handles
            else:
                raise RuntimeError(
                    "Unexpected number of P2P work handles from batch_isend_irecv: "
                    f"got {len(handles)} handles for {len(keyed_ops)} ops"
                )
            for (key, op_kind, op), handle in zip(keyed_ops, handles_for_ops):
                pending_p2p[key].append(
                    (
                        handle,
                        op,
                        op_kind,
                        completed_p2p_overlap_compute_steps,
                    )
                )

        # reload_event: Optional[torch.cuda.Event] = None
        for time_step, action in enumerate(self.pipeline_order[self.rank]):
            # print(f'{action}-Start')
            prune_completed_p2p()

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
                # debug_mem_before_reload = torch.cuda.memory_allocated() / (1024**3)
                _ = self.gpu_activation_offloader.async_reload(
                    f"{next_action.stage_index}F{next_action.microbatch_index}"
                )  # in saving, using "F" group for both F and B
                # debug_mem_after_reload = torch.cuda.memory_allocated() / (1024**3)

            keyed_ops: list[tuple[tuple[str, int, int, int], str, Any]] = []
            if action is not None:
                wait_for_action_inputs(action)
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
                            # debug_mem_before_offload = torch.cuda.memory_allocated() / (1024**3)
                            _ = self.gpu_activation_offloader.async_offload(offload_group)
                            # debug_mem_after_offload = torch.cuda.memory_allocated() / (1024**3)
                    # self._maybe_compute_loss(stage, output, target_mbs, mb_index)
                    self._maybe_local_forward_handoff(stage, mb_index, stage_index_to_stage)
                    keyed_ops.extend(
                        (
                            ("F", stage_index, stage_index + 1, mb_index),
                            "F_SEND",
                            op,
                        )
                        for op in stage.get_fwd_send_ops(mb_index)
                    )
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
                            # debug_mem_before_release = torch.cuda.memory_allocated() / (1024**3)
                            self.gpu_activation_offloader.manual_release_group(
                                f"{action.stage_index}F{action.microbatch_index}"
                            )
                            # debug_mem_after_release = torch.cuda.memory_allocated() / (1024**3)
                        self._maybe_local_backward_handoff(stage, mb_index, stage_index_to_stage)
                        keyed_ops.extend(
                            (
                                ("B", stage_index, stage_index - 1, mb_index),
                                "B_SEND",
                                op,
                            )
                            for op in stage.get_bwd_send_ops(mb_index)
                        )
                        past_first_backward = True
                elif computation_type == PipelineActionType.FULL_BACKWARD_CONT:
                    # continuation of full backward, no computation
                    pass
                else:
                    raise ValueError(f"Unknown computation type {computation_type}")
            else:
                # No operation for this time step
                pass

            # FULL_BACKWARD_CONT is only a schedule placeholder for the second
            # half of backward. It does not launch model work, so do not age
            # outstanding P2P against the overlap budget for that slot.
            if self._action_advances_p2p_overlap(action):
                completed_p2p_overlap_compute_steps += 1

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
                            keyed_ops.extend(
                                (
                                    ("F", stage_index, stage_index + 1, mb_index),
                                    "F_RECV",
                                    op,
                                )
                                for op in stage.get_fwd_recv_ops(mb_index)
                            )
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
                                keyed_ops.extend(
                                    (
                                        ("B", stage_index, stage_index - 1, mb_index),
                                        "B_RECV",
                                        op,
                                    )
                                    for op in stage.get_bwd_recv_ops(mb_index)
                                )
                    else:
                        raise ValueError(f"Unknown computation type {computation_type}")

            launch_p2p_ops(keyed_ops)

            # do the communication
            if pending_p2p:
                if forward_only:
                    # in forward only mode, just wait right away for simplicity
                    wait_all_p2p()
                elif not self.enable_p2p_overlap:
                    wait_all_p2p()
                else:
                    has_blocking_op = any(
                        op_kind not in self.p2p_overlap_ops for _key, op_kind, _op in keyed_ops
                    )
                    if (
                        not past_first_backward
                    ):  # it's only safe collect the p2p handles at N+1 step in the stable 1F1B phase. Need to collect it immediately in the warmup phase
                        wait_all_p2p()
                    elif has_blocking_op:
                        wait_all_p2p()
                    if self.max_p2p_overlap_steps is not None:
                        wait_p2p_launched_at_or_before(
                            completed_p2p_overlap_compute_steps - self.max_p2p_overlap_steps
                        )

            # print(f'{action}-Done')

            pass  # time step done

        wait_all_p2p()

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
        # total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops
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


class CustomSchedule1F1BV(CustomScheduleInterleaved1F1B):
    placement_style = "v"
    enable_activation_offload_schedule = False
    enable_p2p_overlap = True
    p2p_overlap_kinds = frozenset(("F",))
    p2p_overlap_ops = frozenset(("F_SEND", "F_RECV", "B_RECV"))
    max_p2p_overlap_steps = None
    prioritize_next_action_input_p2p = True
    _symbol_pattern_size = 6
    forward_pull_ahead_extra_activations: int | Sequence[int] | Mapping[int, int] = 0

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
        if self.p2p_transport is not None:
            payload_meta: Optional[torch.Tensor] = None
            for stage in self._stages:
                for meta in (stage.outputs_meta, stage.inputs_meta):
                    if meta is not None and meta.dtype == stage.p2p_dtype and meta.dim() == 3:
                        payload_meta = meta
                        break
                if payload_meta is not None:
                    break
            if payload_meta is None:
                raise RuntimeError("Could not infer NCCL RMA P2P payload metadata")
            self.p2p_transport.prepare_step(
                num_microbatches=self._n_microbatches,
                payload_shape=tuple(payload_meta.size()),
                payload_dtype=payload_meta.dtype,
            )

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
        past_first_backward = False
        outstanding_p2p: dict[
            tuple[str, int, int, int],
            list[tuple[Any, Any, str, int]],
        ] = defaultdict(list)
        completed_p2p_overlap_compute_steps = 0

        def debug(message: str) -> None:
            if self.p2p_backend != "nccl_rma" or os.environ.get("OLMO_NCCL_RMA_P2P_DEBUG") != "1":
                return
            print(f"[rank {self.rank} pipeline-rma] {message}", flush=True)

        def p2p_launch_label(
            key: tuple[str, int, int, int],
            peer_keyed_ops: list[tuple[tuple[str, int, int, int], str, Any]],
        ) -> str:
            kind, src_stage, dst_stage, mb_index = key
            labels = set()
            for _key, op_kind, _op in peer_keyed_ops:
                if op_kind.endswith("SEND"):
                    labels.add(f"{src_stage}{kind}{mb_index}-S")
                else:
                    labels.add(f"{dst_stage}{kind}{mb_index}-R")
            return ",".join(sorted(labels))

        def wait_p2p_key(key: tuple[str, int, int, int]) -> None:
            debug(f"wait key={key} entries={len(outstanding_p2p.get(key, []))}")
            for handle, _op, _op_kind, _launch_overlap_step in outstanding_p2p.pop(key, []):
                handle.wait()

        def wait_all_p2p() -> None:
            for key in list(outstanding_p2p):
                wait_p2p_key(key)

        def wait_p2p_ops(op_kinds: set[str] | frozenset[str]) -> None:
            for key in list(outstanding_p2p):
                pending = outstanding_p2p.pop(key)
                still_pending: list[tuple[Any, Any, str, int]] = []
                for handle, op, op_kind, launch_overlap_step in pending:
                    if op_kind in op_kinds:
                        handle.wait()
                    else:
                        still_pending.append((handle, op, op_kind, launch_overlap_step))
                if still_pending:
                    outstanding_p2p[key] = still_pending

        def prune_completed_p2p() -> None:
            for key, entries in list(outstanding_p2p.items()):
                pending: list[tuple[Any, Any, str, int]] = []
                for handle, op, op_kind, launch_overlap_step in entries:
                    if handle.is_completed():
                        handle.wait()
                    else:
                        pending.append((handle, op, op_kind, launch_overlap_step))
                if pending:
                    outstanding_p2p[key] = pending
                else:
                    outstanding_p2p.pop(key, None)

        def wait_p2p_launched_at_or_before(overlap_compute_step: int) -> None:
            for key in list(outstanding_p2p):
                pending = outstanding_p2p.pop(key)
                still_pending: list[tuple[Any, Any, str, int]] = []
                for handle, op, op_kind, launch_overlap_step in pending:
                    if launch_overlap_step <= overlap_compute_step:
                        handle.wait()
                    else:
                        still_pending.append((handle, op, op_kind, launch_overlap_step))
                if still_pending:
                    outstanding_p2p[key] = still_pending

        def wait_for_action_inputs(action: PipelineAction) -> None:
            assert action.microbatch_index is not None
            stage = stage_index_to_stage[action.stage_index]
            mb_index = action.microbatch_index
            if action.computation_type == PipelineActionType.FORWARD:
                if not stage.is_first and not stage.has_local_forward_src():
                    wait_p2p_key(("F", action.stage_index - 1, action.stage_index, mb_index))
            elif action.computation_type == PipelineActionType.FULL_BACKWARD:
                if not forward_only and not stage.is_last and not stage.has_local_backward_src():
                    wait_p2p_key(("B", action.stage_index + 1, action.stage_index, mb_index))

        # reload_event: Optional[torch.cuda.Event] = None
        for time_step, action in enumerate(self.pipeline_order[self.rank]):
            debug(f"time_step={time_step} action={action}")
            # print(f'{action}-Start')
            prune_completed_p2p()

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
                # debug_mem_before_reload = torch.cuda.memory_allocated() / (1024**3)
                _ = self.gpu_activation_offloader.async_reload(
                    f"{next_action.stage_index}F{next_action.microbatch_index}"
                )  # in saving, using "F" group for both F and B
                # debug_mem_after_reload = torch.cuda.memory_allocated() / (1024**3)

            keyed_ops: list[tuple[tuple[str, int, int, int], str, Any]] = []
            if action is not None:
                wait_for_action_inputs(action)
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
                            # debug_mem_before_offload = torch.cuda.memory_allocated() / (1024**3)
                            _ = self.gpu_activation_offloader.async_offload(offload_group)
                            # debug_mem_after_offload = torch.cuda.memory_allocated() / (1024**3)
                    # self._maybe_compute_loss(stage, output, target_mbs, mb_index)
                    self._maybe_local_forward_handoff(stage, mb_index, stage_index_to_stage)
                    keyed_ops.extend(
                        (
                            ("F", stage_index, stage_index + 1, mb_index),
                            "F_SEND",
                            op,
                        )
                        for op in stage.get_fwd_send_ops(mb_index)
                    )
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
                            # debug_mem_before_release = torch.cuda.memory_allocated() / (1024**3)
                            self.gpu_activation_offloader.manual_release_group(
                                f"{action.stage_index}F{action.microbatch_index}"
                            )
                            # debug_mem_after_release = torch.cuda.memory_allocated() / (1024**3)
                        self._maybe_local_backward_handoff(stage, mb_index, stage_index_to_stage)
                        keyed_ops.extend(
                            (
                                ("B", stage_index, stage_index - 1, mb_index),
                                "B_SEND",
                                op,
                            )
                            for op in stage.get_bwd_send_ops(mb_index)
                        )
                        past_first_backward = True
                elif computation_type == PipelineActionType.FULL_BACKWARD_CONT:
                    # continuation of full backward, no computation
                    pass
                else:
                    raise ValueError(f"Unknown computation type {computation_type}")
            else:
                # No operation for this time step
                pass

            # FULL_BACKWARD_CONT is only a schedule placeholder for the second
            # half of backward. It does not launch model work, so do not age
            # outstanding P2P against the overlap budget for that slot.
            if self._action_advances_p2p_overlap(action):
                completed_p2p_overlap_compute_steps += 1

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
                            keyed_ops.extend(
                                (
                                    ("F", stage_index, stage_index + 1, mb_index),
                                    "F_RECV",
                                    op,
                                )
                                for op in stage.get_fwd_recv_ops(mb_index)
                            )
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
                                keyed_ops.extend(
                                    (
                                        ("B", stage_index, stage_index - 1, mb_index),
                                        "B_RECV",
                                        op,
                                    )
                                    for op in stage.get_bwd_recv_ops(mb_index)
                                )
                    else:
                        raise ValueError(f"Unknown computation type {computation_type}")

            if keyed_ops:
                keyed_ops_by_peer_and_key: dict[
                    tuple[int, tuple[str, int, int, int]],
                    list[tuple[tuple[str, int, int, int], str, Any]],
                ] = defaultdict(list)
                next_action_input_keys: set[tuple[str, int, int, int]] = set()
                if next_action is not None and next_action.microbatch_index is not None:
                    next_stage = stage_index_to_stage[next_action.stage_index]
                    next_mb_index = next_action.microbatch_index
                    if next_action.computation_type == PipelineActionType.FORWARD:
                        if not next_stage.is_first and not next_stage.has_local_forward_src():
                            next_action_input_keys.add(
                                (
                                    "F",
                                    next_action.stage_index - 1,
                                    next_action.stage_index,
                                    next_mb_index,
                                )
                            )
                    elif next_action.computation_type == PipelineActionType.FULL_BACKWARD:
                        if (
                            not forward_only
                            and not next_stage.is_last
                            and not next_stage.has_local_backward_src()
                        ):
                            next_action_input_keys.add(
                                (
                                    "B",
                                    next_action.stage_index + 1,
                                    next_action.stage_index,
                                    next_mb_index,
                                )
                            )

                for keyed_op in keyed_ops:
                    key, _op_kind, op = keyed_op
                    keyed_ops_by_peer_and_key[(op.peer, key)].append(keyed_op)

                peers = {peer for peer, _key in keyed_ops_by_peer_and_key}
                if self.prioritize_next_action_input_p2p:
                    priority_peers = {
                        peer
                        for peer, key in keyed_ops_by_peer_and_key
                        if key in next_action_input_keys
                    }
                    peer_order = sorted(
                        peers,
                        key=lambda peer: (peer not in priority_peers, peer),
                    )
                else:
                    # All ranks use the same total order over undirected PP
                    # edges. This avoids ring cycles in interleaved warmup such
                    # as 0->3->2->1->0 where every rank waits for a different
                    # peer's first batch_isend_irecv call.
                    peer_order = sorted(
                        peers,
                        key=lambda peer: (min(self.rank, peer), max(self.rank, peer)),
                    )
                for peer in peer_order:
                    keys_for_peer = sorted(
                        key for keyed_peer, key in keyed_ops_by_peer_and_key if keyed_peer == peer
                    )
                    for key in keys_for_peer:
                        peer_keyed_ops = sorted(
                            keyed_ops_by_peer_and_key[(peer, key)],
                            key=lambda item: item[1],
                        )
                        with nvtx.annotate(
                            p2p_launch_label(key, peer_keyed_ops),
                            color="blue",
                        ):
                            debug(
                                "launch "
                                f"peer={peer} key={key} "
                                f"kinds={[op_kind for _, op_kind, _ in peer_keyed_ops]}"
                            )
                            if self.p2p_backend == "nccl_rma":
                                handles = [op.start() for _, _, op in peer_keyed_ops]
                            else:
                                handles = dist.batch_isend_irecv(
                                    [op for _, _, op in peer_keyed_ops]
                                )
                        if len(handles) == 1 and len(peer_keyed_ops) > 1:
                            # NCCL coalescing may return a single Work that covers
                            # the whole same-key peer batch. Associate it with every
                            # keyed op so later dependency waits cannot skip any
                            # receive, but do not coalesce unrelated keys together:
                            # waiting on one key would otherwise force waits for
                            # independent overlappable transfers.
                            handles_for_ops = handles * len(peer_keyed_ops)
                        elif len(handles) == len(peer_keyed_ops):
                            handles_for_ops = handles
                        else:
                            raise RuntimeError(
                                "Unexpected number of P2P work handles from batch_isend_irecv: "
                                f"got {len(handles)} handles for {len(peer_keyed_ops)} ops"
                            )
                        for (key, op_kind, op), handle in zip(peer_keyed_ops, handles_for_ops):
                            outstanding_p2p[key].append(
                                (
                                    handle,
                                    op,
                                    op_kind,
                                    completed_p2p_overlap_compute_steps,
                                )
                            )

            # do the communication
            if outstanding_p2p:
                if forward_only:
                    # in forward only mode, just wait right away for simplicity
                    wait_all_p2p()
                elif not self.enable_p2p_overlap:
                    wait_all_p2p()
                else:
                    # In training mode, the overlap cap is measured in local
                    # compute steps, not raw schedule slots. Placeholder slots
                    # like FULL_BACKWARD_CONT do not consume the budget.
                    blocking_ops = {
                        "F_SEND",
                        "F_RECV",
                        "B_SEND",
                        "B_RECV",
                    } - set(self.p2p_overlap_ops)
                    if blocking_ops:
                        wait_p2p_ops(blocking_ops)
                    if (
                        not past_first_backward
                    ):  # it's only safe collect the p2p handles at N+1 step in the stable 1F1B phase. Need to collect it immediately in the warmup phase
                        wait_all_p2p()
                    if self.max_p2p_overlap_steps is not None:
                        wait_p2p_launched_at_or_before(
                            completed_p2p_overlap_compute_steps - self.max_p2p_overlap_steps
                        )

            # print(f'{action}-Done')

            pass  # time step done

        wait_all_p2p()

        return

    def configure_pipeline_order(self):
        if self._num_stages != 2 * self.pp_group_size:
            raise ValueError(
                "1F1B-V requires exactly 2 virtual stages per pipeline rank, "
                f"got num_stages={self._num_stages}, pp_group_size={self.pp_group_size}"
            )
        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size, self._num_stages, style=self.placement_style
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank

        self.pipeline_order = self._generate_1f1b_v_pipeline_order()
        self.pipeline_order = pad_to_max_length(self.pipeline_order)
        self._validate_pipeline_order()

    def _stage_to_rank(self, stage_index: int) -> int:
        return self.stage_index_to_group_rank[stage_index]

    def _dependency_ready_time(
        self, producer_stage: int, consumer_stage: int, producer_time: int
    ) -> int:
        # Stage outputs are available to adjacent logical stages on the next
        # schedule slot. Inter-rank handoffs enqueue P2P in the producer slot,
        # and same-rank handoffs are direct local transfers.
        return producer_time + 1

    @classmethod
    def _get_1f1bv_pattern_str(cls, positions: list[int]) -> str:
        pattern = [" "] * cls._symbol_pattern_size
        notations = "FfBbWw"
        for idx, position in enumerate(positions):
            if position >= 0:
                pattern[position] = notations[idx]
        return "".join(pattern)

    @classmethod
    def _create_1f1bv_whole_pattern(cls, pp_size: int) -> list[list[int]]:
        whole_pattern = [[0 for _ in range(cls._symbol_pattern_size)] for _ in range(pp_size)]
        now = 0
        for rank in range(pp_size):
            now += 1
            whole_pattern[rank][0] = now
        for rank in range(pp_size):
            now += 1
            whole_pattern[pp_size - 1 - rank][1] = now

        now += 1
        if pp_size % 3 == 0:
            now += 3
        cycle = (3 - (pp_size + 2) % 3) % 3
        for rank in range(pp_size):
            whole_pattern[rank][2], whole_pattern[rank][4] = now, now + 1
            cycle += 1
            now += 2
            if cycle == 3:
                cycle = 0
                now += 3
        for rank in range(pp_size):
            whole_pattern[pp_size - 1 - rank][3], whole_pattern[pp_size - 1 - rank][5] = (
                now,
                now + 1,
            )
            cycle += 1
            now += 2
            if cycle == 3:
                cycle = 0
                now += 3

        for rank in range(pp_size):
            for idx in range(cls._symbol_pattern_size):
                whole_pattern[rank][idx] %= cls._symbol_pattern_size
        return whole_pattern

    @classmethod
    def _init_1f1bv_repeated_schedule(
        cls,
        pp_size: int,
        n_microbatches: int,
        patterns: list[list[int]],
    ) -> list[list[str]]:
        repeated = []
        repeat_count = 4 * pp_size + n_microbatches + 1
        for rank in range(pp_size):
            repeated.append(list(cls._get_1f1bv_pattern_str(patterns[rank]) * repeat_count))
        return repeated

    @classmethod
    def _clear_1f1bv_invalid(
        cls,
        symbols: list[list[str]],
        rank: int,
        position: int,
        *,
        offset: int = -1,
    ) -> None:
        while 0 <= position < len(symbols[rank]):
            symbols[rank][position] = " "
            position += offset * cls._symbol_pattern_size

    @classmethod
    def _clear_1f1bv_invalid_indices(
        cls,
        symbols: list[list[str]],
        n_microbatches: int,
    ) -> list[list[str]]:
        pp_size = len(symbols)
        index = cls._symbol_pattern_size
        for identifier in "FfBb":
            ranks = range(pp_size) if identifier in "FB" else range(pp_size - 1, -1, -1)
            for rank in ranks:
                for _ in range(cls._symbol_pattern_size):
                    if symbols[rank][index] == identifier:
                        cls._clear_1f1bv_invalid(symbols, rank, index - cls._symbol_pattern_size)
                        cls._clear_1f1bv_invalid(
                            symbols,
                            rank,
                            index + cls._symbol_pattern_size * n_microbatches,
                            offset=1,
                        )
                        index += 1
                        if identifier in "Bb":
                            continuation = {"B": "W", "b": "w"}[identifier]
                            for continuation_offset in range(cls._symbol_pattern_size):
                                continuation_index = index + continuation_offset
                                if symbols[rank][continuation_index] == continuation:
                                    cls._clear_1f1bv_invalid(
                                        symbols,
                                        rank,
                                        continuation_index - cls._symbol_pattern_size,
                                    )
                                    cls._clear_1f1bv_invalid(
                                        symbols,
                                        rank,
                                        continuation_index
                                        + cls._symbol_pattern_size * n_microbatches,
                                        offset=1,
                                    )
                                    break
                        break
                    index += 1
        return symbols

    @classmethod
    def _process_1f1bv_warmup_without_increasing_peak_mem(
        cls,
        symbols: list[list[str]],
        n_microbatches: int,
        *,
        extra_peak_mem_by_rank: Sequence[int],
    ) -> list[list[str]]:
        peak_mem = 0
        memory = [[0 for _ in range(len(symbols[0]))] for _ in range(len(symbols))]
        locations = [
            [{key: -1 for key in ("F", "f", "B", "b", "W", "w")} for _ in range(n_microbatches + 2)]
            for _ in range(len(symbols))
        ]
        counters = [{key: 0 for key in ("F", "f", "B", "b", "W", "w")} for _ in range(len(symbols))]

        for rank in range(len(symbols)):
            current = 0
            for idx, symbol in enumerate(symbols[rank]):
                if symbol in "Ff":
                    current += 1
                if symbol in "Ww":
                    current -= 1
                memory[rank][idx] = current
                peak_mem = max(peak_mem, current)

        for idx in range(len(symbols[0])):
            for rank in range(len(symbols)):
                symbol = symbols[rank][idx]
                if symbol == " ":
                    continue

                counters[rank][symbol] += 1
                count = counters[rank][symbol]
                position = -1
                if count > 1:
                    position = locations[rank][count - 1][symbol]
                if symbol == "W":
                    position = max(position, locations[rank][count]["B"])
                if symbol == "w":
                    position = max(position, locations[rank][count]["b"])
                if symbol == "F" and rank > 0:
                    position = max(position, locations[rank - 1][count]["F"])
                if symbol == "f":
                    if rank != len(symbols) - 1:
                        position = max(position, locations[rank + 1][count]["f"])
                    else:
                        position = max(position, locations[rank][count]["F"])
                if symbol == "B":
                    if rank != 0:
                        position = max(position, locations[rank - 1][count]["W"])
                    else:
                        position = max(position, locations[rank][count]["f"])
                if symbol == "b":
                    if rank != len(symbols) - 1:
                        position = max(position, locations[rank + 1][count]["w"])
                    else:
                        position = max(position, locations[rank][count]["W"])

                position += 1
                while position < idx and symbols[rank][position] != " ":
                    position += 1
                if symbol in "Bb":
                    while position < idx and (
                        symbols[rank][position] != " " or symbols[rank][position + 1] != " "
                    ):
                        position += 1
                if position == idx:
                    locations[rank][count][symbol] = idx
                    continue

                if symbol in "BbWw":
                    symbols[rank][position] = symbol
                    symbols[rank][idx] = " "
                    if symbol in "Ww":
                        for mem_idx in range(position, idx):
                            memory[rank][mem_idx] -= 1
                    locations[rank][count][symbol] = position
                    continue

                allowed_peak_mem = peak_mem + extra_peak_mem_by_rank[rank]
                place = idx
                while place > position and memory[rank][place - 1] < allowed_peak_mem:
                    place -= 1
                while place < idx and symbols[rank][place] != " ":
                    place += 1
                if place == idx:
                    locations[rank][count][symbol] = idx
                    continue

                symbols[rank][place] = symbol
                symbols[rank][idx] = " "
                for mem_idx in range(place, idx):
                    memory[rank][mem_idx] += 1
                locations[rank][count][symbol] = place

        return symbols

    @classmethod
    def _generate_1f1bv_symbol_table(
        cls,
        pp_size: int,
        n_microbatches: int,
        *,
        forward_pull_ahead_extra_activations: int | Sequence[int] | Mapping[int, int] = 0,
    ) -> list[list[str]]:
        extra_peak_mem_by_rank = cls._normalize_forward_pull_ahead_extra_activations(
            pp_size,
            forward_pull_ahead_extra_activations,
        )
        symbols = cls._init_1f1bv_repeated_schedule(
            pp_size,
            n_microbatches,
            cls._create_1f1bv_whole_pattern(pp_size),
        )
        symbols = cls._clear_1f1bv_invalid_indices(symbols, n_microbatches)
        symbols = cls._process_1f1bv_warmup_without_increasing_peak_mem(
            symbols,
            n_microbatches,
            extra_peak_mem_by_rank=extra_peak_mem_by_rank,
        )
        for rank in range(len(symbols)):
            counts = {symbol: 0 for symbol in "FfBbWw"}
            for idx, symbol in enumerate(symbols[rank]):
                if symbol == " ":
                    continue
                if counts[symbol] >= n_microbatches:
                    symbols[rank][idx] = " "
                else:
                    counts[symbol] += 1

        last_non_empty = max(
            idx for row in symbols for idx, symbol in enumerate(row) if symbol != " "
        )
        return [row[: last_non_empty + 1] for row in symbols]

    @staticmethod
    def _normalize_forward_pull_ahead_extra_activations(
        pp_size: int,
        extra_activations: int | Sequence[int] | Mapping[int, int],
    ) -> tuple[int, ...]:
        if isinstance(extra_activations, int):
            if extra_activations < 0:
                raise ValueError("forward pull-ahead extra activations must be non-negative")
            return tuple(extra_activations for _ in range(pp_size))

        if isinstance(extra_activations, Mapping):
            values = tuple(int(extra_activations.get(rank, 0)) for rank in range(pp_size))
        else:
            values = tuple(int(value) for value in extra_activations)
            if len(values) != pp_size:
                raise ValueError(
                    "forward pull-ahead extra activation sequence must have one entry per PP rank"
                )

        if any(value < 0 for value in values):
            raise ValueError("forward pull-ahead extra activations must be non-negative")
        return values

    @staticmethod
    def _format_forward_pull_ahead_source(extra_activations: tuple[int, ...]) -> str:
        if not any(extra_activations):
            return ""
        if len(set(extra_activations)) == 1:
            return f"_pull_fwd_plus{extra_activations[0]}"
        active = "_".join(
            f"r{rank}p{value}" for rank, value in enumerate(extra_activations) if value
        )
        return f"_pull_fwd_{active}"

    @staticmethod
    def _convert_1f1bv_symbols_to_actions(
        symbols: list[list[str]],
        pp_size: int,
    ) -> dict[int, list[Optional[PipelineAction]]]:
        rows: dict[int, list[Optional[PipelineAction]]] = {}
        num_stages = 2 * pp_size
        for rank, row in enumerate(symbols):
            low_stage = rank
            high_stage = num_stages - 1 - rank
            next_microbatch = {symbol: 0 for symbol in ("F", "f", "B", "b")}
            pending_continuation: Optional[tuple[str, int, int]] = None
            actions: list[Optional[PipelineAction]] = []

            for idx, symbol in enumerate(row):
                if symbol == " ":
                    if pending_continuation is not None:
                        raise AssertionError(
                            "1F1B-V symbol table has a bubble between backward "
                            f"and continuation on rank={rank}, time={idx}"
                        )
                    actions.append(None)
                elif symbol == "F":
                    mb_index = next_microbatch["F"]
                    next_microbatch["F"] += 1
                    actions.append(PipelineAction(low_stage, FORWARD, mb_index))
                elif symbol == "f":
                    mb_index = next_microbatch["f"]
                    next_microbatch["f"] += 1
                    actions.append(PipelineAction(high_stage, FORWARD, mb_index))
                elif symbol == "B":
                    mb_index = next_microbatch["B"]
                    next_microbatch["B"] += 1
                    pending_continuation = ("W", high_stage, mb_index)
                    actions.append(PipelineAction(high_stage, FULL_BACKWARD, mb_index))
                elif symbol == "W":
                    if pending_continuation != ("W", high_stage, next_microbatch["B"] - 1):
                        raise AssertionError(
                            f"1F1B-V W is not the continuation of B on rank={rank}, time={idx}"
                        )
                    _, stage_index, mb_index = pending_continuation
                    actions.append(PipelineAction(stage_index, FULL_BACKWARD_CONT, mb_index))
                    pending_continuation = None
                elif symbol == "b":
                    mb_index = next_microbatch["b"]
                    next_microbatch["b"] += 1
                    pending_continuation = ("w", low_stage, mb_index)
                    actions.append(PipelineAction(low_stage, FULL_BACKWARD, mb_index))
                elif symbol == "w":
                    if pending_continuation != ("w", low_stage, next_microbatch["b"] - 1):
                        raise AssertionError(
                            f"1F1B-V w is not the continuation of b on rank={rank}, time={idx}"
                        )
                    _, stage_index, mb_index = pending_continuation
                    actions.append(PipelineAction(stage_index, FULL_BACKWARD_CONT, mb_index))
                    pending_continuation = None
                else:
                    raise AssertionError(f"Unexpected 1F1B-V symbol {symbol!r}")

            if pending_continuation is not None:
                raise AssertionError(
                    f"1F1B-V symbol row ended during backward continuation on rank={rank}"
                )
            rows[rank] = actions

        return rows

    def _generate_1f1b_v_pipeline_order(self) -> dict[int, list[Optional[PipelineAction]]]:
        num_microbatches = self._n_microbatches

        extra_activations = self._normalize_forward_pull_ahead_extra_activations(
            self.pp_group_size,
            getattr(self, "forward_pull_ahead_extra_activations", 0),
        )
        self.pipeline_order_source = (
            "generic_symbol_pattern" + self._format_forward_pull_ahead_source(extra_activations)
        )
        symbols = self._generate_1f1bv_symbol_table(
            self.pp_group_size,
            num_microbatches,
            forward_pull_ahead_extra_activations=extra_activations,
        )
        return self._convert_1f1bv_symbols_to_actions(symbols, self.pp_group_size)

    def _validate_pipeline_order(self) -> None:
        fwd_seen: dict[tuple[int, int], int] = {}
        bwd_seen: dict[tuple[int, int], int] = {}
        bwd_cont_seen: dict[tuple[int, int], int] = {}

        for rank, actions in self.pipeline_order.items():
            for time_step, action in enumerate(actions):
                if action is None:
                    continue
                if self._stage_to_rank(action.stage_index) != rank:
                    raise AssertionError(
                        f"1F1B-V schedule placed stage {action.stage_index} on rank {rank}, "
                        f"expected rank {self._stage_to_rank(action.stage_index)}"
                    )
                if action.computation_type == FORWARD:
                    assert action.microbatch_index is not None
                    key = (action.stage_index, action.microbatch_index)
                    if key in fwd_seen:
                        raise AssertionError(f"Duplicate forward action for {key}")
                    fwd_seen[key] = time_step
                elif action.computation_type == FULL_BACKWARD:
                    assert action.microbatch_index is not None
                    key = (action.stage_index, action.microbatch_index)
                    if key in bwd_seen:
                        raise AssertionError(f"Duplicate backward action for {key}")
                    bwd_seen[key] = time_step
                elif action.computation_type == FULL_BACKWARD_CONT:
                    assert action.microbatch_index is not None
                    key = (action.stage_index, action.microbatch_index)
                    if key in bwd_cont_seen:
                        raise AssertionError(f"Duplicate backward continuation for {key}")
                    bwd_cont_seen[key] = time_step

        expected = {
            (stage_index, mb_index)
            for stage_index in range(self._num_stages)
            for mb_index in range(self._n_microbatches)
        }
        if set(fwd_seen) != expected:
            raise AssertionError(
                f"Invalid 1F1B-V schedule forwards: got {len(fwd_seen)}, expected {len(expected)}"
            )
        if set(bwd_seen) != expected:
            raise AssertionError(
                f"Invalid 1F1B-V schedule backwards: got {len(bwd_seen)}, expected {len(expected)}"
            )
        if set(bwd_cont_seen) != expected:
            raise AssertionError(
                "Invalid 1F1B-V schedule backward continuations: "
                f"got {len(bwd_cont_seen)}, expected {len(expected)}"
            )

        for stage_index, mb_index in expected:
            fwd_time = fwd_seen[(stage_index, mb_index)]
            bwd_time = bwd_seen[(stage_index, mb_index)]
            bwd_cont_time = bwd_cont_seen[(stage_index, mb_index)]
            if fwd_time >= bwd_time:
                raise AssertionError(
                    f"Backward for stage={stage_index}, mb={mb_index} occurs before forward"
                )
            if bwd_cont_time != bwd_time + 1:
                raise AssertionError(
                    f"Backward continuation for stage={stage_index}, mb={mb_index} "
                    "must immediately follow backward"
                )
            if stage_index > 0:
                dep_time = fwd_seen[(stage_index - 1, mb_index)]
                ready_time = self._dependency_ready_time(stage_index - 1, stage_index, dep_time)
                if ready_time > fwd_time:
                    raise AssertionError(
                        f"Forward dependency violation for stage={stage_index}, mb={mb_index}"
                    )
            if stage_index < self._num_stages - 1:
                dep_time = bwd_cont_seen[(stage_index + 1, mb_index)]
                ready_time = self._dependency_ready_time(stage_index + 1, stage_index, dep_time)
                if ready_time > bwd_time:
                    raise AssertionError(
                        f"Backward dependency violation for stage={stage_index}, mb={mb_index}"
                    )


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
