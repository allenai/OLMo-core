# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Functionality for CPU offloading of tensors saved for backward pass."""
from __future__ import annotations
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple, List

import torch

__all__ = ["get_cpu_offload_context"]

CPUOffloadEnabled = False


def mark_activation_offload(*tensors):
    """Set the type of the offloading needed for a tensor."""
    for tensor in tensors:
        if tensor is None:
            continue
        if type(tensor) in [torch.Tensor, torch.nn.Parameter]:
            tensor.activation_offloading = True
        else:
            raise TypeError(f"Unsupported type {type(tensor)} for activation offloading.")


def is_cpu_offload_enabled() -> bool:
    """Check if CPU offloading is currently enabled."""
    return CPUOffloadEnabled


class CpuOffloadHook:

    def __init__(
        self,
        offload_handler: "CpuOffloadHandler",
    ) -> None:
        self.offload_handler: CpuOffloadHandler = offload_handler
        self.inside_context = False

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        retrieve_identifier = self.offload_handler.tensor_push(tensor)
        return retrieve_identifier

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        tensor = self.offload_handler.tensor_pop(saved_state)
        return tensor

    def __enter__(self):
        global CPUOffloadEnabled
        CPUOffloadEnabled = True

        self.inside_context = True
        torch._C._autograd._push_saved_tensors_default_hooks(  # type: ignore
            self.on_save_for_backward, self.on_get_saved_tensor
        )

    def __exit__(self, *args: Any):
        global CPUOffloadEnabled
        CPUOffloadEnabled = False

        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()  # type: ignore

class GroupCommitFunction(torch.autograd.Function):
    """Dummy op whose output equals input.
    Used to mark timepoints to trigger offload/reload work on both
    forward and backward passes.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        cpu_offload_handler.on_group_commit_forward()
        ctx.cpu_offload_handler = cpu_offload_handler
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward()
        return grad_output[0], None


group_prefetch_offload_commit = GroupCommitFunction.apply


class CpuOffloadHandler:
    """Overlaps host/device copies with compute using a windowed, bulk
    offload-and-prefetch scheme and an alternating double-buffer for reloads.

    This variant supports *variable numbers of tensors per group/layer*.
    Instead of preallocating one flat list of GPU copy buffers (which assumes
    uniform counts), we maintain two banks (double buffers). For each group
    reload we (re)size only the active bank to the exact number and shapes
    of tensors in that group. This preserves overlap without stalling and
    avoids large upfront allocations.
    """

    def __init__(
        self,
        num_offload_group: int,  # <= actual number of groups (commits)
        num_model_group: int,
        tensor_need_offloading_checker=(lambda t: True),
    ) -> None:
        self.num_offload_group = num_offload_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker

        # Counters and registries
        self.current_group, self.tensor_count_current_group = (0, 0)
        self.torch_tensor_count = 0  # fake/functional tensors
        self.tensor_tag_to_state: Dict[Tuple[int, int], Any] = {}

        # Number of layers in the model (used by windowing logic)
        self.num_layers = num_model_group

        # Track in-flight activations (GPU refs) for freeing
        self.tensor_tag_to_buf: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}

        # How many groups have been offloaded so far
        self.offloaded_group_count = 0

        # Windowing: map each offload-count milestone to the last group id
        self.layer_window_map: Dict[int, int] = {}
        constant = 0
        for i in range(self.num_offload_group):
            self.layer_window_map[i] = (self.num_layers // self.num_offload_group) * (i + 1) - 1
            if i < (self.num_layers % self.num_offload_group):
                self.layer_window_map[i] += i + 1
                constant = i + 1
            else:
                self.layer_window_map[i] += constant

        # Streams
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

        # Double buffer banks (alternating); each bank is a list of GPU tensors
        # sized *for the group being reloaded into that bank*.
        self.reload_double_buffer: List[List[torch.Tensor]] = [[], []]

    def groupid_reset(self):
        """Reset group and state registries (for a new forward)."""
        self.current_group, self.tensor_count_current_group = (0, 0)
        self.torch_tensor_count = 0
        self.tensor_tag_to_state.clear()

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:

        stray = isinstance(
            tensor,
            (
                torch._subclasses.fake_tensor.FakeTensor, # type: ignore
                torch._subclasses.functional_tensor.FunctionalTensor, # type: ignore
            ),
        )

        if stray:
            tensor_tag = (-1, self.torch_tensor_count)
            self.torch_tensor_count += 1
            self.tensor_tag_to_state[tensor_tag] = tensor
            return tensor_tag


        tensor_tag = (self.current_group, self.tensor_count_current_group)
        self.tensor_count_current_group += 1
        assert tensor_tag not in self.tensor_tag_to_state

        self.tensor_tag_to_state[tensor_tag] = tensor
        if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(tensor):
            self.tensor_tag_to_buf[tensor_tag] = tensor
        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Return the reloaded GPU tensor to autograd when needed."""
        assert tensor_tag in self.tensor_tag_to_state
        tensor = self.tensor_tag_to_state.pop(tensor_tag)

        tensor._do_not_clear = True  # guard against accidental clears
        self.tensor_tag_to_buf.pop(tensor_tag, None)

        # It should have been reloaded in on_group_commit_backward()
        assert not isinstance(tensor, tuple), "Tensor was not reloaded from CPU."
        return tensor

    def bulk_offload_group(self, group_to_offload: int):
        """Copy all tensors from a group to pinned CPU (async, D2H stream)."""
        with torch.cuda.stream(self.d2h_stream):  # type: ignore
            for (group_id, _), state in list(self.tensor_tag_to_state.items()):
                if group_id != group_to_offload:
                    continue
                assert not isinstance(state, tuple), "Duplicate offload?"
                tensor_on_device = state  # torch.Tensor

                if tensor_on_device.device.type == "cpu":
                    continue  # already on CPU
                if self.tensor_need_offloading_checker(tensor_on_device):
                    self.tensor_tag_to_state[(group_id, _) ] = CpuOffloadHandler.offload(tensor_on_device)

    def _collect_group_entries(self, group_id: int) -> List[Tuple[Tuple[int, int], Any]]:
        """Collect (tag, state) for a specific group, preserving insertion order."""
        return [(tag, st) for tag, st in self.tensor_tag_to_state.items() if tag[0] == group_id]

    def _ensure_bank_for_group(self, bank_idx: int, group_entries: List[Tuple[Tuple[int, int], Any]]):
        """Ensure the chosen bank has buffers sized to this group's tensors."""
        bank = self.reload_double_buffer[bank_idx]

        # Ensure capacity
        needed = len(group_entries)
        if len(bank) < needed:
            # Grow with placeholder Nones first to avoid reallocation churn
            bank.extend([None] * (needed - len(bank)))  # type: ignore

        # For each entry, allocate or resize if needed
        for i, (_, state) in enumerate(group_entries):
            if not isinstance(state, tuple):
                # Not offloaded (e.g., stayed on GPU or quantized path not used)
                bank[i] = None  # type: ignore
                continue

            dev, cpu_backup = state
            # Allocate fresh if missing or shape/dtype/device mismatched
            need_new = (
                (i >= len(bank)) or
                (bank[i] is None) or
                (bank[i].device != dev) or
                (bank[i].dtype != cpu_backup.dtype) or
                (bank[i].layout != cpu_backup.layout) or
                (tuple(bank[i].size()) != tuple(cpu_backup.size()))
            )
            if need_new:
                bank[i] = torch.empty(
                    cpu_backup.size(),
                    dtype=cpu_backup.dtype,
                    layout=cpu_backup.layout,
                    device=dev,
                )  # type: ignore

        # Trim extra slots if previous group had more tensors
        if len(bank) > needed:
            del bank[needed:]

    def bulk_reload_group(self, group_to_reload: int):
        """Reload all tensors for a group from pinned CPU to GPU copy buffers.
        Uses the alternating bank determined by group parity.
        """
        assert group_to_reload < self.num_offload_group

        # Gather entries for this group (preserve original order)
        entries = self._collect_group_entries(group_to_reload)
        double_bank_idx = group_to_reload % 2

        # Prepare the active bank to match this group's tensor shapes/count
        self._ensure_bank_for_group(double_bank_idx, entries)

        with torch.cuda.stream(self.h2d_stream):  # type: ignore
            bank = self.reload_double_buffer[double_bank_idx]
            buf_i = 0
            for (tensor_tag, state) in entries:
                if isinstance(state, tuple):
                    # Reload into the pre-sized buffer
                    reload_buf = bank[buf_i]
                    assert reload_buf is not None, "Missing reload buffer."
                    recovered_tensor = CpuOffloadHandler.reload(state, reload_buf)
                    self.tensor_tag_to_state[tensor_tag] = recovered_tensor
                    buf_i += 1
                elif isinstance(state, list):
                    raise RuntimeError("Quantized tensors not supported in CPU offloading")
                else:
                    # Already on GPU/CPU (not offloaded); nothing to do
                    buf_i += 1


    def on_group_commit_forward(self):
        """Called at each group boundary in forward pass."""
        # Kick off first offload after first compute finishes
        if self.current_group == 0:
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            self.bulk_offload_group(self.current_group)

        # Windowed sync and freeing of GPU activations
        if self.layer_window_map[self.offloaded_group_count] == self.current_group:
            # Sync streams so copies are visible
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_stream(self.d2h_stream)

            # Free GPU references for this window
            for (group_id, _), tensor_buf in list(self.tensor_tag_to_buf.items()):
                if group_id == self.offloaded_group_count:
                    # Release reference to allow memory reuse
                    self.tensor_tag_to_buf[(group_id, _)] = None

            # Offload next group if any
            if self.offloaded_group_count < (self.num_offload_group - 1):
                self.bulk_offload_group(self.offloaded_group_count + 1)

            self.offloaded_group_count += 1

        # Advance to next group
        self.current_group += 1
        self.tensor_count_current_group = 0

    def on_group_commit_backward(self):
        """Called at each group boundary in backward pass (reverse order)."""
        self.current_group -= 1
        assert self.current_group >= 0

        if self.layer_window_map[self.offloaded_group_count - 1] == self.current_group:
            # Synchronize so we don't race use of reloaded buffers
            self.h2d_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.current_stream().wait_stream(self.h2d_stream)

            # Reload tensors for the just-finished forward group
            self.bulk_reload_group(self.offloaded_group_count - 1)

            # Decrement window count but keep >= 1 while there are groups left
            self.offloaded_group_count -= 1 if self.offloaded_group_count > 1 else 0

        # After last group, ensure all reloads completed
        if self.current_group == 0:
            torch.cuda.current_stream().wait_stream(self.h2d_stream)
            self.offloaded_group_count = 0


    @staticmethod
    def offload(src_tensor: torch.Tensor):
        """Async device->host (pinned) copy; return offload state."""
        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=True,
        )
        cpu_backup.copy_(src_tensor, non_blocking=True)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state, copy_buffer=None):
        """Async host->device copy into provided buffer (if given)."""
        dev, cpu_backup = state
        assert cpu_backup.is_pinned()

        if copy_buffer is None:
            return cpu_backup.to(dev, non_blocking=True)

        assert cpu_backup.size() == copy_buffer.size(), "Can't copy two buffers of different sizes!"

        copy_buffer.copy_(cpu_backup, non_blocking=True)
        return copy_buffer

def get_cpu_offload_context(
    enabled: bool = False,
    num_layers: int = 1,
    model_layers: int = 1,
):
    """Return (context_manager, group_commit_op) when enabled, else (nullcontext, None)."""
    cpu_offload_handler = CpuOffloadHandler(
        num_offload_group=num_layers,
        num_model_group=model_layers,
        tensor_need_offloading_checker=tensor_need_offloading_checker_activations,
    )

    def group_prefetch_offload_commit_async(tensor):
        return group_prefetch_offload_commit(tensor, cpu_offload_handler)

    if enabled:
        return (
            CpuOffloadHook(offload_handler=cpu_offload_handler),
            group_prefetch_offload_commit_async,
        )
    return nullcontext(), None


def tensor_need_offloading_checker_activations(tensor):
    # return hasattr(tensor, "activation_offloading")
    return True
