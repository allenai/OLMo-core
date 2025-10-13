import torch
from olmo_core.utils import get_or_init_stream
from typing import Any, Dict, List, Optional, Tuple, cast
from functools import partial

# TODO: consider switching to weakref to automatically free memory when tensor is gone

class _OffloadCtx:
    def __init__(self, outer: "GPUActivationOffloader", group: str, enable: bool):
        self.outer = outer
        self.group = group
        self.enable = enable

    def __enter__(self):
        if not self.enable:
            return self.outer
        self.outer._context_manager = torch.autograd.graph.saved_tensors_hooks(
            pack_hook=partial(self.outer._pack_hook, self.group),
            unpack_hook=partial(self.outer._unpack_hook, self.group)
        )
        self.outer._context_manager.__enter__()
        return self.outer
    
    def __exit__(self, exc_type, exc, tb):
        if not self.enable:
            return
        
        if self.outer._context_manager is not None:
            self.outer._context_manager.__exit__(exc_type, exc, tb)
            self.outer._context_manager = None

class GPUActivationOffloader:
    def __init__(self,
                 target_device: str | torch.device,
                ):
        self.target_device = torch.device(target_device)

        # one dedicated memcpy stream per device we touch
        self._stream = get_or_init_stream(id=6)
        self._context_manager: Optional[torch.autograd.graph.saved_tensors_hooks] = None

        # example: {"0F1": {id1: tensor1, id2: tensor2}, "0F2": {id3: tensor3}}
        self._saved_tensors_by_group: Dict[str, Dict[int, torch.Tensor]] = {}

        self._offload_events: Dict[str, torch.cuda.Event] = {}
        self._reload_events: Dict[str, torch.cuda.Event] = {}

        # TODO: exclude model parameters

    
    def wait_reload(self, group: str):
        """Wait for the reload event of the given group to complete."""
        if group in self._reload_events:
            reload_event = self._reload_events[group]
            # torch.cuda.current_stream().wait_event(reload_event) # type: ignore
            # reload_event.synchronize()
            # torch.cuda.current_stream().synchronize()
            # self._stream.synchronize()
            torch.cuda.current_stream().wait_stream(self._stream)
            del self._reload_events[group]
            print(f"[GPUActivationOffloader] Reload event for group {group} completed.")
        else:
            raise RuntimeError("No reload event recorded, cannot wait.")

    def get_offload_context(self, group: str, enable: bool) -> _OffloadCtx:
        """Context manager to capture saved-for-backward tensors produced in this block."""
        return _OffloadCtx(self, group, enable)
    
    def manual_release_group(self, group: str):
        """Manually release all saved tensors in the given group."""

        if group in self._offload_events:
            raise RuntimeError("Offload event still exists for this group, cannot release.")

        if group in self._reload_events:
            raise RuntimeError("Reload event still exists for this group, cannot release.")
        
        if group in self._saved_tensors_by_group:
            del self._saved_tensors_by_group[group]
        else:
            raise RuntimeError("No saved tensors found for this group, cannot release.")

    # ---------- hooks ----------
    def _pack_hook(self, group: str, t: torch.Tensor):
        # Only consider CUDA tensors; skip tiny ones if threshold set
        if (not t.is_cuda):
            return t

        key = id(t)
        if key in self._saved_tensors_by_group.get(group, {}):
            # it's possible that the same tensor is packed multiple times in the same group
            return key

        if t.device.index != torch.cuda.current_device():
            raise RuntimeError(f"Tensor device {t.device} does not match current device {torch.cuda.current_device()}")

        self._saved_tensors_by_group.setdefault(group, {})[key] = t
        return key  # autograd will store this handle instead of the raw tensor

    def _unpack_hook(self, group: str, handle):
        if (not isinstance(handle, int)):
            return handle

        key = handle
        if key not in self._saved_tensors_by_group.get(group, {}):
            raise RuntimeError("Tensor handle not found?")

        # t = self._saved_tensors_by_group[group].pop(key)
        t = self._saved_tensors_by_group[group][key]

        # remove empty group
        if len(self._saved_tensors_by_group[group]) == 0:
            del self._saved_tensors_by_group[group]

        # when the unpack happens, the tensors should have already been transferred back to current device
        if t.device.index != torch.cuda.current_device():
            raise RuntimeError(f"Tensor device {t.device} does not match current device {torch.cuda.current_device()}")

        return t


    def async_offload(self, group: str) -> Optional[torch.cuda.Event]:
        """Async offload all saved tensors to the paired GPU."""


        _saved_tensors = self._saved_tensors_by_group.get(group, {})

        if len(_saved_tensors) == 0:
            return

        # wait for current stream to be done with these tensors
        # self._stream.wait_stream(torch.cuda.current_stream()) # BUG;TODO
        torch.cuda.current_stream().synchronize()

        total_sz_gb = 0
        # async copy each tensor
        with torch.cuda.stream(self._stream):
            # start_event = self._stream.record_event()
            for key, t in _saved_tensors.items():
                if t.device.index != torch.cuda.current_device():
                    raise RuntimeError(f"Tensor device {t.device} does not match current device {torch.cuda.current_device()}")

                _saved_tensors[key] = t.to(self.target_device, non_blocking=True)
                total_sz_gb += t.numel() * t.element_size() / 1024**3
            end_event = self._stream.record_event()
            end_event = cast(torch.cuda.Event, end_event)
            self._offload_events[group] = end_event

        print(f"[GPUActivationOffloader] Offload {len(_saved_tensors)} tensors ({total_sz_gb:.2f} GB, group {group}) from GPU {torch.cuda.current_device()} to {self.target_device} in background.")
        return end_event

    def async_reload(self, group: str) -> Optional[torch.cuda.Event]:
        """Async reload all saved tensors back to current device."""


        _saved_tensors = self._saved_tensors_by_group.get(group, {})
        if len(_saved_tensors) == 0:
            return

        # before reloading, make sure offload is done
        if group in self._offload_events:
            offload_event = self._offload_events[group]
            self._stream.wait_event(offload_event) # type: ignore
            del self._offload_events[group]
        else: # no offload event recorded, this means no offload happened
            raise RuntimeError("No offload event recorded, cannot reload.")

        # wait for current stream to be done with these tensors
        # self._stream.wait_stream(torch.cuda.current_stream())
        total_sz_gb = 0

        # async copy each tensor
        with torch.cuda.stream(self._stream):
            # start_event = self._stream.record_event()
            for key, t in _saved_tensors.items():
                if t.device.index == torch.cuda.current_device():
                    raise RuntimeError(f"Tensor device {t.device} already on current device {torch.cuda.current_device()}")

                _saved_tensors[key] = t.to(torch.cuda.current_device(), non_blocking=True)
                total_sz_gb += t.numel() * t.element_size() / 1024**3

            end_event = self._stream.record_event()
            end_event = cast(torch.cuda.Event, end_event)
            self._reload_events[group] = end_event

        print(f"[GPUActivationOffloader] Reload {len(_saved_tensors)} tensors ({total_sz_gb:.2f} GB, group {group}) from {self.target_device} to GPU {torch.cuda.current_device()} in background.")
        return end_event
    