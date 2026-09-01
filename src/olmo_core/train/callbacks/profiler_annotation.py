import dataclasses
import functools as ft
import logging
import threading
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, cast

import torch
import torch.nn as nn
from torch.profiler import record_function

from olmo_core.config import StrEnum
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import Transformer

from .callback import Callback
from .profiler import ProfilerCallback, should_profile_rank

log = logging.getLogger(__name__)


_MIXER_ABBREVIATIONS = {
    "Attention": "attn",
    "NormalizedAttention": "attn",
    "FusedAttention": "attn",
    "GatedDeltaNet": "gdn",
    "KimiDeltaAttention": "kda",
}

# (block attribute, label used in the marker name), in forward order.
#
# NOTE: these tables are per block *shape*, not one flat table, because the same attribute
# name sits at different points in the forward pass depending on the block. On the standard
# block 'attention_norm' is the pre-norm; on 'OLMoDDPTransformerBlock', which runs peri-norm,
# it is the *post*-norm and 'attention_input_norm' is the pre-norm. A single table would have
# to mislabel one of them.
_SUBMODULE_LABELS = (
    ("attention_norm", "norm_pre_mixer"),
    ("attention", "mixer"),
    ("post_attention_norm", "norm_post_mixer"),
    ("feed_forward_norm", "norm_pre_ffn"),
    ("feed_forward", "ffn"),
    ("post_feed_forward_norm", "norm_post_ffn"),
    ("feed_forward_moe_norm", "norm_pre_moe"),
    ("feed_forward_moe", "moe"),
)

# The fused MoE-v2 block ('olmo_core.nn.ddp.block.OLMoDDPTransformerBlock'), which runs
# peri-norm and reaches its experts through 'routed_experts' / 'shared_experts' rather than a
# single 'feed_forward_moe'. Order follows '_res_norm_attn' and the 'combined_forward_*'
# functions in 'olmo_core.nn.moe.v2'.
#
# Two members of that forward pass are deliberately absent because no forward hook can see
# them: the shared expert is invoked as 'shared_experts.forward1(...)' / '.forward2(...)',
# bypassing '__call__', and the attention sublayer is reached through the compiled
# '_res_norm_attn' method. Both are covered by nvtx ranges in the MoE code instead.
_OLMO_DDP_SUBMODULE_LABELS = (
    ("attention_input_norm", "norm_pre_mixer"),
    ("attention", "mixer"),
    ("attention_norm", "norm_post_mixer"),
    ("feed_forward_input_norm", "norm_pre_moe"),
    ("routed_experts_router", "router"),
    ("shared_experts_router", "shared_router"),
    ("routed_experts", "experts"),
    ("feed_forward_norm", "norm_post_moe"),
)


# Sentinel for "torch.profiler didn't tell us what it's doing", so a missing
# attribute is distinguishable from a real profiler action.
_UNKNOWN_ACTION = object()


class AnnotationBackend(StrEnum):
    """
    How annotation ranges are emitted.
    """

    record_function = "record_function"
    """Emit :class:`torch.profiler.record_function` ranges, for :class:`ProfilerCallback` traces."""

    nvtx = "nvtx"
    """Emit NVTX ranges, for external profilers like Nsight Systems."""

    both = "both"
    """Emit both."""


def unwrap_block(block: nn.Module) -> nn.Module:
    """
    Unwrap an activation-checkpoint-wrapped block, returning ``block`` unchanged otherwise.
    """
    return getattr(block, "_checkpoint_wrapped_module", block)


def block_annotation_name(block: nn.Module, block_idx: int, index_width: int = 2) -> str:
    """
    Build a stable, sortable annotation name for a transformer block, like ``"block04.attn"``.

    :param block: The block, possibly wrapped for activation checkpointing.
    :param block_idx: The index of the block within the model.
    :param index_width: How many digits to zero-pad the block index to.
    """
    inner = unwrap_block(block)
    # NOTE: the sequence mixer lives at '.attention' even when it's a linear-attention variant
    # like GatedDeltaNet.
    mixer = getattr(inner, "attention", None)
    if mixer is None:
        kind = "block"
    else:
        cls_name = type(mixer).__name__
        kind = _MIXER_ABBREVIATIONS.get(cls_name, cls_name.lower())
    if getattr(inner, "is_moe", False):
        kind = f"{kind}+moe"
    return f"block{block_idx:0{index_width}d}.{kind}"


def submodule_labels(block: nn.Module) -> tuple[tuple[str, str], ...]:
    """
    Pick the ``(attribute, label)`` table describing the inside of ``block``, in forward order.

    Selection is by attribute shape rather than by class, so a block that grows out of either
    layout keeps working without importing it here. Attributes the block doesn't have are
    skipped by the caller, so a table may name more than a given block carries (e.g. the dense
    first layer of an MoE model has no router or routed experts).

    :param block: The block, already unwrapped by :func:`unwrap_block`.
    """
    # 'attention_input_norm' is the peri-norm pre-norm, which only the fused MoE-v2 block has.
    if hasattr(block, "attention_input_norm") or hasattr(block, "routed_experts"):
        return _OLMO_DDP_SUBMODULE_LABELS
    return _SUBMODULE_LABELS


def _grad_output_tensor(output: Any) -> Optional[torch.Tensor]:
    """
    Pick the tensor whose gradient marks the start of a module's backward pass.
    """
    if isinstance(output, torch.Tensor):
        return output if output.requires_grad else None
    loss = getattr(output, "loss", None)  # e.g. 'LMOutputWithLoss'
    if isinstance(loss, torch.Tensor) and loss.requires_grad:
        return loss
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.requires_grad:
                return item
    return None


class _Range:
    """
    A single open annotation range.
    """

    __slots__ = ("name", "thread_id", "_rf", "_nvtx")

    def __init__(self, name: str, *, use_rf: bool, use_nvtx: bool):
        self.name = name
        self.thread_id = threading.get_ident()
        self._rf = None
        self._nvtx = use_nvtx
        if use_rf:
            self._rf = record_function(name)
            self._rf.__enter__()
        if use_nvtx:
            torch.cuda.nvtx.range_push(name)

    def close(self):
        # NOTE: NVTX was pushed last, so pop it first to keep both stacks LIFO.
        if self._nvtx:
            torch.cuda.nvtx.range_pop()
        if self._rf is not None:
            self._rf.__exit__(None, None, None)


@dataclass
class ProfilerAnnotationCallback(Callback):
    """
    Emits named profiler ranges around the phases of a training step and around every transformer
    block, so traces from :class:`ProfilerCallback` (or from an external profiler like Nsight
    Systems) can be attributed to something other than anonymous "Torch-Compiled Region" entries.

    With the defaults you get these ranges:

    - ``data_loading`` — from just before the data loader is polled until the batch is handed to
      the train module.
    - ``fwd`` / ``bwd`` — one pair per micro-batch.
    - ``fwd/blockNN.<kind>`` / ``bwd/blockNN.<kind>`` — per block, where ``<kind>`` identifies the
      sequence mixer (``gdn``, ``attn``, ...), e.g. ``fwd/block04.attn``.
    - ``fwd/lm_head`` / ``bwd/lm_head``.
    - ``optim_step``, plus ``optim_step/pre`` for grad clipping and LR scheduling. Note that
      :mod:`torch.optim` already emits ``Optimizer.step#<name>.step`` inside of that.

    .. important::
        The annotations are emitted from eager code *outside* of every compiled region, so
        compilation is untouched (one graph per block, no graph breaks). ``record_function``
        markers placed *inside* a compiled region are useless: dynamo only traces them under
        ``torch._dynamo.config.capture_profiler_record_function`` and inductor strips them again
        in its ``_remove_profiler_ops`` pass.

    .. warning::
        Setting :data:`depth` to ``2`` annotates the *inside* of each block, which splits the
        block's graph at every annotated child (roughly 5 graphs instead of 1). Compilation stays
        on, but fusion across those boundaries is lost, so absolute times shift. Use it to find
        the expensive part of a block, not to measure the block.
    """

    # NOTE: must run *after* 'ProfilerCallback' (priority 0) so that ranges opened in
    # 'pre_load_batch' land inside the right 'ProfilerStep#N' row, since the profiler's step
    # boundary is also in 'pre_load_batch'.
    priority: ClassVar[int] = -3

    enabled: bool = False
    """
    Master switch. When ``False`` no hooks are registered at all.
    """

    backend: AnnotationBackend = AnnotationBackend.record_function
    """
    Whether to emit :class:`~torch.profiler.record_function` ranges, NVTX ranges, or both.
    """

    ranks: Optional[str] = None
    """
    Which ranks to annotate. Same semantics as :data:`ProfilerCallback.ranks`.
    """

    start: Optional[int] = None
    """
    First step to annotate (inclusive). Takes precedence over :data:`follow_profiler`.
    """

    end: Optional[int] = None
    """
    Last step to annotate (inclusive). Takes precedence over :data:`follow_profiler`.
    """

    follow_profiler: bool = True
    """
    When :data:`start` and :data:`end` are both unset, annotate exactly the steps that an enabled
    sibling :class:`ProfilerCallback` is warming up or recording on. If there's no such callback
    every step is annotated.
    """

    annotate_phases: bool = True
    """
    Emit the ``data_loading``, ``fwd``, ``bwd``, and ``optim_step`` ranges.
    """

    annotate_blocks: bool = True
    """
    Emit the per-block ranges.
    """

    annotate_lm_head: bool = True
    """
    Emit the ``fwd/lm_head`` and ``bwd/lm_head`` ranges.
    """

    annotate_backward: bool = True
    """
    Attribute the backward pass by opening a range when a module's output-gradient arrives.
    Set to ``False`` for forward-only markers.
    """

    annotate_eval: bool = False
    """
    Also annotate forward passes run by evaluators (i.e. while the model is in eval mode).
    """

    depth: int = 1
    """
    ``1`` annotates whole blocks and doesn't introduce any graph breaks. ``2`` also annotates the
    sequence mixer, feed-forward, and norms inside each block. See the warning above.
    """

    include_dp_comms: bool = True
    """
    Register the forward pre-hooks with ``prepend=True`` so that a block's range also covers
    FSDP2's parameter all-gather and reshard for that block (FSDP registers its own pre-hook with
    ``prepend=True``, so an appended hook would start *after* the unshard). Set to ``False`` to
    measure block compute with the data-parallel communication excluded.
    """

    name_prefix: str = ""
    """
    Optional prefix prepended to every marker name, e.g. ``"olmo/"``.
    """

    # NOTE: following 'GAPMonitorCallback', internal state is declared with 'field(repr=False)'
    # so it stays out of the callback's repr while remaining serializable (always ``None`` at
    # config-build time).
    _handles: Optional[List[Any]] = dataclasses.field(default=None, repr=False)
    _model: Optional[Transformer] = dataclasses.field(default=None, repr=False)
    _profiler_cb: Optional[ProfilerCallback] = dataclasses.field(default=None, repr=False)
    _fwd_stack: Optional[List["_Range"]] = dataclasses.field(default=None, repr=False)
    _bwd_open: Optional[Dict[int, "_Range"]] = dataclasses.field(default=None, repr=False)
    _use_rf: bool = dataclasses.field(default=True, repr=False)
    _use_nvtx: bool = dataclasses.field(default=False, repr=False)
    _active: bool = dataclasses.field(default=False, repr=False)
    _optim_step_hooks: bool = dataclasses.field(default=False, repr=False)

    def post_attach(self):
        if not self.enabled:
            return
        # NOTE: duck-typed rather than an isinstance check against TransformerTrainModule.
        # Everything this callback touches is 'train_module.model' (a Transformer, whose
        # '.blocks' / '.embeddings' / '.lm_head' it hooks) plus an optional '.optim'. Several
        # train modules satisfy that without sharing a base class -- OLMoDDPTrainModule, which
        # trains the fused MoE-v2 stack, derives straight from TrainModule.
        model = getattr(self.trainer.train_module, "model", None)
        if not isinstance(model, Transformer):
            raise OLMoConfigurationError(
                f"{type(self).__name__} needs a train module exposing a 'model' of type "
                f"Transformer, but {type(self.trainer.train_module).__name__} exposes "
                f"{type(model).__name__}."
            )
        if self.depth not in (1, 2):
            raise OLMoConfigurationError(f"'depth' must be 1 or 2, got {self.depth}")
        if self.backend != AnnotationBackend.record_function and not torch.cuda.is_available():
            log.warning("NVTX annotations require CUDA, falling back to record_function")
            self.backend = AnnotationBackend.record_function

    def pre_train(self):
        self._reset()
        if not self.enabled or not should_profile_rank(self.ranks):
            return

        train_module = self.trainer.train_module
        model = cast(Transformer, train_module.model)
        self._model = model
        self._use_rf = self.backend in (AnnotationBackend.record_function, AnnotationBackend.both)
        self._use_nvtx = self.backend in (AnnotationBackend.nvtx, AnnotationBackend.both)
        self._fwd_stack = []
        self._bwd_open = {}
        handles: List[Any] = []

        if self.follow_profiler and self.start is None and self.end is None:
            for cb in self.trainer.callbacks.values():
                if isinstance(cb, ProfilerCallback) and cb.enabled:
                    self._profiler_cb = cb
                    break

        if self.annotate_phases:
            # The root model is never compiled, so these hooks are plain eager code.
            handles.append(
                model.register_forward_pre_hook(ft.partial(self._fwd_pre, base="", bwd_level=0))
            )
            handles.append(
                model.register_forward_hook(ft.partial(self._fwd_post, base="", bwd_level=0))
            )
            # The gradient w.r.t. the embedding output is the last thing the backward produces,
            # so it's where the backward ranges get closed out.
            if model.embeddings is not None:
                handles.append(model.embeddings.register_forward_hook(self._embeddings_post))
            # 'optim' is None on an eval-only train module, which has no optimizer step to
            # name, and absent entirely on a train module that doesn't own one.
            #
            # NOTE: the step hooks come from torch.optim.Optimizer, and not every optimizer
            # here is one. 'OLMoDDPOptimizer' (the fused MoE-v2 optimizer) is deliberately a
            # plain class, so it has no 'register_step_pre_hook'. Without this guard attaching
            # the callback to that train module raises AttributeError. The coarse 'optim_step'
            # range still works: 'pre_optim_step' opens it and 'post_step' drains it. Only the
            # inner 'optim_step/pre' split (grad clipping, LR scheduling) is lost.
            optim = getattr(train_module, "optim", None)
            if optim is None:
                log.warning("train module has no optimizer, skipping 'optim_step' annotations")
            elif not hasattr(optim, "register_step_pre_hook"):
                log.warning(
                    f"{type(optim).__name__} is not a torch.optim.Optimizer, so it has no step "
                    "hooks; skipping the 'optim_step/pre' annotation"
                )
            else:
                handles.append(optim.register_step_pre_hook(self._optim_step_pre))
                handles.append(optim.register_step_post_hook(self._optim_step_post))
                self._optim_step_hooks = True

        if self.annotate_blocks:
            index_width = max(2, len(str(max(0, model.n_layers - 1))))
            for key, block in model.blocks.items():
                block_idx = int(key)
                name = block_annotation_name(block, block_idx, index_width)
                # NOTE: hook the module stored in 'blocks', which with activation checkpointing is
                # the eager wrapper, so that recompute doesn't fire these hooks a second time.
                handles.append(
                    block.register_forward_pre_hook(
                        ft.partial(self._fwd_pre, base=name, bwd_level=1),
                        prepend=self.include_dp_comms,
                    )
                )
                handles.append(
                    block.register_forward_hook(ft.partial(self._fwd_post, base=name, bwd_level=1))
                )
                if self.depth >= 2:
                    inner = unwrap_block(block)
                    for attr, label in submodule_labels(inner):
                        child = getattr(inner, attr, None)
                        if not isinstance(child, nn.Module):
                            continue
                        child_name = f"{name}/{label}"
                        handles.append(
                            child.register_forward_pre_hook(
                                ft.partial(self._fwd_pre, base=child_name, bwd_level=2)
                            )
                        )
                        handles.append(
                            child.register_forward_hook(
                                ft.partial(self._fwd_post, base=child_name, bwd_level=2)
                            )
                        )

        if self.annotate_lm_head and model.lm_head is not None:
            handles.append(
                model.lm_head.register_forward_pre_hook(
                    ft.partial(self._fwd_pre, base="lm_head", bwd_level=1),
                    prepend=self.include_dp_comms,
                )
            )
            handles.append(
                model.lm_head.register_forward_hook(
                    ft.partial(self._fwd_post, base="lm_head", bwd_level=1)
                )
            )

        self._handles = handles
        log.info(
            f"Registered {len(handles)} profiler annotation hooks "
            f"(depth={self.depth}, backend={self.backend})"
        )

    def _name(self, phase: str, base: str) -> str:
        return f"{self.name_prefix}{phase}/{base}" if base else f"{self.name_prefix}{phase}"

    def _begin(self, name: str) -> _Range:
        return _Range(name, use_rf=self._use_rf, use_nvtx=self._use_nvtx)

    def _push(self, name: str):
        assert self._fwd_stack is not None
        self._fwd_stack.append(self._begin(name))

    def _close(self, name: str):
        stack = self._fwd_stack
        if not stack or not any(r.name == name for r in stack):
            # Never opened (e.g. the annotation window opened mid-forward).
            return
        while stack:
            range_ = stack.pop()
            self._end(range_)
            if range_.name == name:
                return

    def _end(self, range_: _Range):
        if range_.thread_id != threading.get_ident():
            # Closing another thread's range would corrupt that thread's NVTX / profiler
            # stack, so drop it instead. In the normal case this never fires: backward ranges
            # are opened and closed on the autograd thread (the '_bwd_end' grad hook on the
            # embeddings output runs '_bwd_finish' there), and '_drain' from '_fwd_pre' only
            # meets them if a backward never reached the embeddings -- an error, or the
            # annotation window closing mid-step. Under the nvtx backend a dropped range
            # leaves its push unmatched, which shows up in nsys as a range on that thread
            # running to the end of the capture.
            log.warning(
                f"dropping annotation range '{range_.name}' opened on another thread; "
                "under backend='nvtx' this leaves an unclosed range in the trace"
            )
            return
        range_.close()

    def _bwd_advance(self, level: int, base: str):
        """
        Move the backward cursor at ``level`` onto ``base``, closing deeper levels first.

        The backward pass has no natural "exit" hook per module: on the tensor shared by two
        adjacent blocks the hooks fire in registration order, so paired enter/exit hooks would
        cross rather than nest. Instead each output-gradient hook advances a cursor, which makes
        ``bwd/blockN`` run from "grad of block N's output is ready" to "grad of block N-1's output
        is ready".
        """
        assert self._bwd_open is not None
        if level > 0 and 0 not in self._bwd_open:
            self._bwd_open[0] = self._begin(self._name("bwd", ""))
        for open_level in sorted([lv for lv in self._bwd_open if lv >= level], reverse=True):
            self._end(self._bwd_open.pop(open_level))
        self._bwd_open[level] = self._begin(self._name("bwd", base))

    def _bwd_finish(self):
        if not self._bwd_open:
            return
        for level in sorted(self._bwd_open, reverse=True):
            self._end(self._bwd_open.pop(level))

    def _drain(self):
        self._bwd_finish()
        if self._fwd_stack:
            while self._fwd_stack:
                self._end(self._fwd_stack.pop())

    @torch._dynamo.disable()
    def _fwd_pre(self, module: nn.Module, args, *, base: str, bwd_level: int):
        del module, args
        if not self._active or not self._annotating():
            return None  # NOTE: a non-None return value would replace the module's inputs.
        if bwd_level == 0:
            # New micro-batch, nothing should still be open.
            self._drain()
        self._push(self._name("fwd", base))
        return None

    @torch._dynamo.disable()
    def _fwd_post(self, module: nn.Module, args, output, *, base: str, bwd_level: int):
        del module, args
        if not self._active or not self._annotating():
            return None
        self._close(self._name("fwd", base))
        if not self.annotate_backward or not torch.is_grad_enabled():
            return None
        if bwd_level >= 1:
            out = _grad_output_tensor(output)
            if out is not None:
                out.register_hook(ft.partial(self._bwd_enter, base=base, level=bwd_level))
        return None

    @torch._dynamo.disable()
    def _embeddings_post(self, module: nn.Module, args, output):
        del module, args
        if not self._active or not self._annotating() or not torch.is_grad_enabled():
            return None
        if not self.annotate_backward:
            return None
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(self._bwd_end)
        return None

    @torch._dynamo.disable()
    def _bwd_enter(self, grad, *, base: str, level: int):
        del grad
        if self._active:
            self._bwd_advance(level, base)
        return None  # NOTE: a non-None return value would replace the gradient.

    @torch._dynamo.disable()
    def _bwd_end(self, grad=None):
        del grad
        self._bwd_finish()
        return None

    @torch._dynamo.disable()
    def _optim_step_pre(self, optimizer, args, kwargs):
        del optimizer, args, kwargs
        if self._active and self.annotate_phases:
            self._close(f"{self.name_prefix}optim_step/pre")
        return None  # NOTE: a non-None return value must be an (args, kwargs) tuple.

    @torch._dynamo.disable()
    def _optim_step_post(self, optimizer, args, kwargs):
        del optimizer, args, kwargs
        if self._active and self.annotate_phases:
            self._close(f"{self.name_prefix}optim_step")

    def pre_load_batch(self):
        if self._handles is None:
            return
        self._drain()  # Safety net in case anything from the last step is still open.
        self._active = self._window_active()
        if self._active and self.annotate_phases:
            self._push(f"{self.name_prefix}data_loading")

    def pre_step(self, batch: Dict[str, Any]):
        del batch
        if self._handles is None or not self._active or not self.annotate_phases:
            return
        self._close(f"{self.name_prefix}data_loading")

    def pre_optim_step(self):
        if self._handles is None or not self._active or not self.annotate_phases:
            return
        self._bwd_finish()  # The backward pass is definitely over by now.
        self._push(f"{self.name_prefix}optim_step")
        # NOTE: '_optim_step_pre' is what closes this, so without the optimizer step hooks it
        # would stay open until 'post_step' drains it -- reporting the whole optimizer step as
        # "pre". Better to omit the sub-range than to report a wrong one.
        if self._optim_step_hooks:
            self._push(f"{self.name_prefix}optim_step/pre")

    def post_step(self):
        if self._handles is not None:
            self._drain()

    def on_error(self, exc: BaseException):
        del exc
        if self._handles is not None:
            self._drain()

    def close(self):
        self._reset()

    def _reset(self):
        if self._fwd_stack is not None or self._bwd_open is not None:
            self._drain()
        if self._handles is not None:
            for handle in self._handles:
                handle.remove()
        self._handles = None
        self._model = None
        self._profiler_cb = None
        self._fwd_stack = None
        self._bwd_open = None
        self._active = False
        self._optim_step_hooks = False

    def _annotating(self) -> bool:
        if self.annotate_eval:
            return True
        return self._model is None or self._model.training

    def _window_active(self) -> bool:
        from torch.profiler import ProfilerAction

        if self.start is not None or self.end is not None:
            step = self.step + 1  # 'global_step' is incremented after 'pre_load_batch'.
            if self.start is not None and step < self.start:
                return False
            if self.end is not None and step > self.end:
                return False
            return True

        profiler = getattr(self._profiler_cb, "_profiler", None)
        if profiler is None:
            return True
        # We run after 'ProfilerCallback.pre_load_batch()' stepped the profiler, so
        # 'current_action' already refers to the step that's about to run.
        action = getattr(profiler, "current_action", _UNKNOWN_ACTION)
        if action is _UNKNOWN_ACTION:
            # 'current_action' is internal to torch.profiler and may not exist on
            # every version. Annotate unconditionally when we can't read it: the
            # profiler only records during its own window, so extra ranges cost a
            # few microseconds and land nowhere, whereas failing closed would
            # silently produce an unannotated trace -- the exact problem this
            # callback exists to fix.
            return True
        return action in (
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
        )
