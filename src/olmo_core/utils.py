import dataclasses
import gc
import logging
import os
import socket
import sys
import time
import uuid
import warnings
from datetime import datetime
from itertools import cycle, islice
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, Iterable, Optional, TypeVar, Union

import rich
import torch
import torch.nn as nn
from rich.console import Console, ConsoleRenderable
from rich.highlighter import NullHighlighter
from rich.text import Text
from rich.traceback import Traceback

from .config import StrEnum
from .exceptions import OLMoCLIError, OLMoEnvironmentError, OLMoError, OLMoThreadError

OLMO_NUM_THREADS_ENV_VAR = "OLMO_NUM_THREADS"


_log_extra_fields: Dict[str, Any] = {}
log = logging.getLogger(__name__)


def generate_uuid() -> str:
    return str(uuid.uuid4())


def get_default_thread_count() -> int:
    env_val = os.environ.get(OLMO_NUM_THREADS_ENV_VAR)
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            raise OLMoEnvironmentError(
                f"Invalid value for {OLMO_NUM_THREADS_ENV_VAR} environment variable ('{env_val}')"
            )
    else:
        return min(16, (os.cpu_count() or 1) + 4)


def wait_for(condition: Callable[[], bool], description: str, timeout: float = 10.0):
    """Wait for the condition function to return True."""
    start_time = time.monotonic()
    while not condition():
        time.sleep(0.5)
        if time.monotonic() - start_time > timeout:
            raise TimeoutError(f"{description} timed out")


def apply_to_tensors(fn, container: Any) -> None:
    """
    Recursively apply ``fn`` to all tensors in a container.
    """
    if isinstance(container, torch.Tensor):
        fn(container)
    elif isinstance(container, (list, tuple, set)):
        for x in container:
            apply_to_tensors(fn, x)
    elif isinstance(container, dict):
        for k, v in container.items():
            apply_to_tensors(fn, k)
            apply_to_tensors(fn, v)
    elif hasattr(container, "__dataclass_fields__"):
        for f in dataclasses.fields(container):
            name = f.name
            apply_to_tensors(fn, getattr(container, name))
    elif hasattr(container, "__next__"):
        for x in container:
            apply_to_tensors(fn, x)


T = TypeVar("T")


def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o


def get_default_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def seed_all(seed: int):
    """Seed all rng objects."""
    import random

    import numpy as np

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


def get_grad_norm(params: Iterable[nn.Parameter], norm_type: float) -> torch.Tensor:
    """
    Return the gradient norm of parameters, where the gradients are viewed as a single vector.

    The returned norm is in FP32 even if parameters/gradients are in a low precision. This is because the downstream
    use of this return value is a reduction across ranks.
    """
    grads = [param.grad for param in params if param.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    grad_dtypes = {grad.dtype for grad in grads}
    if len(grad_dtypes) != 1:
        raise ValueError(f"Requires uniform dtype across all gradients but got {grad_dtypes}")
    # Compute the gradient norm in FP32, where we treat the gradients as a
    # single vector
    grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(grad.detach(), norm_type, dtype=torch.float32)
                for grad in grads
            ],
        ),
        norm_type,
        dtype=torch.float32,
    )
    return grad_norm


def same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    """
    Check if two tensors share the same storage.
    """
    x_ptrs = set(e.data_ptr() for e in x.view(-1))
    y_ptrs = set(e.data_ptr() for e in y.view(-1))
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)


def gc_cuda():
    """
    Run CUDA garbage collection.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_document_lengths(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    doc_boundaries = torch.cat(
        [
            torch.tensor([-1], dtype=torch.int32),
            (input_ids == eos_token_id).nonzero(as_tuple=True)[0].to(dtype=torch.int32),
            torch.tensor(
                [] if input_ids[-1] == eos_token_id else [input_ids.shape[0] - 1], dtype=torch.int32
            ),
        ]
    )
    return doc_boundaries[1:] - doc_boundaries[:-1]


def get_cumulative_document_lengths(doc_lens: torch.Tensor) -> torch.Tensor:
    """
    Transform a batched tensor of document lengths into a 1D tensor of cumulative document
    lengths for the whole batch.
    """
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=doc_lens.device),
            torch.cumsum(doc_lens.masked_select(doc_lens != 0), 0, dtype=torch.int32),
        ]
    )


def has_flash_attn() -> bool:
    """
    Check if flash-attn is available.
    """
    try:
        import flash_attn  # type: ignore

        del flash_attn
        return True
    except ModuleNotFoundError:
        return False


class LogFilterType(StrEnum):
    rank0_only = "rank0_only"
    local_rank0_only = "local_rank0_only"
    all_ranks = "all_ranks"


def log_extra_field(field_name: str, field_value: Any) -> None:
    global _log_extra_fields
    if field_value is None:
        if field_name in _log_extra_fields:
            del _log_extra_fields[field_name]
    else:
        _log_extra_fields[field_name] = field_value


def setup_logging(log_filter_type: LogFilterType = LogFilterType.rank0_only) -> None:
    """
    :param rank0_only: INFO and below messages will only be emitted on the rank0 process.
    """
    from .distributed.utils import get_local_rank, get_rank

    log_extra_field("hostname", socket.gethostname())
    log_extra_field("local_rank", get_local_rank())
    log_extra_field("global_rank", get_rank())

    old_log_record_factory = logging.getLogRecordFactory()

    def log_record_factory(*args, **kwargs) -> logging.LogRecord:
        record = old_log_record_factory(*args, **kwargs)
        for field_name, field_value in _log_extra_fields.items():
            setattr(record, field_name, field_value)
        return record

    logging.setLogRecordFactory(log_record_factory)

    handler: logging.Handler
    if (
        os.environ.get("OLMo_NONINTERACTIVE", False)
        or os.environ.get("DEBIAN_FRONTEND", None) == "noninteractive"
        or not sys.stdout.isatty()
    ):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s\t%(hostname)s:%(local_rank)s\t%(name)s:%(lineno)s\t%(levelname)s\t%(message)s"
        )
        formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
        formatter.default_msec_format = "%s.%03d"
        handler.setFormatter(formatter)
    else:
        handler = RichHandler()

    def rank0_filter(record: logging.LogRecord) -> int:
        if record.levelno > logging.INFO:
            return 1
        if getattr(record, "global_rank", 0) == 0:
            return 1
        else:
            return 0

    def local_rank0_filter(record: logging.LogRecord) -> int:
        if record.levelno > logging.INFO:
            return 1
        if getattr(record, "local_rank", 0) == 0:
            return 1
        else:
            return 0

    if log_filter_type == LogFilterType.rank0_only:
        filter = rank0_filter
    elif log_filter_type == LogFilterType.local_rank0_only:
        filter = local_rank0_filter  # type: ignore
    elif log_filter_type == LogFilterType.all_ranks:
        filter = None
    else:
        raise ValueError(log_filter_type)

    if filter is not None:
        handler.addFilter(filter)  # type: ignore
    logging.basicConfig(handlers=[handler], level=logging.INFO)

    logging.captureWarnings(True)
    logging.getLogger("urllib3").setLevel(logging.ERROR)


def excepthook(exctype, value, traceback):
    """
    Used to patch `sys.excepthook` in order to log exceptions.
    """
    if issubclass(exctype, KeyboardInterrupt):
        sys.__excepthook__(exctype, value, traceback)
    elif issubclass(exctype, OLMoCLIError):
        rich.get_console().print(f"[yellow]{value}[/]", highlight=False)
    elif issubclass(exctype, OLMoError):
        rich.get_console().print(Text(f"{exctype.__name__}:", style="red"), value, highlight=False)
    else:
        log.critical(
            "Uncaught %s: %s", exctype.__name__, value, exc_info=(exctype, value, traceback)
        )


def install_excepthook():
    sys.excepthook = excepthook


def filter_warnings():
    # Filter internal deprecation warnings from torch
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="torch.distributed.*_base is a private function and will be deprecated.*",
    )
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="TypedStorage is deprecated.*",
    )
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message="Please use DTensor instead.*",
    )
    warnings.filterwarnings(
        action="ignore",
        category=FutureWarning,
        message="You are using `torch.load` with `weights_only=False`.*",
    )
    warnings.filterwarnings(
        action="ignore",
        category=FutureWarning,
        module="flash_attn.ops.triton.layer_norm",
    )
    # Torchvision warnings. We don't actually use torchvision.
    warnings.filterwarnings(
        action="ignore",
        message="failed to load.*",
        module="torchvision.io.image",
    )


def set_env_variables():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_cli_environment(log_filter_type: Optional[LogFilterType] = None):
    if log_filter_type is None:
        log_filter_type = LogFilterType(os.environ.get("LOG_FILTER_TYPE", "rank0_only"))
    rich.reconfigure(width=max(rich.get_console().width, 180), soft_wrap=True)
    setup_logging(log_filter_type=log_filter_type)
    install_excepthook()
    filter_warnings()
    set_env_variables()


class RichHandler(logging.Handler):
    """
    A simplified version of rich.logging.RichHandler from
    https://github.com/Textualize/rich/blob/master/rich/logging.py
    """

    def __init__(
        self,
        *,
        level: Union[int, str] = logging.NOTSET,
        console: Optional[Console] = None,
        markup: bool = False,
    ) -> None:
        super().__init__(level=level)
        self.console = console or rich.get_console()
        self.highlighter = NullHighlighter()
        self.markup = markup

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if hasattr(record.msg, "__rich__") or hasattr(record.msg, "__rich_console__"):
                self.console.print(record.msg)
            else:
                msg: Any = record.msg
                if isinstance(record.msg, str):
                    msg = self.render_message(record=record, message=record.getMessage())
                renderables = [
                    self.get_time_text(record),
                    self.get_level_text(record),
                    self.get_location_text(record),
                    msg,
                ]
                if record.exc_info is not None:
                    tb = Traceback.from_exception(*record.exc_info)  # type: ignore
                    renderables.append(tb)
                self.console.print(*renderables)
        except Exception:
            self.handleError(record)

    def render_message(self, *, record: logging.LogRecord, message: str) -> ConsoleRenderable:
        use_markup = getattr(record, "markup", self.markup)
        message_text = Text.from_markup(message) if use_markup else Text(message)

        highlighter = getattr(record, "highlighter", self.highlighter)
        if highlighter:
            message_text = highlighter(message_text)

        return message_text

    def get_time_text(self, record: logging.LogRecord) -> Text:
        log_time = datetime.fromtimestamp(record.created)
        time_str = log_time.strftime("[%Y-%m-%d %X]")
        return Text(time_str, style="log.time", end=" ")

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname
        level_text = Text.styled(level_name.ljust(8), f"logging.level.{level_name.lower()}")
        level_text.style = "log.level"
        level_text.end = " "
        return level_text

    def get_location_text(self, record: logging.LogRecord) -> Text:
        name_and_line = f"{record.name}:{record.lineno}" if record.name != "root" else "root"
        text = f"[{name_and_line}, rank={record.local_rank}]"  # type: ignore
        return Text(text, style="log.path")


def threaded_generator(g, maxsize: int = 16, thread_name: Optional[str] = None):
    q: Queue = Queue(maxsize=maxsize)

    sentinel = object()

    def fill_queue():
        try:
            for value in g:
                q.put(value)
        except Exception as e:
            q.put(e)
        finally:
            q.put(sentinel)

    thread_name = thread_name or repr(g)
    thread = Thread(name=thread_name, target=fill_queue, daemon=True)
    thread.start()

    for x in iter(q.get, sentinel):
        if isinstance(x, Exception):
            raise OLMoThreadError(f"generator thread {thread_name} failed") from x
        else:
            yield x


def roundrobin(*iterables):
    """
    Call the given iterables in a round-robin fashion. For example:
    ``roundrobin('ABC', 'D', 'EF') --> A D E B F C``
    """
    # Adapted from https://docs.python.org/3/library/itertools.html#itertools-recipes
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))
