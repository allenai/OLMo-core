import dataclasses
import gc
import logging
import os
import socket
import sys
import time
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime
from itertools import cycle, islice
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import rich
import torch
from rich.console import Console, ConsoleRenderable
from rich.highlighter import NullHighlighter
from rich.text import Text
from rich.traceback import Traceback

from .config import StrEnum
from .exceptions import OLMoCLIError, OLMoEnvironmentError, OLMoError, OLMoThreadError

OLMO_NUM_THREADS_ENV_VAR = "OLMO_NUM_THREADS"
LOG_FILTER_TYPE_ENV_VAR = "LOG_FILTER_TYPE"


_log_extra_fields: Dict[str, Any] = {}
_LOGGING_CONFIGURED: bool = False
log = logging.getLogger(__name__)


def generate_uuid() -> str:
    """
    Generate a unique ID.
    """
    return str(uuid.uuid4())


def get_default_thread_count() -> int:
    """
    Get the default maximum number of threads allowed.
    """
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
            raise TimeoutError(f"timed out {description}")


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


def move_to_device(o: T, device: torch.device, non_blocking: Optional[bool] = None) -> T:
    """
    Move a tensor or container of tensors to the given device.

    :param o: The object to move.
    :param device: The device to move to.
    """
    if non_blocking is None:
        non_blocking = device.type != "cpu"
    if isinstance(o, torch.Tensor):
        return o.to(device, non_blocking=non_blocking)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o


def get_default_device() -> torch.device:
    """
    Get the default device.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def seed_all(seed: int):
    """
    Seed all RNG states.
    """
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


def same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    """
    Check if two tensors share the same storage.
    """
    x_ptrs = set(e.data_ptr() for e in x.view(-1))
    y_ptrs = set(e.data_ptr() for e in y.view(-1))
    return (x_ptrs <= y_ptrs) or (y_ptrs <= x_ptrs)


def gc_cuda():
    """
    Run garbage collection, including emptying the CUDA cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def set_env_var(name: str, value: str, override: bool = False, secret: bool = False):
    value_str = "****" if secret else value
    if name in os.environ:
        if override and os.environ[name] != value:
            msg = f"Overriding env var '{name}' to '{value_str}'"
            if logging_configured():
                log.warning(msg)
            else:
                print(msg)
            os.environ[name] = value
    else:
        msg = f"Setting env var '{name}' to '{value_str}'"
        if logging_configured():
            log.info(msg)
        else:
            print(msg)
        os.environ[name] = value


class LogFilterType(StrEnum):
    """
    Determines which ranks are allowed to emit INFO messages.
    """

    rank0_only = "rank0_only"
    """
    INFO messages are only emitted from the global rank 0.
    """

    local_rank0_only = "local_rank0_only"
    """
    INFO messages are only emitted from the local (node) rank 0.
    """

    all_ranks = "all_ranks"
    """
    All ranks emit INFO messages.
    """


def log_extra_field(field_name: str, field_value: Any) -> None:
    """
    Add an additional field to each log record.

    .. note::
        For these fields to actually show up in the logs you need to use a formatter/handler
        that displays them.

    :param field_name: The name of the field to attach.
    :param field_value: The value of the field to attach.
    """
    global _log_extra_fields
    if field_value is None:
        if field_name in _log_extra_fields:
            del _log_extra_fields[field_name]
    else:
        _log_extra_fields[field_name] = field_value


def setup_logging(
    log_filter_type: LogFilterType = LogFilterType.rank0_only, force: bool = False
) -> None:
    """
    Configure logging.

    .. seealso::
        :func:`prepare_cli_environment()`

    :param log_filter_type: Which ranks emit INFO and below messages.
    :param force: Force configuring logging even if it was already configured.
    """
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED and not force:
        return

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
        handler = _RichHandler()

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

    _LOGGING_CONFIGURED = True


def logging_configured() -> bool:
    """
    Returns ``True`` if logging has been configured (like with :func:`setup_logging()`),
    otherwise returns ``False``.
    """
    if _LOGGING_CONFIGURED:
        return True
    else:
        # Otherwise check if the root logger has any handlers.
        return len(logging.getLogger().handlers) > 0


def excepthook(exctype, value, traceback):
    """
    Used to patch ``sys.excepthook`` in order to log exceptions. Use :func:`install_excepthook()`
    to install this.
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
    """
    Install the custom :func:`excepthook`.

    .. seealso::
        :func:`prepare_cli_environment()`
    """
    sys.excepthook = excepthook


def filter_warnings():
    """
    Configure warning filters for warnings we don't need to see.

    .. seealso::
        :func:`prepare_cli_environment()`
    """
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
        category=UserWarning,
        message="Synchronization debug mode is a prototype feature.*",
        module="torch.cuda",
    )
    warnings.filterwarnings(
        action="ignore",
        category=FutureWarning,
        message="You are using `torch.load` with `weights_only=False`.*",
    )
    # flash_attn warnings.
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
    """
    Set common needed env vars if they're not already set.

    .. seealso::
        :func:`prepare_cli_environment()`
    """
    set_env_var("OMP_NUM_THREADS", "8")
    set_env_var("TOKENIZERS_PARALLELISM", "false")


def prepare_cli_environment(log_filter_type: Optional[LogFilterType] = None):
    """
    Prepare the environment for a script/CLI.
    This should be called at the very beginning of the script/command, like at the top
    of the ``if __name__ == "__main__": ...`` block.

    Internally this calls:

    - :func:`setup_logging()`
    - :func:`install_excepthook()`
    - :func:`filter_warnings()`
    - :func:`set_env_variables()`
    - :func:`~olmo_core.io.add_cached_path_clients()`

    .. tip::
        If you're looking to setup the environment specifically for distributed training,
        see :func:`~olmo_core.train.prepare_training_environment` instead.

    :param log_filter_type: Determines which ranks are allowed to emit log messages below the
        ``WARNING`` level. You can also configure this through the env var ``LOG_FILTER_TYPE``.
        If neither are set, this defaults to "rank0_only".

        .. note::
            All ranks will always emit messages at the ``WARNING`` level or higher.
    """
    from .io import add_cached_path_clients

    if log_filter_type is None:
        log_filter_type = LogFilterType(os.environ.get(LOG_FILTER_TYPE_ENV_VAR, "rank0_only"))
    rich.reconfigure(width=max(rich.get_console().width, 180), soft_wrap=True)
    setup_logging(log_filter_type=log_filter_type)
    install_excepthook()
    filter_warnings()
    set_env_variables()
    add_cached_path_clients()


class _RichHandler(logging.Handler):
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
    """
    Wraps a generator ``g`` and runs it in a thread.
    """
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


def powers_of_2(x: int) -> List[int]:
    powers: List[int] = []
    i = 1
    while i <= x:
        if i & x:
            powers.insert(0, i)
        i <<= 1
    return powers


def capped_powers_of_2(x: int, cap: int) -> List[int]:
    powers = []
    for i in powers_of_2(x):
        if i > cap:
            assert i % cap == 0
            for _ in range(i // cap):
                powers.append(cap)
        else:
            powers.append(i)
    return powers


def format_float(value: float) -> str:
    if value == 0.0:
        return "0.0"
    elif value < 0.0001:
        return f"{value:.2E}"
    elif value > 1000:
        return f"{int(value):,d}"
    elif value > 100:
        return f"{value:.1f}"
    elif value > 10:
        return f"{value:.2f}"
    elif value > 1:
        return f"{value:.3f}"
    else:
        return f"{value:.4f}"


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested dictionary with strings keys using dot notation.
    """
    out = {}

    def add_sub_dict(prefix: str, sub_dict: Dict[str, Any]):
        for k, v in sub_dict.items():
            if isinstance(v, dict):
                add_sub_dict(f"{prefix}.{k}", v)
            else:
                out[f"{prefix}.{k}"] = v

    for k, v in d.items():
        if isinstance(v, dict):
            add_sub_dict(k, v)
        else:
            out[k] = v

    return out


@contextmanager
def cuda_sync_debug_mode(debug_mode: Union[int, str]):
    """
    A context manager for temporarily setting the CUDA sync debug mode.
    """
    current_mode: Optional[int] = None

    try:
        if torch.cuda.is_available():
            current_mode = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode(debug_mode)
        yield
    finally:
        if current_mode is not None:
            torch.cuda.set_sync_debug_mode(debug_mode)
