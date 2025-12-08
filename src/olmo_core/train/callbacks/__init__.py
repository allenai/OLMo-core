"""
Trainer :class:`Callback` implementations.
"""

from .batch_size_scheduler import BatchSizeSchedulerCallback
from .beaker import BeakerCallback
from .callback import Callback, CallbackConfig
from .checkpointer import CheckpointerCallback, CheckpointRemovalStrategy
from .comet import CometCallback, CometNotificationSetting
from .config_saver import ConfigSaverCallback
from .console_logger import ConsoleLoggerCallback
from .evaluator_callback import (
    DownstreamEvaluatorCallbackConfig,
    EvaluatorCallback,
    LMEvaluatorCallbackConfig,
)
from .gap_monitor import GAPMonitorCallback
from .garbage_collector import GarbageCollectorCallback
from .gpu_memory_monitor import GPUMemoryMonitorCallback
from .list_checkpointer import ListCheckpointerCallback
from .metric_saver import MetricSaverCallback
from .monkey_patcher import MonkeyPatcherCallback
from .profiler import ProfilerCallback
from .sequence_length_scheduler import SequenceLengthSchedulerCallback
from .slack_notifier import SlackNotificationSetting, SlackNotifierCallback
from .speed_monitor import SpeedMonitorCallback
from .wandb import WandBCallback

__all__ = [
    "Callback",
    "CallbackConfig",
    "CheckpointerCallback",
    "CheckpointRemovalStrategy",
    "CometCallback",
    "CometNotificationSetting",
    "ConfigSaverCallback",
    "ConsoleLoggerCallback",
    "EvaluatorCallback",
    "LMEvaluatorCallbackConfig",
    "DownstreamEvaluatorCallbackConfig",
    "GAPMonitorCallback",
    "GarbageCollectorCallback",
    "GPUMemoryMonitorCallback",
    "ProfilerCallback",
    "SlackNotifierCallback",
    "SlackNotificationSetting",
    "SequenceLengthSchedulerCallback",
    "SpeedMonitorCallback",
    "WandBCallback",
    "BeakerCallback",
    "BatchSizeSchedulerCallback",
    "MonkeyPatcherCallback",
    "MetricSaverCallback",
    "ListCheckpointerCallback",
]

__doc__ += "\n"
for name in __all__[2:]:
    if name.endswith("Callback"):
        __doc__ += f"- :class:`{name}`\n"
