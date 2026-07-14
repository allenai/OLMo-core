import pytest

from olmo_core.train.callbacks.speed_monitor import get_device_peak_flops_per_second
from olmo_core.train.train_module import (
    OLMoDDPTrainModule,
    TransformerPipelineTrainModule,
    TransformerTrainModule,
)


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("NVIDIA B200", int(4.5e15 * 0.5)),
        ("NVIDIA B300", int(4.5e15 * 0.5)),
        ("NVIDIA GB300", int(4.5e15 * 0.5)),
        ("NVIDIA RTX PRO 6000", int(1008e12 * 0.5)),
        ("NVIDIA H100 NVL", int(1671e12 * 0.5)),
    ],
)
def test_gpu_peak_flops_uses_dense_bf16_spec(device_name: str, expected: int):
    assert get_device_peak_flops_per_second(device_name, using_half_precision=True) == expected


def test_device_peak_flops_returns_none_without_half_precision():
    assert get_device_peak_flops_per_second("NVIDIA B300", using_half_precision=False) is None


@pytest.mark.parametrize(
    "train_module_cls",
    [TransformerTrainModule, TransformerPipelineTrainModule, OLMoDDPTrainModule],
)
def test_supported_train_modules_expose_dp_config(train_module_cls):
    # SpeedMonitorCallback.pre_train() reads train_module.dp_config for every supported module type
    # when setting up CUDA throughput metrics. Each must expose it as a property so the precision
    # check never raises AttributeError (a pipeline module storing only _dp_config regressed this).
    assert isinstance(getattr(train_module_cls, "dp_config", None), property)
