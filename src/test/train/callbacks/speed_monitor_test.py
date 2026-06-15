import pytest

from olmo_core.train.callbacks.speed_monitor import get_device_peak_flops_per_second


@pytest.mark.parametrize(
    ("device_name", "expected"),
    [
        ("NVIDIA B200", int(4.5e15 * 0.5)),
        ("NVIDIA B300", int(4.5e15 * 0.5)),
        ("NVIDIA GB300", int(4.5e15 * 0.5)),
    ],
)
def test_blackwell_gpu_peak_flops_uses_dense_bf16_spec(device_name: str, expected: int):
    assert get_device_peak_flops_per_second(device_name, using_half_precision=True) == expected


def test_device_peak_flops_returns_none_without_half_precision():
    assert get_device_peak_flops_per_second("NVIDIA B300", using_half_precision=False) is None
