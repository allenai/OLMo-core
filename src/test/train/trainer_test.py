import torch

from olmo_core.train.trainer import _metric_value_to_tensor


def test_metric_value_to_tensor_accepts_large_python_int():
    value = 10**20
    tensor = _metric_value_to_tensor(value)

    assert tensor.dtype == torch.float64
    assert tensor.item() == float(value)
