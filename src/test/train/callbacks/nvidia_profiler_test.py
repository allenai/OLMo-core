"""Capture start/stop must not depend on optional autograd NVTX annotations."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from olmo_core.train.callbacks.profiler import NvidiaProfilerCallback


@pytest.mark.parametrize("nvtx", [False, True])
@pytest.mark.parametrize("finish", ["step", "close"])
def test_nvidia_capture_lifecycle(nvtx, finish):
    callback = NvidiaProfilerCallback(start=36, end=37, emit_autograd_nvtx=nvtx)
    callback.trainer = SimpleNamespace(global_step=34)
    cudart, context = MagicMock(), MagicMock()
    with patch("torch.cuda.cudart", return_value=cudart), patch(
        "torch.autograd.profiler.emit_nvtx", return_value=context
    ) as emit, patch("olmo_core.train.callbacks.profiler.get_rank", return_value=0):
        callback.pre_load_batch()
        cudart.cudaProfilerStart.assert_not_called()
        callback.trainer.global_step = 35
        callback.pre_load_batch()
        callback.pre_load_batch()
        cudart.cudaProfilerStart.assert_called_once()
        assert emit.call_count == int(nvtx)
        assert context.__enter__.call_count == int(nvtx)
        callback.trainer.global_step = 36
        callback.post_train_batch()
        cudart.cudaProfilerStop.assert_not_called()
        if finish == "step":
            callback.trainer.global_step = 37
            callback.post_train_batch()
        else:
            callback.close()
        callback.close()
        cudart.cudaProfilerStop.assert_called_once()
        assert context.__exit__.call_count == int(nvtx)


def test_unselected_rank():
    callback = NvidiaProfilerCallback(start=36, end=37, emit_autograd_nvtx=False)
    callback.trainer = SimpleNamespace(global_step=35)
    with patch("torch.cuda.cudart") as cudart, patch(
        "olmo_core.train.callbacks.profiler.get_rank", return_value=1
    ):
        callback.pre_load_batch()
        callback.close()
        cudart.assert_not_called()
