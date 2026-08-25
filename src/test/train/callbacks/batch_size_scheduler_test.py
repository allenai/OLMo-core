from types import SimpleNamespace
from unittest.mock import Mock

from olmo_core.train.callbacks.batch_size_scheduler import BatchSizeSchedulerCallback
from olmo_core.train.train_module import OLMoDDPTrainModule


def test_batch_size_change_rebuilds_olmo_ddp_pipeline_schedule():
    train_module = OLMoDDPTrainModule.__new__(OLMoDDPTrainModule)
    train_module.rebuild_train_pp_schedule = Mock()
    train_module.optim = None
    train_module.scheduler = None
    data_loader = SimpleNamespace(global_batch_size=8)
    callback = BatchSizeSchedulerCallback()
    callback.trainer = SimpleNamespace(
        train_module=train_module,
        data_loader=data_loader,
        callbacks={},
    )

    callback._update_batch_size_and_lr(16)

    train_module.rebuild_train_pp_schedule.assert_called_once_with(16)
    assert data_loader.global_batch_size == 16
