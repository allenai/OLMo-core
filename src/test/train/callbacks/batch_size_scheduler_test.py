import copy
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim import INITIAL_LR_FIELD, LR_FIELD, OLMoDDPOptimizer
from olmo_core.train.callbacks.batch_size_scheduler import BatchSizeSchedulerCallback
from olmo_core.train.common import Duration
from olmo_core.train.train_module import OLMoDDPTrainModule, TransformerTrainModule


def _build_callback(module_type, *, pp_enabled=False, enabled=True, step=0):
    module = object.__new__(module_type)
    module._pp_config = SimpleNamespace(degree=2) if pp_enabled else None
    module._train_pp_schedule = SimpleNamespace(num_microbatches=4)
    module.scheduler = None
    optimizer_type = OLMoDDPOptimizer if module_type is OLMoDDPTrainModule else torch.optim.AdamW
    module.optim = Mock(
        spec=optimizer_type,
        param_groups=[{LR_FIELD: 0.01, INITIAL_LR_FIELD: 0.01}],
    )
    trainer = SimpleNamespace(
        train_module=module,
        data_loader=SimpleNamespace(global_batch_size=64, batches_processed=step),
        global_step=step,
        global_train_tokens_seen=step * 64,
        epoch=1,
        callbacks={},
    )
    callback = BatchSizeSchedulerCallback(
        batch_sizes=[32, 128],
        schedule=[Duration.steps(0), Duration.steps(10)],
        enabled=enabled,
    )
    callback.trainer = trainer
    return callback, trainer


@pytest.mark.parametrize("hook", ["post_attach", "pre_load_batch", "post_checkpoint_loaded"])
@pytest.mark.parametrize("step", [0, 10])
@pytest.mark.parametrize("enabled", [False, True])
def test_olmo_ddp_pipeline_batch_schedule_guard_preserves_state(hook, step, enabled):
    callback, trainer = _build_callback(
        OLMoDDPTrainModule, pp_enabled=True, enabled=enabled, step=step
    )
    before = copy.deepcopy(trainer.train_module.optim.param_groups)

    if enabled:
        with pytest.raises(OLMoConfigurationError, match="batch-size schedules.*pipeline"):
            getattr(callback, hook)()
    else:
        getattr(callback, hook)()

    assert trainer.data_loader.global_batch_size == 64
    assert trainer.train_module.optim.param_groups == before
    assert trainer.train_module._train_pp_schedule.num_microbatches == 4


@pytest.mark.parametrize("hook", ["post_attach", "pre_load_batch", "post_checkpoint_loaded"])
@pytest.mark.parametrize("step,batch_size", [(0, 32), (10, 128)])
@pytest.mark.parametrize("module_type", [TransformerTrainModule, OLMoDDPTrainModule])
def test_non_pipeline_batch_schedule_preserves_batch_and_lr_updates(
    hook, step, batch_size, module_type
):
    callback, trainer = _build_callback(module_type, step=step)
    getattr(callback, hook)()

    assert trainer.data_loader.global_batch_size == batch_size
    group = trainer.train_module.optim.param_groups[0]
    expected_lr = 0.01 * math.sqrt(batch_size / 64)
    assert group[LR_FIELD] == pytest.approx(expected_lr)
    assert group[INITIAL_LR_FIELD] == pytest.approx(expected_lr)


@pytest.mark.parametrize("batch_sizes", [[], [64]])
def test_olmo_ddp_pipeline_accepts_no_batch_size_changes(batch_sizes):
    callback, trainer = _build_callback(OLMoDDPTrainModule, pp_enabled=True)
    callback.batch_sizes = batch_sizes
    callback.schedule = [Duration.steps(0)] if batch_sizes else []
    callback.post_attach()
    callback.pre_load_batch()
    callback.post_checkpoint_loaded()
    assert trainer.data_loader.global_batch_size == 64
    assert trainer.train_module.optim.param_groups[0][LR_FIELD] == 0.01
