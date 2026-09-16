from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from olmo_core.distributed.parallel.pipeline_parallel import (
    PipelineSchedule,
    draw_pipeline_timeline,
    get_pipeline_bubble_stats,
)


@pytest.mark.parametrize("first,last", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("forward_only,override", [(False, None), (False, 2), (True, None)])
def test_pipeline_step_preserves_stage_inputs_and_restores_microbatches(
    first, last, forward_only, override
):
    schedule = PipelineSchedule.__new__(PipelineSchedule)
    schedule.stages = [SimpleNamespace(is_first=first, is_last=last)]
    expected_output = object()
    implementation = SimpleNamespace(
        _n_microbatches=4,
        prepare_step=Mock(),
        step=Mock(return_value=expected_output),
        clear_step_info=Mock(),
    )

    def reset(count):
        implementation._n_microbatches = count

    implementation.reset_n_microbatches = Mock(side_effect=reset)
    schedule.schedule_impl = implementation
    inputs = torch.zeros((8, 16), dtype=torch.long)
    target = torch.ones_like(inputs)

    assert (
        schedule.step(inputs, target=target, forward_only=forward_only, num_microbatches=override)
        is expected_output
    )
    args, kwargs = implementation.step.call_args
    assert len(args) == int(first)
    if first:
        assert args[0] is inputs
    assert kwargs == {"target": target if last else None, "forward_only": forward_only}
    implementation.prepare_step.assert_called_once_with(global_batch_size=8, seqlen=16)
    implementation.clear_step_info.assert_called_once()
    assert implementation._n_microbatches == 4
    if forward_only or override is not None:
        assert [call.args[0] for call in implementation.reset_n_microbatches.call_args_list] == [
            8 if forward_only else override,
            4,
        ]
    else:
        implementation.reset_n_microbatches.assert_not_called()


def test_pipeline_bubble_stats_handles_unequal_rows():
    assert get_pipeline_bubble_stats({}) == (0, 0, 0.0)
    assert get_pipeline_bubble_stats({0: [object(), None], 1: [object()]}) == (2, 4, 0.5)


def test_pipeline_timeline_preserves_action_widths(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    forward = SimpleNamespace(name="FORWARD")
    backward = SimpleNamespace(name="FULL_BACKWARD")
    continuation = SimpleNamespace(name="FULL_BACKWARD_CONT")
    order = {
        0: [
            (0, forward, 0, False, False),
            (0, backward, 0, False, False),
            (0, continuation, 0, False, False),
        ],
        1: [None, (1, forward, 0, False, False), (1, backward, 0, False, False)],
    }
    output = tmp_path / "schedule.png"
    fig, ax = draw_pipeline_timeline(order, outpath=str(output))
    try:
        assert [patch.get_width() for patch in ax.patches] == [1, 2, 1, 1, 2]
        assert [label.get_text() for label in ax.texts] == ["0F0", "0B0", "1F0", "1B0"]
        assert output.is_file()
    finally:
        plt.close(fig)
