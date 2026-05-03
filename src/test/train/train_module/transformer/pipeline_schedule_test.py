from collections import defaultdict
from types import SimpleNamespace

import pytest

from olmo_core.train.train_module.transformer.pipeline.helpers import generate_stage_to_rank_mapping
from olmo_core.distributed.parallel.pipeline_parallel import (
    get_pipeline_activation_stats,
    get_pipeline_tick_exchange_stats,
)
from olmo_core.train.train_module.transformer.pipeline.pipeline_schedule import (
    CustomSchedule1F1BV,
    PipelineActionType,
    pad_to_max_length,
)
from olmo_core.train.train_module.transformer.pipeline.pipeline_stage import CustomPipelineStage


def _build_1f1b_v_schedule(
    pp_size: int,
    n_microbatches: int,
    *,
    forward_pull_ahead_extra_activations: int = 0,
) -> CustomSchedule1F1BV:
    schedule = CustomSchedule1F1BV.__new__(CustomSchedule1F1BV)
    schedule.pp_group_size = pp_size
    schedule._num_stages = 2 * pp_size
    schedule._n_microbatches = n_microbatches
    schedule._stages = [SimpleNamespace(stage_index_to_group_rank={}) for _ in range(2)]
    schedule.forward_pull_ahead_extra_activations = forward_pull_ahead_extra_activations
    schedule.configure_pipeline_order()
    return schedule


def _fake_stage(stage_index: int, mapping: dict[int, int]) -> CustomPipelineStage:
    stage = CustomPipelineStage.__new__(CustomPipelineStage)
    stage.stage_index = stage_index
    stage.num_stages = len(mapping)
    stage.group_rank = mapping[stage_index]
    stage.stage_index_to_group_rank = mapping
    stage.received_activations = {}
    stage.received_grads = {}
    stage.fwd_cache = {}
    stage.bwd_cache = {}
    stage.inputs_meta = None
    stage.outputs_meta = None
    return stage


def _activation_residency_peaks(schedule: CustomSchedule1F1BV) -> tuple[dict[int, int], dict[int, int]]:
    held_by_rank = defaultdict(int)
    held_by_stage = defaultdict(int)
    peak_by_rank = defaultdict(int)
    peak_by_stage = defaultdict(int)

    for time_step in range(len(next(iter(schedule.pipeline_order.values())))):
        for rank, actions in schedule.pipeline_order.items():
            action = actions[time_step]
            if action is None:
                continue
            if action.computation_type == PipelineActionType.FORWARD:
                held_by_rank[rank] += 1
                held_by_stage[action.stage_index] += 1
            elif action.computation_type == PipelineActionType.FULL_BACKWARD_CONT:
                held_by_rank[rank] -= 1
                held_by_stage[action.stage_index] -= 1

            peak_by_rank[rank] = max(peak_by_rank[rank], held_by_rank[rank])
            peak_by_stage[action.stage_index] = max(
                peak_by_stage[action.stage_index], held_by_stage[action.stage_index]
            )

    return dict(peak_by_rank), dict(peak_by_stage)


def _format_action(action) -> str:
    if action is None:
        return ".."
    mb = action.microbatch_index + 1
    if action.computation_type == PipelineActionType.FORWARD:
        return f"{action.stage_index}F{mb}"
    if action.computation_type == PipelineActionType.FULL_BACKWARD:
        return f"{action.stage_index}B{mb}"
    if action.computation_type == PipelineActionType.FULL_BACKWARD_CONT:
        return f"{action.stage_index}B_{mb}"
    raise AssertionError(f"unexpected action type: {action.computation_type}")


def test_1f1b_v_mapping():
    assert generate_stage_to_rank_mapping(4, 8, style="v") == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 3,
        5: 2,
        6: 1,
        7: 0,
    }


@pytest.mark.parametrize("pp_size,n_microbatches", [(2, 1), (2, 4), (4, 4), (4, 8), (8, 8)])
def test_1f1b_v_schedule_is_complete_and_owned(pp_size: int, n_microbatches: int):
    schedule = _build_1f1b_v_schedule(pp_size, n_microbatches)
    expected_actions = 2 * pp_size * n_microbatches

    forward_actions = []
    backward_actions = []
    for rank, actions in schedule.pipeline_order.items():
        for action in actions:
            if action is None:
                continue
            assert schedule.stage_index_to_group_rank[action.stage_index] == rank
            if action.computation_type == PipelineActionType.FORWARD:
                forward_actions.append(action)
            elif action.computation_type == PipelineActionType.FULL_BACKWARD:
                backward_actions.append(action)

    assert len(forward_actions) == expected_actions
    assert len(backward_actions) == expected_actions


def test_1f1b_v_schedule_rejects_non_two_stage_placement():
    schedule = CustomSchedule1F1BV.__new__(CustomSchedule1F1BV)
    schedule.pp_group_size = 4
    schedule._num_stages = 12
    schedule._n_microbatches = 4
    schedule._stages = []

    with pytest.raises(ValueError, match="2 virtual stages"):
        schedule.configure_pipeline_order()


def test_1f1b_v_has_only_middle_local_boundary_for_two_chunks_per_rank():
    schedule = _build_1f1b_v_schedule(4, 8)

    local_forward_boundaries = [
        (stage_index, stage_index + 1, schedule.stage_index_to_group_rank[stage_index])
        for stage_index in range(schedule._num_stages - 1)
        if schedule.stage_index_to_group_rank[stage_index]
        == schedule.stage_index_to_group_rank[stage_index + 1]
    ]
    local_backward_boundaries = [
        (stage_index, stage_index - 1, schedule.stage_index_to_group_rank[stage_index])
        for stage_index in range(1, schedule._num_stages)
        if schedule.stage_index_to_group_rank[stage_index]
        == schedule.stage_index_to_group_rank[stage_index - 1]
    ]

    assert local_forward_boundaries == [(3, 4, 3)]
    assert local_backward_boundaries == [(4, 3, 3)]


def test_1f1b_v_middle_rank_warmup_matches_reference_shape():
    schedule = _build_1f1b_v_schedule(5, 8)
    middle_rank_actions = [
        action
        for action in schedule.pipeline_order[4]
        if action is not None
        and action.computation_type
        in (PipelineActionType.FORWARD, PipelineActionType.FULL_BACKWARD)
    ]

    assert [
        (action.stage_index, action.computation_type, action.microbatch_index)
        for action in middle_rank_actions[:10]
    ] == [
        (4, PipelineActionType.FORWARD, 0),
        (5, PipelineActionType.FORWARD, 0),
        (4, PipelineActionType.FORWARD, 1),
        (5, PipelineActionType.FORWARD, 1),
        (4, PipelineActionType.FORWARD, 2),
        (5, PipelineActionType.FORWARD, 2),
        (4, PipelineActionType.FORWARD, 3),
        (5, PipelineActionType.FORWARD, 3),
        (5, PipelineActionType.FULL_BACKWARD, 0),
        (4, PipelineActionType.FULL_BACKWARD, 0),
    ]


def test_1f1b_v_pp5_m8_uses_generic_generator():
    schedule = _build_1f1b_v_schedule(5, 8)

    assert schedule.pipeline_order_source == "generic_symbol_pattern"
    assert not hasattr(CustomSchedule1F1BV, "_generate_1f1bv_reference_pp5_m8_symbol_table")
    assert not hasattr(CustomSchedule1F1BV, "_generate_1f1b_v_reference_pp5_m8_order")

    peak_by_rank, _ = _activation_residency_peaks(schedule)
    assert max(peak_by_rank.values()) <= 2 * 5


def test_1f1b_v_pp5_m8_generic_symbols_flow_through_action_adapter():
    symbols = CustomSchedule1F1BV._generate_1f1bv_symbol_table(5, 8)
    actions = pad_to_max_length(
        CustomSchedule1F1BV._convert_1f1bv_symbols_to_actions(symbols, 5)
    )

    assert set(actions) == set(range(5))
    expected = {(stage_index, mb_index) for stage_index in range(10) for mb_index in range(8)}
    forwards = set()
    backwards = set()
    backward_continuations = set()
    for row in actions.values():
        for action in row:
            if action is None:
                continue
            key = (action.stage_index, action.microbatch_index)
            if action.computation_type == PipelineActionType.FORWARD:
                forwards.add(key)
            elif action.computation_type == PipelineActionType.FULL_BACKWARD:
                backwards.add(key)
            elif action.computation_type == PipelineActionType.FULL_BACKWARD_CONT:
                backward_continuations.add(key)

    assert forwards == expected
    assert backwards == expected
    assert backward_continuations == expected


def test_1f1b_v_symbolic_generator_scales_to_large_supported_case():
    schedule = _build_1f1b_v_schedule(16, 256)
    peak_by_rank, _ = _activation_residency_peaks(schedule)

    assert len(schedule.pipeline_order) == 16
    assert len(next(iter(schedule.pipeline_order.values()))) < 2_000
    assert max(peak_by_rank.values()) <= 2 * (16 - 1)


@pytest.mark.parametrize("pp_size,n_microbatches", [(4, 16), (8, 32)])
def test_1f1b_v_schedule_limits_early_stage_activation_residency(
    pp_size: int, n_microbatches: int
):
    schedule = _build_1f1b_v_schedule(pp_size, n_microbatches)
    peak_by_rank, peak_by_stage = _activation_residency_peaks(schedule)

    assert peak_by_stage[0] <= 2 * pp_size
    assert peak_by_rank[0] <= 2 * pp_size
    assert max(peak_by_rank.values()) <= 2 * pp_size


def test_1f1b_v_forward_pull_ahead_reduces_tick_exchange_edges():
    baseline = _build_1f1b_v_schedule(4, 16)
    pulled = _build_1f1b_v_schedule(4, 16, forward_pull_ahead_extra_activations=1)

    baseline_exchanges = get_pipeline_tick_exchange_stats(baseline.pipeline_order)
    pulled_exchanges = get_pipeline_tick_exchange_stats(pulled.pipeline_order)
    baseline_peaks = get_pipeline_activation_stats(baseline.pipeline_order)
    pulled_peaks = get_pipeline_activation_stats(pulled.pipeline_order)

    assert pulled.pipeline_order_source == "generic_symbol_pattern_pull_fwd_plus1"
    assert pulled_exchanges["tight_edges"] < baseline_exchanges["tight_edges"]
    assert pulled_exchanges["bidirectional_ticks"] < baseline_exchanges["bidirectional_ticks"]
    assert max(pulled_peaks.values()) == max(baseline_peaks.values()) + 1


def test_1f1b_v_forward_pull_ahead_can_be_rank_selective():
    baseline = _build_1f1b_v_schedule(4, 16)
    pulled = _build_1f1b_v_schedule(
        4,
        16,
        forward_pull_ahead_extra_activations={0: 1, 2: 1, 3: 1},
    )

    baseline_exchanges = get_pipeline_tick_exchange_stats(baseline.pipeline_order)
    pulled_exchanges = get_pipeline_tick_exchange_stats(pulled.pipeline_order)
    pulled_peaks = get_pipeline_activation_stats(pulled.pipeline_order)

    assert pulled.pipeline_order_source == "generic_symbol_pattern_pull_fwd_r0p1_r2p1_r3p1"
    assert pulled_exchanges["tight_edges"] < baseline_exchanges["tight_edges"]
    assert pulled_peaks[0] == 7
    assert pulled_peaks[1] == 6
    assert pulled_peaks[2] == 7
    assert pulled_peaks[3] == 7


def test_local_middle_boundary_skips_p2p_without_touching_buffers():
    mapping = generate_stage_to_rank_mapping(4, 8, style="v")
    stage_3 = _fake_stage(3, mapping)
    stage_4 = _fake_stage(4, mapping)

    assert stage_3.has_local_forward_dst()
    assert stage_3.get_fwd_send_ops(0) == []

    stage_4.received_activations["keep"] = "value"
    assert stage_4.has_local_forward_src()
    assert stage_4.get_fwd_recv_ops(0) == []
    assert stage_4.received_activations == {"keep": "value"}

    stage_4.bwd_cache["keep"] = "value"
    assert stage_4.has_local_backward_dst()
    assert stage_4.get_bwd_send_ops(0) == []
    assert stage_4.bwd_cache == {"keep": "value"}

    stage_3.received_grads["keep"] = "value"
    assert stage_3.has_local_backward_src()
    assert stage_3.get_bwd_recv_ops(0) == []
    assert stage_3.received_grads == {"keep": "value"}
