from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)

from olmo_core.distributed.parallel.pipeline_parallel import (
    PipelineSchedule,
    PipelineScheduleType,
    get_pipeline_activation_stats,
    get_pipeline_tick_exchange_stats,
)
from olmo_core.train.train_module.transformer.pipeline.helpers import (
    generate_stage_to_rank_mapping,
)
from olmo_core.train.train_module.transformer.pipeline.pipeline_schedule import (
    BATCH_LEADING_MODEL_KWARGS,
    SUPPORTED_MODEL_KWARGS,
    CustomSchedule1F1BV,
    CustomScheduleInterleaved1F1B,
    PipelineActionType,
    pad_to_max_length,
)
from olmo_core.train.train_module.transformer.pipeline.pipeline_stage import (
    CustomPipelineStage,
)


def _build_1f1b_v_schedule(
    pp_size: int,
    n_microbatches: int,
    *,
    forward_pull_ahead_extra_activations: Any = 0,
) -> CustomSchedule1F1BV:
    schedule = CustomSchedule1F1BV.__new__(CustomSchedule1F1BV)
    schedule.pp_group_size = pp_size
    schedule._num_stages = 2 * pp_size
    schedule._n_microbatches = n_microbatches
    schedule._stages = [SimpleNamespace(stage_index_to_group_rank={}) for _ in range(2)]  # type: ignore[misc]
    schedule.forward_pull_ahead_extra_activations = forward_pull_ahead_extra_activations
    schedule.configure_pipeline_order()
    return schedule


def _build_interleaved_1f1b_schedule(
    pp_size: int,
    n_microbatches: int,
) -> CustomScheduleInterleaved1F1B:
    schedule = CustomScheduleInterleaved1F1B.__new__(CustomScheduleInterleaved1F1B)
    schedule.pp_group_size = pp_size
    schedule._num_stages = 2 * pp_size
    schedule.n_local_stages = 2
    schedule.enable_activation_offload_schedule = False
    schedule.reset_n_microbatches(n_microbatches)
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


def _activation_residency_peaks(
    schedule: CustomSchedule1F1BV,
) -> tuple[dict[int, int], dict[int, int]]:
    held_by_rank: Dict[int, int] = defaultdict(int)
    held_by_stage: Dict[int, int] = defaultdict(int)
    peak_by_rank: Dict[int, int] = defaultdict(int)
    peak_by_stage: Dict[int, int] = defaultdict(int)

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


def test_interleaved_1f1b_p2p_overlap_ignores_backward_continuation_slots():
    schedule = _build_interleaved_1f1b_schedule(8, 64)
    rank_1_actions = schedule.pipeline_order[1]

    action_names = [str(action) if action is not None else ".." for action in rank_1_actions]
    assert action_names[42:46] == ["1F15", "9B3", "9B_3", "9F8"]

    overlap_steps = []
    completed_overlap_steps = 0
    for action in rank_1_actions:
        if schedule._action_advances_p2p_overlap(action):
            completed_overlap_steps += 1
        overlap_steps.append(completed_overlap_steps)

    p2p_after_9b3_launch_step = overlap_steps[43]
    assert overlap_steps[44] == p2p_after_9b3_launch_step
    assert overlap_steps[45] == p2p_after_9b3_launch_step + 1


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
    actions = pad_to_max_length(CustomSchedule1F1BV._convert_1f1bv_symbols_to_actions(symbols, 5))

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
def test_1f1b_v_schedule_limits_early_stage_activation_residency(pp_size: int, n_microbatches: int):
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

    stage_4.received_activations["keep"] = "value"  # type: ignore[index]
    assert stage_4.has_local_forward_src()
    assert stage_4.get_fwd_recv_ops(0) == []
    assert stage_4.received_activations == {"keep": "value"}

    stage_4.bwd_cache["keep"] = "value"  # type: ignore[index]
    assert stage_4.has_local_backward_dst()
    assert stage_4.get_bwd_send_ops(0) == []
    assert stage_4.bwd_cache == {"keep": "value"}

    stage_3.received_grads["keep"] = "value"  # type: ignore[index]
    assert stage_3.has_local_backward_src()
    assert stage_3.get_bwd_recv_ops(0) == []
    assert stage_3.received_grads == {"keep": "value"}


@pytest.mark.parametrize(
    ("schedule_name", "schedule_base", "num_parts"),
    [
        (PipelineScheduleType.single_1F1B, PipelineScheduleSingle, 1),
        (PipelineScheduleType.interleaved_1F1B, PipelineScheduleMulti, 2),
        (PipelineScheduleType.gpipe, PipelineScheduleSingle, 1),
    ],
)
def test_pipeline_schedule_builds_standard_schedules(
    monkeypatch, schedule_name: PipelineScheduleType, schedule_base: type, num_parts: int
):
    class FakeSchedule(schedule_base):
        def __init__(self, stages, *, n_microbatches, loss_fn):
            self.stages = stages
            self._n_microbatches = n_microbatches
            self._loss_fn = loss_fn

        def _step_microbatches(self, *args, **kwargs):
            del args, kwargs

    monkeypatch.setattr(
        "olmo_core.distributed.parallel.pipeline_parallel.get_schedule_class",
        lambda _: FakeSchedule,
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    stages = [SimpleNamespace(is_first=True, is_last=True) for _ in range(num_parts)]

    def loss_fn(output, target):
        del target
        return output

    schedule = PipelineSchedule(
        model_parts=[torch.nn.Identity() for _ in range(num_parts)],
        stages=stages,  # type: ignore[arg-type]
        pp_mesh=SimpleNamespace(size=lambda: 2),  # type: ignore[arg-type]
        schedule_name=schedule_name,
        loss_fn=loss_fn,
    )

    assert isinstance(schedule.schedule_impl, FakeSchedule)
    assert schedule.num_microbatches == 2
    assert schedule.schedule_impl._loss_fn is loss_fn


def _build_splitter(n_microbatches: int) -> CustomScheduleInterleaved1F1B:
    schedule = CustomScheduleInterleaved1F1B.__new__(CustomScheduleInterleaved1F1B)
    schedule._n_microbatches = n_microbatches
    schedule._args_chunk_spec = None
    schedule._kwargs_chunk_spec = None
    return schedule


def test_split_inputs_keeps_segment_ids_aligned_with_input_ids():
    splitter = _build_splitter(2)
    input_ids = torch.arange(16).reshape(4, 4)
    segment_ids = torch.arange(16).reshape(4, 4) // 2

    args_split, kwargs_split = splitter._split_inputs((input_ids,), {"segment_ids": segment_ids})

    assert len(args_split) == len(kwargs_split) == 2
    for i, (start, stop) in enumerate([(0, 2), (2, 4)]):
        assert torch.equal(args_split[i][0], input_ids[start:stop])
        assert torch.equal(kwargs_split[i]["segment_ids"], segment_ids[start:stop])


def test_split_inputs_splits_packed_document_metadata():
    splitter = _build_splitter(2)
    input_ids = torch.arange(16).reshape(4, 4)
    doc_lens = torch.tensor([[4, 0], [2, 2], [3, 1], [4, 0]])
    max_doc_lens = [4, 2, 3, 4]

    _, kwargs_split = splitter._split_inputs(
        (input_ids,), {"doc_lens": doc_lens, "max_doc_lens": max_doc_lens}
    )

    assert torch.equal(kwargs_split[0]["doc_lens"], doc_lens[0:2])
    assert torch.equal(kwargs_split[1]["doc_lens"], doc_lens[2:4])
    # The Python list has to be sliced on the same boundaries as the tensors.
    assert kwargs_split[0]["max_doc_lens"] == [4, 2]
    assert kwargs_split[1]["max_doc_lens"] == [3, 4]


def test_split_inputs_rejects_uneven_microbatches():
    # Stages size their P2P buffers from one floor-divided microbatch shape, so a batch that
    # doesn't divide evenly would leave receivers with undersized buffers.
    splitter = _build_splitter(4)

    with pytest.raises(ValueError, match="not divisible"):
        splitter._split_inputs((torch.arange(24).reshape(6, 4),), {})


@pytest.mark.parametrize("present_keys", [("labels",), ("segment_ids",), ("max_doc_lens",)])
def test_split_inputs_infers_batch_size_on_later_stages(present_keys):
    # Ranks that don't own the first stage get empty positional args, so the batch size has to come
    # from whichever batch-leading kwargs happen to be present.
    splitter = _build_splitter(2)
    available = {
        "labels": torch.arange(16).reshape(4, 4),
        "segment_ids": torch.zeros(4, 4, dtype=torch.long),
        "max_doc_lens": [4, 4, 4, 4],
    }
    kwargs = {key: available[key] for key in present_keys}

    args_split, kwargs_split = splitter._split_inputs((), kwargs)

    assert args_split == [(), ()]
    for key in present_keys:
        assert len(kwargs_split[0][key]) == 2
        assert len(kwargs_split[1][key]) == 2


def test_split_inputs_rejects_metadata_with_mismatched_batch_size():
    splitter = _build_splitter(2)

    with pytest.raises(ValueError, match="does not match input batch size"):
        splitter._split_inputs((torch.arange(16).reshape(4, 4),), {"max_doc_lens": [1, 2, 3]})


def test_supported_model_kwargs_covers_every_batch_leading_kwarg():
    # The independent PP dry run validates against this same set, so a kwarg the splitter accepts
    # but the set omits would train fine and then fail the dry run.
    assert set(BATCH_LEADING_MODEL_KWARGS) <= SUPPORTED_MODEL_KWARGS


@pytest.mark.parametrize("key", sorted(SUPPORTED_MODEL_KWARGS - {"labels"}))
def test_split_inputs_accepts_every_supported_model_kwarg(key):
    splitter = _build_splitter(2)
    values = {
        "segment_ids": torch.zeros(4, 4, dtype=torch.long),
        "doc_lens": torch.tensor([[4, 0], [2, 2], [3, 1], [4, 0]]),
        "max_doc_lens": [4, 2, 3, 4],
        "loss_div_factor": 2.0,
        "ignore_index": -100,
        "loss_reduction": "sum",
        "z_loss_multiplier": None,
        "return_logits": False,
        "cp_already_sharded": False,
        "cp_original_seq_len": 4,
    }
    _, kwargs_split = splitter._split_inputs((torch.arange(16).reshape(4, 4),), {key: values[key]})
    assert all(key in kwargs_mb for kwargs_mb in kwargs_split)
