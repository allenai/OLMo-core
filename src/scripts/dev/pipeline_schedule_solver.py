#!/usr/bin/env python
"""Experimental exact solver for 1F1B-V pipeline schedules.

This is an offline oracle/prototyping tool, not a production scheduler.
It requires OR-Tools when actually solving:

    pip install ortools

The training code should use a deterministic pattern generator once we derive
one; this script exists to make the constraint problem explicit and test small
instances.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Kind(str, Enum):
    F = "F"
    B = "B"
    B_CONT = "B_"


@dataclass(frozen=True)
class Action:
    stage: int
    kind: Kind
    microbatch: int

    @property
    def label(self) -> str:
        mb = self.microbatch + 1
        if self.kind == Kind.F:
            return f"{self.stage}F{mb}"
        elif self.kind == Kind.B:
            return f"{self.stage}B{mb}"
        else:
            return f"{self.stage}B_{mb}"


def rank_for_stage(stage: int, pp_size: int) -> int:
    return min(stage, 2 * pp_size - 1 - stage)


def conservative_horizon(pp_size: int, n_microbatches: int) -> int:
    # Per rank useful work is 2 stages * m * (1 F slot + 2 B slots) = 6m.
    # Add a generous p-dependent bubble allowance for oracle experiments.
    return 6 * n_microbatches + 12 * pp_size + 32


def validate_table(
    table: dict[int, list[Action | None]],
    pp_size: int,
    n_microbatches: int,
) -> None:
    num_stages = 2 * pp_size
    f_seen: dict[tuple[int, int], int] = {}
    b_seen: dict[tuple[int, int], int] = {}
    bc_seen: dict[tuple[int, int], int] = {}

    for rank, row in table.items():
        for time, action in enumerate(row):
            if action is None:
                continue
            if rank_for_stage(action.stage, pp_size) != rank:
                raise AssertionError(
                    f"{action.label} placed on rank {rank}, expected "
                    f"{rank_for_stage(action.stage, pp_size)}"
                )
            key = (action.stage, action.microbatch)
            if action.kind == Kind.F:
                if key in f_seen:
                    raise AssertionError(f"duplicate forward {key}")
                f_seen[key] = time
            elif action.kind == Kind.B:
                if key in b_seen:
                    raise AssertionError(f"duplicate backward {key}")
                b_seen[key] = time
            else:
                if key in bc_seen:
                    raise AssertionError(f"duplicate backward continuation {key}")
                bc_seen[key] = time

    expected = {
        (stage, mb)
        for stage in range(num_stages)
        for mb in range(n_microbatches)
    }
    if set(f_seen) != expected:
        raise AssertionError("missing or extra forwards")
    if set(b_seen) != expected:
        raise AssertionError("missing or extra backwards")
    if set(bc_seen) != expected:
        raise AssertionError("missing or extra backward continuations")

    for stage, mb in expected:
        if bc_seen[(stage, mb)] != b_seen[(stage, mb)] + 1:
            raise AssertionError(f"{stage}B_{mb + 1} does not follow {stage}B{mb + 1}")
        if b_seen[(stage, mb)] <= f_seen[(stage, mb)]:
            raise AssertionError(f"{stage}B{mb + 1} starts before its forward completes")
        if stage > 0 and f_seen[(stage, mb)] <= f_seen[(stage - 1, mb)]:
            raise AssertionError(f"{stage}F{mb + 1} violates forward dependency")
        if stage < num_stages - 1 and b_seen[(stage, mb)] <= bc_seen[(stage + 1, mb)]:
            raise AssertionError(f"{stage}B{mb + 1} violates backward dependency")


def solve_with_cp_sat(
    pp_size: int,
    n_microbatches: int,
    *,
    memory_cap: int | None = None,
    time_limit_s: float = 30.0,
) -> dict[int, list[Action | None]]:
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:
        raise RuntimeError(
            "OR-Tools is required for exact solving. Install it with `pip install ortools`."
        ) from e

    num_stages = 2 * pp_size
    horizon = conservative_horizon(pp_size, n_microbatches)
    memory_cap = memory_cap if memory_cap is not None else 2 * (pp_size - 1)

    model = cp_model.CpModel()

    start_f: dict[tuple[int, int], Any] = {}
    start_b: dict[tuple[int, int], Any] = {}
    interval_f: dict[tuple[int, int], Any] = {}
    interval_b: dict[tuple[int, int], Any] = {}
    activation_intervals_by_rank: dict[int, list[Any]] = {
        rank: [] for rank in range(pp_size)
    }
    compute_intervals_by_rank: dict[int, list[Any]] = {
        rank: [] for rank in range(pp_size)
    }

    for stage in range(num_stages):
        rank = rank_for_stage(stage, pp_size)
        for mb in range(n_microbatches):
            key = (stage, mb)
            sf = model.NewIntVar(0, horizon, f"start_f_s{stage}_m{mb}")
            sb = model.NewIntVar(0, horizon, f"start_b_s{stage}_m{mb}")
            start_f[key] = sf
            start_b[key] = sb
            f_itv = model.NewIntervalVar(sf, 1, sf + 1, f"f_s{stage}_m{mb}")
            b_itv = model.NewIntervalVar(sb, 2, sb + 2, f"b_s{stage}_m{mb}")
            interval_f[key] = f_itv
            interval_b[key] = b_itv
            compute_intervals_by_rank[rank].extend([f_itv, b_itv])

            activation_start = model.NewIntVar(0, horizon, f"act_start_s{stage}_m{mb}")
            activation_end = model.NewIntVar(0, horizon, f"act_end_s{stage}_m{mb}")
            activation_size = model.NewIntVar(0, horizon, f"act_size_s{stage}_m{mb}")
            model.Add(activation_start == sf + 1)
            model.Add(activation_end == sb + 2)
            model.Add(activation_size == activation_end - activation_start)
            activation_intervals_by_rank[rank].append(
                model.NewIntervalVar(
                    activation_start,
                    activation_size,
                    activation_end,
                    f"act_s{stage}_m{mb}",
                )
            )

    for rank in range(pp_size):
        model.AddNoOverlap(compute_intervals_by_rank[rank])
        model.AddCumulative(
            activation_intervals_by_rank[rank],
            [1] * len(activation_intervals_by_rank[rank]),
            memory_cap,
        )

    for stage in range(num_stages):
        for mb in range(n_microbatches):
            model.Add(start_b[(stage, mb)] >= start_f[(stage, mb)] + 1)
            if stage > 0:
                model.Add(start_f[(stage, mb)] >= start_f[(stage - 1, mb)] + 1)
            if stage < num_stages - 1:
                model.Add(start_b[(stage, mb)] >= start_b[(stage + 1, mb)] + 2)

    makespan = model.NewIntVar(0, horizon, "makespan")
    for mb in range(n_microbatches):
        model.Add(makespan >= start_b[(0, mb)] + 2)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"solver failed with status {solver.StatusName(status)}")

    table: dict[int, dict[int, Action]] = {rank: {} for rank in range(pp_size)}
    last_time = 0
    for stage in range(num_stages):
        rank = rank_for_stage(stage, pp_size)
        for mb in range(n_microbatches):
            f_time = solver.Value(start_f[(stage, mb)])
            b_time = solver.Value(start_b[(stage, mb)])
            table[rank][f_time] = Action(stage, Kind.F, mb)
            table[rank][b_time] = Action(stage, Kind.B, mb)
            table[rank][b_time + 1] = Action(stage, Kind.B_CONT, mb)
            last_time = max(last_time, b_time + 1)

    result: dict[int, list[Action | None]] = {}
    for rank, row in table.items():
        result[rank] = [row.get(time) for time in range(last_time + 1)]

    validate_table(result, pp_size, n_microbatches)
    return result


def table_to_jsonable(table: dict[int, list[Action | None]]) -> dict[str, list[str]]:
    return {
        str(rank): [action.label if action is not None else ".." for action in row]
        for rank, row in table.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--microbatches", "-m", type=int, required=True)
    parser.add_argument("--memory-cap", type=int)
    parser.add_argument("--time-limit-s", type=float, default=30.0)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    table = solve_with_cp_sat(
        args.pp,
        args.microbatches,
        memory_cap=args.memory_cap,
        time_limit_s=args.time_limit_s,
    )
    jsonable = table_to_jsonable(table)

    text = json.dumps(jsonable, indent=2) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
