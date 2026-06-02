"""Cumulative global constraint for scheduling problems.

The Cumulative constraint ensures that at any point in time, the total
resource usage of overlapping tasks does not exceed a given capacity.
This is fundamental for resource-constrained scheduling.

Cumulative(starts, durations, demands, capacity):
  For all time points t:
    sum(demands[i] for i where starts[i] <= t < starts[i] + durations[i]) <= capacity
"""

from __future__ import annotations

from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class CumulativeConstraint(Constraint):
    """Cumulative resource constraint for scheduling.

    Args:
        start_vars: Variables representing task start times.
        durations: Fixed duration for each task.
        demands: Resource demand for each task.
        capacity: Maximum resource capacity at any time point.
    """

    def __init__(
        self,
        start_vars: Sequence[Variable],
        durations: Sequence[int],
        demands: Sequence[int],
        capacity: int,
        name: str = "",
    ) -> None:
        if len(start_vars) != len(durations) or len(start_vars) != len(demands):
            raise ValueError("start_vars, durations, and demands must have same length")
        super().__init__(list(start_vars), name or "Cumulative")
        self._durations = list(durations)
        self._demands = list(demands)
        self._capacity = capacity

    @property
    def durations(self) -> list[int]:
        return self._durations

    @property
    def demands(self) -> list[int]:
        return self._demands

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def num_tasks(self) -> int:
        return len(self._variables)

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        """Check if the cumulative constraint is satisfied.

        Only checks time points where assigned tasks are active.
        """
        # Collect assigned tasks
        tasks = []
        for i, var in enumerate(self._variables):
            if var in assignment:
                start = assignment[var]
                end = start + self._durations[i]
                tasks.append((start, end, self._demands[i]))

        if not tasks:
            return True

        # Check all relevant time points
        time_points = set()
        for start, end, _ in tasks:
            time_points.add(start)
            time_points.add(end)

        for t in time_points:
            usage = sum(
                demand for start, end, demand in tasks
                if start <= t < end
            )
            if usage > self._capacity:
                return False

        return True

    def get_supported_values(self, var: Variable, assignment: dict[Variable, int]) -> set[int]:
        """Get start times for var that don't exceed capacity."""
        var_idx = list(self._variables).index(var)
        duration = self._durations[var_idx]
        demand = self._demands[var_idx]

        # Collect other assigned tasks
        other_tasks = []
        for i, other_var in enumerate(self._variables):
            if other_var is var:
                continue
            if other_var in assignment:
                start = assignment[other_var]
                end = start + self._durations[i]
                other_tasks.append((start, end, self._demands[i]))

        supported = set()
        for start_val in var.domain.values():
            end_val = start_val + duration
            # Check all time points in [start_val, end_val)
            feasible = True
            for t in range(start_val, end_val):
                usage = demand  # This task's demand
                for other_start, other_end, other_demand in other_tasks:
                    if other_start <= t < other_end:
                        usage += other_demand
                if usage > self._capacity:
                    feasible = False
                    break
            if feasible:
                supported.add(start_val)

        return supported

    def earliest_start(self, task_idx: int, assignment: dict[Variable, int]) -> int:
        """Compute earliest feasible start time for a task.

        Considers capacity constraints from already-assigned tasks.
        """
        var = self._variables[task_idx]
        duration = self._durations[task_idx]
        demand = self._demands[task_idx]

        for start in sorted(var.domain):
            feasible = True
            for t in range(start, start + duration):
                usage = demand
                for i, other_var in enumerate(self._variables):
                    if i == task_idx:
                        continue
                    if other_var in assignment:
                        other_start = assignment[other_var]
                        other_end = other_start + self._durations[i]
                        if other_start <= t < other_end:
                            usage += self._demands[i]
                if usage > self._capacity:
                    feasible = False
                    break
            if feasible:
                return start

        return var.domain.min_value  # Fallback

    def latest_start(self, task_idx: int, assignment: dict[Variable, int]) -> int:
        """Compute latest feasible start time for a task."""
        var = self._variables[task_idx]
        duration = self._durations[task_idx]
        demand = self._demands[task_idx]

        for start in sorted(var.domain, reverse=True):
            feasible = True
            for t in range(start, start + duration):
                usage = demand
                for i, other_var in enumerate(self._variables):
                    if i == task_idx:
                        continue
                    if other_var in assignment:
                        other_start = assignment[other_var]
                        other_end = other_start + self._durations[i]
                        if other_start <= t < other_end:
                            usage += self._demands[i]
                if usage > self._capacity:
                    feasible = False
                    break
            if feasible:
                return start

        return var.domain.max_value  # Fallback

    def time_table_pruning(self, assignment: dict[Variable, int]) -> dict[Variable, list[int]]:
        """Apply time-table pruning.

        For each time point, compute the minimum resource usage from
        tasks that must be active at that time. If adding another task
        would exceed capacity, prune its start times.
        """
        pruned: dict[Variable, list[int]] = {}

        # Compute compulsory parts (tasks that must be active at certain times)
        compulsory_usage: dict[int, int] = {}
        for i, var in enumerate(self._variables):
            if var.is_assigned:
                start = var.assigned_value
                assert start is not None
                for t in range(start, start + self._durations[i]):
                    compulsory_usage[t] = compulsory_usage.get(t, 0) + self._demands[i]
            else:
                # Compulsory part: [latest_start, earliest_end)
                lst = var.domain.max_value
                ect = var.domain.min_value + self._durations[i]
                if lst < ect:
                    for t in range(lst, ect):
                        compulsory_usage[t] = compulsory_usage.get(t, 0) + self._demands[i]

        # Prune start times that would exceed capacity
        for i, var in enumerate(self._variables):
            if var.is_assigned:
                continue
            demand = self._demands[i]
            duration = self._durations[i]
            removed = []

            for start in list(var.domain):
                for t in range(start, start + duration):
                    other_usage = compulsory_usage.get(t, 0)
                    # Subtract this task's own compulsory contribution
                    lst = var.domain.max_value
                    ect = var.domain.min_value + duration
                    if lst < ect and lst <= t < ect:
                        other_usage -= demand
                    if other_usage + demand > self._capacity:
                        if var.domain.remove(start):
                            removed.append(start)
                        break

            if removed:
                pruned[var] = removed

        return pruned
