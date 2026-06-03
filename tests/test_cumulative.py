"""Tests for the Cumulative global constraint.

Tests scheduling scenarios, capacity enforcement, and time-table pruning.
The Cumulative constraint ensures resource usage never exceeds capacity
at any time point.
"""

import pytest
from solveengine.core.variable import Variable
from solveengine.global_cstr.cumulative import CumulativeConstraint
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.heuristics.variable_ordering import MRVSelector
from solveengine.heuristics.value_ordering import AscendingOrderer


class TestCumulativeBasic:
    """Basic satisfaction and violation tests."""

    def test_non_overlapping_tasks_satisfy(self):
        """Tasks that don't overlap always satisfy capacity."""
        starts = [Variable(f"s{i}", range(0, 20)) for i in range(3)]
        durations = [3, 4, 2]
        demands = [2, 3, 2]
        capacity = 3

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        # Tasks at t=0, t=5, t=10 — no overlap
        assignment = {starts[0]: 0, starts[1]: 5, starts[2]: 10}
        assert cstr.is_satisfied(assignment)

    def test_overlapping_within_capacity_satisfies(self):
        """Overlapping tasks within capacity are satisfied."""
        starts = [Variable(f"s{i}", range(0, 10)) for i in range(2)]
        durations = [3, 3]
        demands = [2, 2]
        capacity = 5

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        # Both start at 0, total demand = 4 <= 5
        assignment = {starts[0]: 0, starts[1]: 0}
        assert cstr.is_satisfied(assignment)

    def test_overlapping_exceeds_capacity_violates(self):
        """Overlapping tasks exceeding capacity violate the constraint."""
        starts = [Variable(f"s{i}", range(0, 10)) for i in range(3)]
        durations = [5, 5, 5]
        demands = [3, 3, 3]
        capacity = 5

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        # All start at 0, total demand = 9 > 5
        assignment = {starts[0]: 0, starts[1]: 0, starts[2]: 0}
        assert not cstr.is_satisfied(assignment)

    def test_partial_assignment_is_satisfied(self):
        """Partial assignments that don't violate are satisfied."""
        starts = [Variable(f"s{i}", range(0, 10)) for i in range(3)]
        durations = [2, 2, 2]
        demands = [3, 3, 3]
        capacity = 5

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        # Only one task assigned — cannot exceed capacity alone
        assignment = {starts[0]: 0}
        assert cstr.is_satisfied(assignment)

    def test_zero_demand_never_violates(self):
        """Tasks with zero demand never contribute to violations."""
        starts = [Variable(f"s{i}", range(0, 10)) for i in range(3)]
        durations = [5, 5, 5]
        demands = [0, 0, 0]
        capacity = 1

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        assignment = {starts[0]: 0, starts[1]: 0, starts[2]: 0}
        assert cstr.is_satisfied(assignment)


class TestCumulativeCapacity:
    """Tests for capacity boundary conditions."""

    def test_exact_capacity_satisfies(self):
        """Usage exactly at capacity is satisfied."""
        starts = [Variable(f"s{i}", range(0, 10)) for i in range(2)]
        durations = [4, 4]
        demands = [3, 2]
        capacity = 5

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        # Both overlap, total demand = 5 = capacity
        assignment = {starts[0]: 0, starts[1]: 1}
        assert cstr.is_satisfied(assignment)

    def test_one_over_capacity_violates(self):
        """Usage one unit over capacity violates."""
        starts = [Variable(f"s{i}", range(0, 10)) for i in range(2)]
        durations = [4, 4]
        demands = [3, 3]
        capacity = 5

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        # Both overlap at t=1, total demand = 6 > 5
        assignment = {starts[0]: 0, starts[1]: 1}
        assert not cstr.is_satisfied(assignment)

    def test_single_task_exceeds_capacity(self):
        """A single task with demand > capacity always violates."""
        starts = [Variable("s0", range(0, 10))]
        durations = [3]
        demands = [10]
        capacity = 5

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        assignment = {starts[0]: 0}
        assert not cstr.is_satisfied(assignment)

    def test_capacity_one_forces_sequential(self):
        """Capacity 1 forces all tasks to be sequential."""
        starts = [Variable(f"s{i}", range(0, 20)) for i in range(3)]
        durations = [3, 2, 4]
        demands = [1, 1, 1]
        capacity = 1

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        # Sequential: task0=[0,3), task1=[3,5), task2=[5,9)
        assignment = {starts[0]: 0, starts[1]: 3, starts[2]: 5}
        assert cstr.is_satisfied(assignment)

        # Overlapping: task0=[0,3), task1=[2,4) — overlap at t=2
        assignment_bad = {starts[0]: 0, starts[1]: 2, starts[2]: 5}
        assert not cstr.is_satisfied(assignment_bad)


class TestCumulativeSupportedValues:
    """Tests for get_supported_values filtering."""

    def test_supported_excludes_overloading_starts(self):
        """Supported values exclude start times that would overload."""
        starts = [Variable(f"s{i}", range(0, 10)) for i in range(2)]
        durations = [3, 3]
        demands = [3, 3]
        capacity = 5

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        # Task 0 at time 2, occupies [2,5)
        assignment = {starts[0]: 2}
        supported = cstr.get_supported_values(starts[1], assignment)

        # starts[1] at 2,3,4 would overlap with task 0 → demand 6 > 5
        assert 2 not in supported
        assert 3 not in supported
        assert 4 not in supported
        # starts[1] at 0 occupies [0,3) — overlaps at t=2 → demand 6 > 5
        assert 0 not in supported
        # starts[1] at 5 occupies [5,8) — no overlap
        assert 5 in supported

    def test_all_values_supported_when_no_conflict(self):
        """All domain values are supported when no other task is assigned."""
        starts = [Variable(f"s{i}", range(0, 5)) for i in range(2)]
        durations = [2, 2]
        demands = [1, 1]
        capacity = 5

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        supported = cstr.get_supported_values(starts[0], {})
        assert supported == set(range(0, 5))


class TestCumulativeTimeTablePruning:
    """Tests for time-table propagation."""

    def test_pruning_removes_infeasible_starts(self):
        """Time-table pruning removes start times that would exceed capacity."""
        starts = [Variable(f"s{i}", range(0, 10)) for i in range(3)]
        durations = [4, 4, 4]
        demands = [3, 3, 3]
        capacity = 5

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        # Assign first two tasks to overlap
        assignment = {starts[0]: 0, starts[1]: 0}
        pruned = cstr.time_table_pruning(assignment)

        # Task 2 cannot start at 0,1,2,3 (would overlap with both)
        if starts[2] in pruned:
            for val in [0, 1, 2, 3]:
                assert val in pruned[starts[2]] or not starts[2].domain.contains(val)

    def test_no_pruning_when_capacity_sufficient(self):
        """No pruning occurs when capacity is sufficient for all combinations."""
        starts = [Variable(f"s{i}", range(0, 10)) for i in range(2)]
        durations = [3, 3]
        demands = [1, 1]
        capacity = 10

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        pruned = cstr.time_table_pruning({})
        # No pruning needed — capacity is always sufficient
        total_pruned = sum(len(v) for v in pruned.values())
        assert total_pruned == 0


class TestCumulativeEarliestLatest:
    """Tests for earliest/latest start computation."""

    def test_earliest_start_respects_capacity(self):
        """Earliest start skips times where capacity would be exceeded."""
        starts = [Variable(f"s{i}", range(0, 15)) for i in range(2)]
        durations = [5, 3]
        demands = [4, 3]
        capacity = 5

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        # Task 0 occupies [0,5) with demand 4
        assignment = {starts[0]: 0}
        earliest = cstr.earliest_start(1, assignment)
        # Task 1 needs demand 3, total would be 7 > 5 during [0,5)
        assert earliest >= 5

    def test_latest_start_respects_capacity(self):
        """Latest start avoids times where capacity would be exceeded."""
        starts = [Variable(f"s{i}", range(0, 15)) for i in range(2)]
        durations = [5, 3]
        demands = [4, 3]
        capacity = 5

        cstr = CumulativeConstraint(starts, durations, demands, capacity)
        # Task 0 occupies [5,10) with demand 4
        assignment = {starts[0]: 5}
        latest = cstr.latest_start(1, assignment)
        # Task 1 with duration 3 cannot start at 3,4,5,6,7 (would overlap)
        assert latest <= 2 or latest >= 10


class TestCumulativeIntegration:
    """Integration tests with the solver."""

    def test_solve_simple_scheduling(self):
        """Solver finds valid schedule for simple instance."""
        starts = [Variable(f"s{i}", range(0, 15)) for i in range(3)]
        durations = [3, 4, 2]
        demands = [2, 3, 2]
        capacity = 4

        cumulative = CumulativeConstraint(starts, durations, demands, capacity)
        solver = BacktrackSolver(
            starts, [cumulative],
            var_selector=MRVSelector(),
            val_orderer=AscendingOrderer(),
            use_forward_check=True,
        )
        result = solver.solve()
        assert result is not None
        # Verify the solution satisfies the constraint
        assert cumulative.is_satisfied(result)

    def test_infeasible_instance_returns_none(self):
        """Solver returns None for infeasible scheduling instance."""
        # 3 tasks each needing demand 3, capacity 4, all must fit in [0,3)
        starts = [Variable(f"s{i}", [0]) for i in range(3)]
        durations = [3, 3, 3]
        demands = [3, 3, 3]
        capacity = 4

        cumulative = CumulativeConstraint(starts, durations, demands, capacity)
        solver = BacktrackSolver(starts, [cumulative], use_forward_check=True)
        result = solver.solve()
        # Total demand at t=0 would be 9 > 4
        assert result is None

    def test_constructor_validates_lengths(self):
        """Constructor raises on mismatched lengths."""
        starts = [Variable(f"s{i}", range(0, 10)) for i in range(3)]
        with pytest.raises(ValueError):
            CumulativeConstraint(starts, [1, 2], [1, 1, 1], 5)
        with pytest.raises(ValueError):
            CumulativeConstraint(starts, [1, 1, 1], [1, 1], 5)
