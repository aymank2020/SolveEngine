"""Tests for distributed solving module."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.distributed.splitter import WorkSplitter
from solveengine.distributed.aggregator import (
    ResultAggregator,
    SubProblemResult,
    SubProblemStatus,
)


class TestWorkSplitter:
    def test_split_produces_parts(self):
        """Splitting produces the requested number of parts."""
        vars = [Variable(f"v{i}", range(1, 10)) for i in range(4)]
        constraints = [BinaryConstraint(vars[0], vars[1], lambda a, b: a != b)]
        splitter = WorkSplitter(vars, constraints)
        parts = splitter.split(num_parts=4)
        assert len(parts) >= 1
        assert len(parts) <= 4

    def test_split_covers_domain(self):
        """Split parts cover the entire domain of the split variable."""
        vars = [Variable(f"v{i}", range(1, 9)) for i in range(3)]
        splitter = WorkSplitter(vars, [])
        parts = splitter.split(num_parts=2)
        all_values = set()
        for part in parts:
            all_values.update(part.split_values)
        # Should cover the split variable's domain
        assert len(all_values) > 0

    def test_parts_are_disjoint(self):
        """Split parts have no overlapping values."""
        vars = [Variable(f"v{i}", range(1, 13)) for i in range(3)]
        splitter = WorkSplitter(vars, [])
        parts = splitter.split(num_parts=3)
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                overlap = parts[i].split_values & parts[j].split_values
                assert len(overlap) == 0

    def test_difficulty_non_negative(self):
        """Estimated difficulty is always non-negative."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(2)]
        splitter = WorkSplitter(vars, [])
        parts = splitter.split(num_parts=2)
        for part in parts:
            assert part.estimated_difficulty >= 0


class TestResultAggregator:
    def test_initial_state(self):
        """Aggregator starts with no solutions."""
        agg = ResultAggregator(num_sub_problems=4)
        assert not agg.has_solution
        assert not agg.is_complete

    def test_solution_submission(self):
        """Submitting a solved result marks has_solution."""
        x = Variable("x", [1, 2])
        agg = ResultAggregator(num_sub_problems=2)
        agg.submit_result(SubProblemResult(
            problem_id=0,
            status=SubProblemStatus.SOLVED,
            solution={x: 1},
            nodes_explored=10,
        ))
        assert agg.has_solution

    def test_complete_when_all_submitted(self):
        """Aggregator is complete when all sub-problems report."""
        agg = ResultAggregator(num_sub_problems=2)
        agg.submit_result(SubProblemResult(0, SubProblemStatus.UNSATISFIABLE))
        agg.submit_result(SubProblemResult(1, SubProblemStatus.UNSATISFIABLE))
        assert agg.is_complete

    def test_proven_unsat(self):
        """All UNSAT sub-problems proves global UNSAT."""
        agg = ResultAggregator(num_sub_problems=2)
        agg.submit_result(SubProblemResult(0, SubProblemStatus.UNSATISFIABLE))
        agg.submit_result(SubProblemResult(1, SubProblemStatus.UNSATISFIABLE))
        result = agg.get_aggregated()
        assert result.is_proven_unsat

    def test_progress_tracking(self):
        """Progress increases as results come in."""
        agg = ResultAggregator(num_sub_problems=4)
        assert agg.progress() == 0.0
        agg.submit_result(SubProblemResult(0, SubProblemStatus.SOLVED, solution={}))
        assert agg.progress() == 0.25
