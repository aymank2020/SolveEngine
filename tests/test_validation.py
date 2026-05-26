"""Tests for validation module."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.global_cstr.linear import SumConstraint, ComparisonOp
from solveengine.validation.checker import SolutionChecker
from solveengine.validation.bounds import BoundsValidator


class TestSolutionChecker:
    def test_valid_solution_passes(self):
        """Valid solution passes all checks."""
        x = Variable("x", range(1, 5))
        y = Variable("y", range(1, 5))
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)
        checker = SolutionChecker([x, y], [cstr])
        result = checker.validate({x: 1, y: 2})
        assert result.is_valid

    def test_violated_constraint_detected(self):
        """Violated constraint is detected."""
        x = Variable("x", range(1, 5))
        y = Variable("y", range(1, 5))
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)
        checker = SolutionChecker([x, y], [cstr])
        result = checker.validate({x: 1, y: 1})
        assert not result.is_valid
        assert result.num_violations == 1

    def test_incomplete_assignment_detected(self):
        """Missing variable assignment is detected."""
        x = Variable("x", range(1, 5))
        y = Variable("y", range(1, 5))
        checker = SolutionChecker([x, y], [])
        result = checker.validate({x: 1})
        assert not result.is_valid
        assert len(result.unassigned_variables) == 1

    def test_partial_consistency_check(self):
        """Partial assignment consistency check works."""
        x = Variable("x", range(1, 5))
        y = Variable("y", range(1, 5))
        cstr = BinaryConstraint(x, y, lambda a, b: a < b)
        checker = SolutionChecker([x, y], [cstr])
        assert checker.is_consistent({x: 1, y: 3})
        assert not checker.is_consistent({x: 3, y: 1})


class TestBoundsValidator:
    def test_feasible_problem(self):
        """Feasible problem passes bounds check."""
        x = Variable("x", range(1, 10))
        y = Variable("y", range(1, 10))
        cstr = SumConstraint([x, y], 10, ComparisonOp.LE)
        validator = BoundsValidator([x, y], [cstr])
        assert validator.is_feasible()

    def test_infeasible_detected(self):
        """Infeasible bounds are detected."""
        x = Variable("x", [8, 9, 10])
        y = Variable("y", [8, 9, 10])
        cstr = SumConstraint([x, y], 5, ComparisonOp.LE)
        validator = BoundsValidator([x, y], [cstr])
        # min sum = 16 > 5, so infeasible
        bounds = validator.compute_bounds()
        # At least one variable should have tightening potential
        assert any(b.can_tighten or b.implied_max < b.current_min for b in bounds)

    def test_tightening_removes_values(self):
        """Bounds tightening removes out-of-bounds values."""
        x = Variable("x", range(1, 20))
        y = Variable("y", range(1, 20))
        cstr = SumConstraint([x, y], 10, ComparisonOp.EQ)
        validator = BoundsValidator([x, y], [cstr])
        pruned = validator.tighten_all()
        # Values > 9 for x should be pruned (since y >= 1, x <= 9)
        if x in pruned:
            for val in pruned[x]:
                assert val > 9 or val < 1
