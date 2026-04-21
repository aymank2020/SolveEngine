"""Tests for global constraints — behavioral invariants."""

import pytest
from solveengine.core.variable import Variable
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.global_cstr.linear import SumConstraint, ScalarProduct, ComparisonOp
from solveengine.global_cstr.element import ElementConstraint
from solveengine.global_cstr.cardinality import CardinalityConstraint


class TestAllDifferent:
    """Qualitative tests for AllDifferent constraint."""

    def test_distinct_values_satisfy(self):
        """Distinct values always satisfy AllDifferent."""
        vars = [Variable(f"v{i}", range(1, 10)) for i in range(4)]
        cstr = AllDifferent(*vars)
        assignment = {vars[0]: 1, vars[1]: 2, vars[2]: 3, vars[3]: 4}
        assert cstr.is_satisfied(assignment)

    def test_duplicate_values_violate(self):
        """Duplicate values always violate AllDifferent."""
        vars = [Variable(f"v{i}", range(1, 10)) for i in range(3)]
        cstr = AllDifferent(*vars)
        assignment = {vars[0]: 1, vars[1]: 1, vars[2]: 2}
        assert not cstr.is_satisfied(assignment)

    def test_partial_assignment_never_violates_if_distinct(self):
        """Partial assignment with distinct values is always satisfied."""
        vars = [Variable(f"v{i}", range(1, 10)) for i in range(5)]
        cstr = AllDifferent(*vars)
        assignment = {vars[0]: 1, vars[2]: 3}
        assert cstr.is_satisfied(assignment)

    def test_supported_excludes_taken_values(self):
        """Supported values exclude values already assigned to others."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(3)]
        cstr = AllDifferent(*vars)
        assignment = {vars[0]: 1, vars[1]: 3}
        supported = cstr.get_supported_values(vars[2], assignment)
        assert 1 not in supported
        assert 3 not in supported

    def test_hall_set_detects_pigeonhole(self):
        """Hall set check detects pigeonhole violations."""
        v1 = Variable("a", [1, 2])
        v2 = Variable("b", [1, 2])
        v3 = Variable("c", [1, 2])
        cstr = AllDifferent(v1, v2, v3)
        # 3 vars, domain size 2 → pigeonhole violation
        assert not cstr.check_hall_set()


class TestSumConstraint:
    """Qualitative tests for Sum constraint."""

    def test_correct_sum_satisfies(self):
        """Values summing to target satisfy the constraint."""
        vars = [Variable(f"v{i}", range(1, 10)) for i in range(3)]
        cstr = SumConstraint(vars, 10, ComparisonOp.EQ)
        assignment = {vars[0]: 3, vars[1]: 3, vars[2]: 4}
        assert cstr.is_satisfied(assignment)

    def test_wrong_sum_violates(self):
        """Values not summing to target violate the constraint."""
        vars = [Variable(f"v{i}", range(1, 10)) for i in range(3)]
        cstr = SumConstraint(vars, 10, ComparisonOp.EQ)
        assignment = {vars[0]: 1, vars[1]: 1, vars[2]: 1}
        assert not cstr.is_satisfied(assignment)

    def test_le_constraint(self):
        """Sum <= target works correctly."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(2)]
        cstr = SumConstraint(vars, 5, ComparisonOp.LE)
        assert cstr.is_satisfied({vars[0]: 2, vars[1]: 3})
        assert not cstr.is_satisfied({vars[0]: 4, vars[1]: 4})


class TestElementConstraint:
    """Qualitative tests for Element constraint."""

    def test_valid_index_and_value(self):
        """array[index] == value satisfies the constraint."""
        idx = Variable("i", range(0, 5))
        val = Variable("v", range(0, 100))
        array = [10, 20, 30, 40, 50]
        cstr = ElementConstraint(idx, array, val)
        assert cstr.is_satisfied({idx: 2, val: 30})

    def test_wrong_value_violates(self):
        """array[index] != value violates the constraint."""
        idx = Variable("i", range(0, 5))
        val = Variable("v", range(0, 100))
        array = [10, 20, 30, 40, 50]
        cstr = ElementConstraint(idx, array, val)
        assert not cstr.is_satisfied({idx: 2, val: 99})

    def test_out_of_bounds_index_violates(self):
        """Index outside array bounds violates."""
        idx = Variable("i", range(0, 10))
        val = Variable("v", range(0, 100))
        array = [10, 20, 30]
        cstr = ElementConstraint(idx, array, val)
        assert not cstr.is_satisfied({idx: 5, val: 10})


class TestCardinalityConstraint:
    """Qualitative tests for GCC."""

    def test_within_bounds_satisfies(self):
        """Assignment within cardinality bounds satisfies."""
        vars = [Variable(f"v{i}", [1, 2, 3]) for i in range(4)]
        # Each value appears 1-2 times
        cstr = CardinalityConstraint(vars, {1: (1, 2), 2: (1, 2), 3: (0, 2)})
        assignment = {vars[0]: 1, vars[1]: 2, vars[2]: 1, vars[3]: 3}
        assert cstr.is_satisfied(assignment)

    def test_exceeding_upper_bound_violates(self):
        """Exceeding upper bound violates."""
        vars = [Variable(f"v{i}", [1, 2]) for i in range(3)]
        cstr = CardinalityConstraint(vars, {1: (0, 1), 2: (0, 2)})
        assignment = {vars[0]: 1, vars[1]: 1, vars[2]: 2}
        assert not cstr.is_satisfied(assignment)
