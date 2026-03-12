"""Tests for Constraint types — behavioral invariants."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint, UnaryConstraint, TableConstraint


class TestBinaryConstraint:
    """Qualitative tests for binary constraints."""

    def test_partial_assignment_always_satisfied(self):
        """A partial assignment (one var missing) never violates."""
        x = Variable("x", range(1, 5))
        y = Variable("y", range(1, 5))
        cstr = BinaryConstraint(x, y, lambda a, b: a < b)
        # Only x assigned
        assert cstr.is_satisfied({x: 4})
        # Only y assigned
        assert cstr.is_satisfied({y: 1})

    def test_supported_values_subset_of_domain(self):
        """Supported values are always a subset of the variable's domain."""
        x = Variable("x", range(1, 6))
        y = Variable("y", range(1, 6))
        cstr = BinaryConstraint(x, y, lambda a, b: a + b == 6)
        supported = cstr.get_supported_values(x, {y: 2})
        assert supported.issubset(x.domain.values())

    def test_arity_is_two(self):
        """Binary constraint always has arity 2."""
        x = Variable("x", [1, 2])
        y = Variable("y", [1, 2])
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)
        assert cstr.arity == 2

    def test_involves_both_variables(self):
        """Binary constraint involves both its variables."""
        x = Variable("x", [1, 2])
        y = Variable("y", [1, 2])
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)
        assert cstr.involves(x)
        assert cstr.involves(y)

    def test_not_equal_symmetry(self):
        """Not-equal is symmetric: if (a,b) satisfies, (b,a) satisfies."""
        x = Variable("x", range(1, 5))
        y = Variable("y", range(1, 5))
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)
        assert cstr.is_satisfied({x: 1, y: 2})
        # Swap values
        assert cstr.is_satisfied({x: 2, y: 1})


class TestUnaryConstraint:
    """Qualitative tests for unary constraints."""

    def test_supported_values_satisfy_predicate(self):
        """All supported values satisfy the predicate."""
        x = Variable("x", range(1, 10))
        cstr = UnaryConstraint(x, lambda v: v % 2 == 0)
        supported = cstr.get_supported_values(x, {})
        for val in supported:
            assert val % 2 == 0

    def test_arity_is_one(self):
        x = Variable("x", [1, 2, 3])
        cstr = UnaryConstraint(x, lambda v: v > 1)
        assert cstr.arity == 1


class TestTableConstraint:
    """Qualitative tests for table constraints."""

    def test_only_allowed_tuples_satisfy(self):
        """Only tuples in the allowed set satisfy the constraint."""
        x = Variable("x", [1, 2, 3])
        y = Variable("y", [1, 2, 3])
        allowed = {(1, 2), (2, 3), (3, 1)}
        cstr = TableConstraint([x, y], allowed)
        assert cstr.is_satisfied({x: 1, y: 2})
        assert not cstr.is_satisfied({x: 1, y: 1})

    def test_supported_values_non_empty_if_tuples_exist(self):
        """If allowed tuples exist for a value, supported is non-empty."""
        x = Variable("x", [1, 2, 3])
        y = Variable("y", [1, 2, 3])
        allowed = {(1, 2), (2, 3), (3, 1)}
        cstr = TableConstraint([x, y], allowed)
        supported = cstr.get_supported_values(x, {})
        assert len(supported) > 0
