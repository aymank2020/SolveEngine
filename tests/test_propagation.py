"""Tests for propagation — qualitative invariants."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.propagation.ac3 import AC3Propagator
from solveengine.propagation.node_consistency import enforce_node_consistency
from solveengine.core.constraint import UnaryConstraint


class TestAC3Invariants:
    """Qualitative tests: propagation invariants that always hold."""

    def test_propagation_never_adds_values(self):
        """Propagation only removes values, never adds them."""
        x = Variable("x", range(1, 6))
        y = Variable("y", range(1, 6))
        cstr = BinaryConstraint(x, y, lambda a, b: a < b)
        initial_x = x.domain.values()
        initial_y = y.domain.values()

        prop = AC3Propagator([x, y], [cstr])
        prop.propagate({})

        assert x.domain.values().issubset(initial_x)
        assert y.domain.values().issubset(initial_y)

    def test_consistent_result_means_nonempty_domains(self):
        """If propagation returns consistent=True, no domain is empty."""
        x = Variable("x", range(1, 5))
        y = Variable("y", range(1, 5))
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)

        prop = AC3Propagator([x, y], [cstr])
        result = prop.propagate({})

        if result.consistent:
            assert x.domain_size > 0
            assert y.domain_size > 0

    def test_wipeout_means_inconsistent(self):
        """If a domain becomes empty, result is inconsistent."""
        x = Variable("x", [1])
        y = Variable("y", [1])
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)

        prop = AC3Propagator([x, y], [cstr])
        result = prop.propagate({})

        # x={1}, y={1}, x!=y → one must be wiped
        assert not result.consistent

    def test_pruned_values_not_in_domain(self):
        """Values reported as pruned are no longer in the domain."""
        x = Variable("x", range(1, 6))
        y = Variable("y", [3])  # y is fixed to 3
        cstr = BinaryConstraint(x, y, lambda a, b: a < b)

        prop = AC3Propagator([x, y], [cstr])
        result = prop.propagate({y: 3})

        for var, removed in result.pruned.items():
            for val in removed:
                assert not var.domain.contains(val)

    def test_revisions_non_negative(self):
        """Number of revisions is always >= 0."""
        x = Variable("x", range(1, 5))
        y = Variable("y", range(1, 5))
        cstr = BinaryConstraint(x, y, lambda a, b: a + b <= 5)

        prop = AC3Propagator([x, y], [cstr])
        result = prop.propagate({})
        assert result.revisions >= 0


class TestNodeConsistency:
    """Tests for node consistency enforcement."""

    def test_removes_violating_values(self):
        """Node consistency removes values that violate unary constraints."""
        x = Variable("x", range(1, 10))
        cstr = UnaryConstraint(x, lambda v: v % 3 == 0)

        pruned = enforce_node_consistency([x], [cstr])

        for val in x.domain:
            assert val % 3 == 0

    def test_preserves_satisfying_values(self):
        """Node consistency keeps values that satisfy the constraint."""
        x = Variable("x", range(1, 10))
        cstr = UnaryConstraint(x, lambda v: v <= 5)

        enforce_node_consistency([x], [cstr])

        assert x.domain.contains(1)
        assert x.domain.contains(5)
