"""Tests for Variable — structural invariants."""

import pytest
from solveengine.core.variable import Variable


class TestVariableInvariants:
    """Qualitative tests for Variable behavior."""

    def test_assigned_value_in_domain(self):
        """Assigned value must be in the domain."""
        v = Variable("x", range(1, 10))
        v.assign(5)
        assert v.domain.contains(5)

    def test_unassign_clears_assignment(self):
        """After unassign, is_assigned is False."""
        v = Variable("x", [1, 2, 3])
        v.assign(2)
        v.unassign()
        assert not v.is_assigned

    def test_double_assign_raises(self):
        """Cannot assign an already-assigned variable."""
        v = Variable("x", [1, 2, 3])
        v.assign(1)
        with pytest.raises(RuntimeError):
            v.assign(2)

    def test_assign_invalid_raises(self):
        """Cannot assign a value not in domain."""
        v = Variable("x", [1, 2, 3])
        with pytest.raises(ValueError):
            v.assign(99)

    def test_weight_non_negative(self):
        """Weight is always >= 1.0."""
        v = Variable("x", [1, 2, 3])
        assert v.weight >= 1.0
        v.increment_weight(5.0)
        assert v.weight >= 1.0

    def test_unique_indices(self):
        """Each variable gets a unique index."""
        v1 = Variable("a", [1, 2])
        v2 = Variable("b", [1, 2])
        v3 = Variable("c", [1, 2])
        assert len({v1.index, v2.index, v3.index}) == 3

    def test_restore_preserves_domain_values(self):
        """Restore brings back removed values."""
        v = Variable("x", range(1, 6))
        gen = v.mark_generation()
        v.assign(3)
        v.restore_to(gen)
        assert v.domain_size == 5

    def test_domain_size_matches_len(self):
        """domain_size equals len(domain)."""
        v = Variable("x", [1, 2, 3, 4, 5])
        assert v.domain_size == len(v.domain)
