"""Tests for Domain data structure — qualitative invariants."""

import pytest
from solveengine.core.domain import Domain


class TestDomainInvariants:
    """Qualitative tests: structural invariants that hold regardless of bug."""

    def test_domain_size_non_negative(self):
        """Domain size is always >= 0."""
        d = Domain(range(1, 10))
        for v in list(d):
            d.remove(v)
        assert d.size >= 0

    def test_domain_contains_consistency(self):
        """If contains(v) is True, v must be in iteration."""
        d = Domain([1, 2, 3, 4, 5])
        d.remove(3)
        for v in d:
            assert d.contains(v)

    def test_remove_decreases_size(self):
        """Removing a present value decreases size by exactly 1."""
        d = Domain(range(1, 6))
        initial = d.size
        d.remove(3)
        assert d.size == initial - 1

    def test_remove_absent_no_change(self):
        """Removing an absent value does not change size."""
        d = Domain([1, 2, 3])
        initial = d.size
        d.remove(99)
        assert d.size == initial

    def test_restore_undoes_removals(self):
        """Restoring to a generation undoes all later removals."""
        d = Domain(range(1, 6))
        gen = d.mark_generation()
        d.remove(2)
        d.remove(4)
        d.restore_to(gen)
        assert d.contains(2)
        assert d.contains(4)

    def test_singleton_after_assign(self):
        """After assign, domain contains exactly one value."""
        d = Domain(range(1, 10))
        d.assign(5)
        assert d.is_singleton
        assert d.contains(5)

    def test_copy_independence(self):
        """Copy is independent of original."""
        d = Domain([1, 2, 3])
        c = d.copy()
        d.remove(1)
        assert c.contains(1)

    def test_empty_domain_raises(self):
        """Cannot create an empty domain."""
        with pytest.raises(ValueError):
            Domain([])

    def test_min_max_consistency(self):
        """min_value <= max_value always."""
        d = Domain([3, 7, 1, 9, 2])
        assert d.min_value <= d.max_value

    def test_restrict_to_subset(self):
        """restrict_to keeps only allowed values."""
        d = Domain(range(1, 10))
        d.restrict_to({2, 4, 6, 8})
        for v in d:
            assert v in {2, 4, 6, 8}
