"""Tests for variable and value ordering heuristics."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.heuristics.variable_ordering import (
    MRVSelector,
    DegreeSelector,
    DomWdegSelector,
)
from solveengine.heuristics.value_ordering import (
    AscendingOrderer,
    LCVOrderer,
    MiddleOutOrderer,
)


class TestVariableOrdering:
    """Qualitative tests for variable selection."""

    def test_mrv_selects_smallest_domain(self):
        """MRV always selects a variable with the smallest domain."""
        v1 = Variable("a", range(1, 10))  # size 9
        v2 = Variable("b", [1, 2])  # size 2
        v3 = Variable("c", range(1, 6))  # size 5

        selector = MRVSelector()
        selected = selector.select([v1, v2, v3], [], {})
        assert selected.domain_size == 2

    def test_selector_returns_unassigned(self):
        """Selector always returns a variable from the unassigned list."""
        variables = [Variable(f"v{i}", range(1, 5)) for i in range(5)]
        selector = MRVSelector()
        selected = selector.select(variables, [], {})
        assert selected in variables

    def test_degree_prefers_more_constrained(self):
        """Degree heuristic prefers variables in more constraints."""
        v1 = Variable("a", range(1, 5))
        v2 = Variable("b", range(1, 5))
        v3 = Variable("c", range(1, 5))
        # v2 is in 2 constraints, v1 and v3 in 1 each
        constraints = [
            BinaryConstraint(v1, v2, lambda a, b: a != b),
            BinaryConstraint(v2, v3, lambda a, b: a != b),
        ]

        selector = DegreeSelector()
        selected = selector.select([v1, v2, v3], constraints, {})
        assert selected is v2

    def test_empty_unassigned_raises(self):
        """Selecting from empty list raises ValueError."""
        selector = MRVSelector()
        with pytest.raises(ValueError):
            selector.select([], [], {})


class TestValueOrdering:
    """Qualitative tests for value ordering."""

    def test_ascending_is_sorted(self):
        """Ascending orderer returns values in sorted order."""
        v = Variable("x", [5, 3, 1, 4, 2])
        orderer = AscendingOrderer()
        values = orderer.order(v, [], {})
        assert values == sorted(values)

    def test_orderer_returns_all_domain_values(self):
        """Value orderer returns exactly the domain values."""
        v = Variable("x", [1, 3, 5, 7])
        orderer = AscendingOrderer()
        values = orderer.order(v, [], {})
        assert set(values) == v.domain.values()

    def test_middle_out_starts_from_center(self):
        """Middle-out orderer starts with the middle value."""
        v = Variable("x", [1, 2, 3, 4, 5])
        orderer = MiddleOutOrderer()
        values = orderer.order(v, [], {})
        assert values[0] == 3  # Middle of [1,2,3,4,5]

    def test_lcv_returns_all_values(self):
        """LCV orderer returns all domain values (just reordered)."""
        v1 = Variable("x", range(1, 5))
        v2 = Variable("y", range(1, 5))
        cstr = BinaryConstraint(v1, v2, lambda a, b: a != b)
        orderer = LCVOrderer()
        values = orderer.order(v1, [cstr], {})
        assert set(values) == v1.domain.values()
