"""Tests for nogood learning — structural invariants."""

import pytest
from solveengine.core.variable import Variable
from solveengine.learning.nogood import Nogood, NogoodStore


class TestNogood:
    """Qualitative tests for Nogood data structure."""

    def test_subsumption_subset(self):
        """A smaller nogood subsumes a larger one containing it."""
        small = Nogood(frozenset({(0, 1), (1, 2)}))
        large = Nogood(frozenset({(0, 1), (1, 2), (2, 3)}))
        assert small.subsumes(large)
        assert not large.subsumes(small)

    def test_self_subsumption(self):
        """A nogood subsumes itself."""
        ng = Nogood(frozenset({(0, 1), (1, 2)}))
        assert ng.subsumes(ng)

    def test_size_matches_assignments(self):
        """Size equals number of variable assignments."""
        ng = Nogood(frozenset({(0, 1), (1, 2), (2, 3)}))
        assert ng.size == 3

    def test_violation_detection(self):
        """Nogood correctly detects when assignment matches."""
        v1 = Variable("x", [1, 2, 3])
        v2 = Variable("y", [1, 2, 3])
        ng = Nogood.from_dict({v1: 1, v2: 2})
        assert ng.is_violated_by({v1: 1, v2: 2})
        assert not ng.is_violated_by({v1: 1, v2: 3})


class TestNogoodStore:
    """Qualitative tests for NogoodStore."""

    def test_add_increases_size(self):
        """Adding a non-subsumed nogood increases store size."""
        store = NogoodStore()
        ng = Nogood(frozenset({(0, 1), (1, 2)}))
        store.add(ng)
        assert store.size == 1

    def test_subsumed_not_added(self):
        """A nogood subsumed by existing one is not added."""
        store = NogoodStore()
        small = Nogood(frozenset({(0, 1)}))
        large = Nogood(frozenset({(0, 1), (1, 2)}))
        store.add(small)
        result = store.add(large)
        assert not result  # large is subsumed by small

    def test_capacity_respected(self):
        """Store never exceeds capacity."""
        store = NogoodStore(capacity=5)
        for i in range(10):
            store.add(Nogood(frozenset({(i, i)})))
        assert store.size <= 5

    def test_clear_empties_store(self):
        """Clear removes all nogoods."""
        store = NogoodStore()
        for i in range(5):
            store.add(Nogood(frozenset({(i, i)})))
        store.clear()
        assert store.size == 0

    def test_conflict_detection(self):
        """Store detects when assignment violates a stored nogood."""
        store = NogoodStore()
        v1 = Variable("x", [1, 2])
        v2 = Variable("y", [1, 2])
        ng = Nogood.from_dict({v1: 1, v2: 2})
        store.add(ng)

        conflict = store.check_conflict({v1: 1, v2: 2})
        assert conflict is not None

        no_conflict = store.check_conflict({v1: 1, v2: 1})
        assert no_conflict is None
