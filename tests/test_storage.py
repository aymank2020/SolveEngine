"""Tests for storage data structures."""

import pytest
from solveengine.storage.trail import Trail
from solveengine.storage.sparse_set import SparseSet


class TestTrail:
    """Qualitative tests for Trail undo mechanism."""

    def test_new_level_increments(self):
        """Each new_level increases the level counter."""
        trail = Trail()
        assert trail.current_level == 0
        trail.new_level()
        assert trail.current_level == 1
        trail.new_level()
        assert trail.current_level == 2

    def test_undo_restores_level(self):
        """Undo returns to previous level."""
        trail = Trail()
        trail.new_level()
        trail.record(0, 5)
        trail.new_level()
        trail.record(1, 3)
        trail.undo_level()
        assert trail.current_level == 1

    def test_undo_returns_entries(self):
        """Undo returns the entries that were at that level."""
        trail = Trail()
        trail.new_level()
        trail.record(0, 1)
        trail.record(0, 2)
        undone = trail.undo_level()
        assert len(undone) == 2

    def test_clear_resets_everything(self):
        """Clear resets trail to initial state."""
        trail = Trail()
        trail.new_level()
        trail.record(0, 1)
        trail.clear()
        assert trail.size == 0
        assert trail.current_level == 0


class TestSparseSet:
    """Qualitative tests for SparseSet."""

    def test_initial_contains_all(self):
        """New sparse set contains all values [0, capacity)."""
        ss = SparseSet(5)
        for i in range(5):
            assert ss.contains(i)

    def test_remove_makes_absent(self):
        """After remove, value is no longer contained."""
        ss = SparseSet(10)
        ss.remove(5)
        assert not ss.contains(5)

    def test_remove_decreases_size(self):
        """Remove decreases size by 1."""
        ss = SparseSet(10)
        initial = ss.size
        ss.remove(3)
        assert ss.size == initial - 1

    def test_restore_brings_back(self):
        """Restore brings back the most recently removed value."""
        ss = SparseSet(5)
        ss.remove(2)
        restored = ss.restore_last()
        assert restored == 2
        assert ss.contains(2)

    def test_out_of_range_not_contained(self):
        """Values outside [0, capacity) are never contained."""
        ss = SparseSet(5)
        assert not ss.contains(-1)
        assert not ss.contains(5)
        assert not ss.contains(100)

    def test_size_never_exceeds_capacity(self):
        """Size is always <= capacity."""
        ss = SparseSet(5)
        assert ss.size <= ss.capacity
        ss.remove(0)
        ss.restore_last()
        assert ss.size <= ss.capacity

    def test_empty_after_all_removed(self):
        """Set is empty after removing all values."""
        ss = SparseSet(3)
        ss.remove(0)
        ss.remove(1)
        ss.remove(2)
        assert ss.is_empty
