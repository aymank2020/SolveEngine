"""Sparse set data structure for efficient domain representation.

A sparse set provides O(1) membership testing, insertion, and deletion,
plus O(size) iteration over present elements. Ideal for CSP domains
where values are removed and restored frequently.
"""

from __future__ import annotations

from typing import Iterator


class SparseSet:
    """Sparse set for integer domains with O(1) operations.

    Uses two arrays (sparse and dense) to achieve:
    - O(1) contains
    - O(1) remove
    - O(1) restore (by incrementing size)
    - O(n) iteration where n = current size

    The universe is [0, capacity).
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self._capacity = capacity
        self._dense = list(range(capacity))
        self._sparse = list(range(capacity))
        self._size = capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_empty(self) -> bool:
        return self._size == 0

    def contains(self, value: int) -> bool:
        """Check if value is in the set. O(1)."""
        if value < 0 or value >= self._capacity:
            return False
        pos = self._sparse[value]
        return pos < self._size

    def remove(self, value: int) -> bool:
        """Remove value from the set. O(1).

        Returns True if the value was present.
        """
        if not self.contains(value):
            return False

        # Swap with last element in dense array
        pos = self._sparse[value]
        last_pos = self._size - 1
        last_value = self._dense[last_pos]

        # Swap in dense
        self._dense[pos] = last_value
        self._dense[last_pos] = value

        # Update sparse
        self._sparse[last_value] = pos
        self._sparse[value] = last_pos

        self._size -= 1
        return True

    def restore_last(self) -> int | None:
        """Restore the most recently removed element. O(1).

        Returns the restored value, or None if set is full.
        """
        if self._size >= self._capacity:
            return None
        restored = self._dense[self._size]
        self._size += 1
        return restored

    def restore_to_size(self, target_size: int) -> list[int]:
        """Restore elements until reaching target_size.

        Returns list of restored values.
        """
        restored = []
        while self._size < target_size and self._size < self._capacity:
            val = self._dense[self._size]
            self._size += 1
            restored.append(val)
        return restored

    def min_value(self) -> int:
        """Find minimum value in the set. O(n)."""
        if self._size == 0:
            raise ValueError("Set is empty")
        return min(self._dense[:self._size])

    def max_value(self) -> int:
        """Find maximum value in the set. O(n)."""
        if self._size == 0:
            raise ValueError("Set is empty")
        return max(self._dense[:self._size])

    def values(self) -> list[int]:
        """Return all values currently in the set."""
        return list(self._dense[:self._size])

    def __contains__(self, value: int) -> bool:
        return self.contains(value)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[int]:
        return iter(self._dense[:self._size])

    def __repr__(self) -> str:
        vals = sorted(self._dense[:self._size])
        if len(vals) <= 10:
            return f"SparseSet({vals})"
        return f"SparseSet(size={self._size}, cap={self._capacity})"
