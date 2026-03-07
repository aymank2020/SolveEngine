"""Finite domain representation with efficient operations."""

from __future__ import annotations

from typing import Iterable, Iterator


class Domain:
    """A finite set of values that a variable can take.

    Supports efficient membership testing, removal, and restoration
    via a generation-based undo mechanism.
    """

    __slots__ = ("_values", "_removed", "_generation")

    def __init__(self, values: Iterable[int]) -> None:
        self._values: set[int] = set(values)
        if not self._values:
            raise ValueError("Domain cannot be empty at creation")
        self._removed: list[tuple[int, int]] = []  # (value, generation)
        self._generation: int = 0

    @property
    def size(self) -> int:
        """Number of values currently in the domain."""
        return len(self._values)

    @property
    def is_empty(self) -> bool:
        return len(self._values) == 0

    @property
    def is_singleton(self) -> bool:
        return len(self._values) == 1

    @property
    def min_value(self) -> int:
        """Smallest value in the domain."""
        if not self._values:
            raise ValueError("Domain is empty")
        return min(self._values)

    @property
    def max_value(self) -> int:
        """Largest value in the domain."""
        if not self._values:
            raise ValueError("Domain is empty")
        return max(self._values)

    @property
    def generation(self) -> int:
        """Current generation counter for undo tracking."""
        return self._generation

    def contains(self, value: int) -> bool:
        """Check if value is in the current domain."""
        return value in self._values

    def values(self) -> frozenset[int]:
        """Return current domain values as a frozen set."""
        return frozenset(self._values)

    def remove(self, value: int) -> bool:
        """Remove a value from the domain.

        Returns True if the value was present and removed, False otherwise.
        Records the removal for potential undo.
        """
        if value not in self._values:
            return False
        self._values.discard(value)
        self._removed.append((value, self._generation))
        return True

    def restrict_to(self, allowed: set[int]) -> list[int]:
        """Remove all values not in the allowed set.

        Returns the list of removed values.
        """
        to_remove = self._values - allowed
        removed_list = []
        for v in to_remove:
            self._values.discard(v)
            self._removed.append((v, self._generation))
            removed_list.append(v)
        return removed_list

    def assign(self, value: int) -> list[int]:
        """Assign a single value, removing all others.

        Returns the list of removed values.
        """
        if value not in self._values:
            raise ValueError(f"Cannot assign {value}: not in domain")
        removed_list = []
        for v in list(self._values):
            if v != value:
                self._values.discard(v)
                self._removed.append((v, self._generation))
                removed_list.append(v)
        return removed_list

    def mark_generation(self) -> int:
        """Advance generation counter and return the new generation."""
        self._generation += 1
        return self._generation

    def restore_to(self, target_generation: int) -> None:
        """Restore all values removed at or after target_generation."""
        while self._removed and self._removed[-1][1] >= target_generation:
            value, _ = self._removed.pop()
            self._values.add(value)
        if self._generation > target_generation:
            self._generation = target_generation

    def copy(self) -> Domain:
        """Create an independent copy of this domain."""
        new = Domain.__new__(Domain)
        new._values = set(self._values)
        new._removed = []
        new._generation = 0
        return new

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self._values))

    def __contains__(self, value: int) -> bool:
        return value in self._values

    def __repr__(self) -> str:
        vals = sorted(self._values)
        if len(vals) <= 8:
            return f"Domain({vals})"
        return f"Domain({vals[:4]}...{vals[-2:]}, size={len(vals)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Domain):
            return NotImplemented
        return self._values == other._values
