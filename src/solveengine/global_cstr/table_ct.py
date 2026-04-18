"""Compact Table (CT) constraint propagation.

Implements the CT (Compact Table) algorithm for table constraints using
bitwise operations for efficient filtering. This is one of the fastest
known algorithms for propagating extensional (table) constraints.

The key idea: represent each tuple's validity as a bit in a bitmask.
For each variable-value pair, maintain a bitmask of tuples that support
that pair. Propagation intersects these bitmasks to find unsupported values.

Supports both positive tables (allowed tuples) and negative tables
(forbidden tuples).
"""

from __future__ import annotations

from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class BitSet:
    """A simple bitset implementation for tuple indexing.

    Uses Python integers as arbitrary-precision bit arrays.
    """

    __slots__ = ("_bits", "_size")

    def __init__(self, size: int, initial_value: int = 0) -> None:
        self._size = size
        self._bits = initial_value

    @classmethod
    def all_ones(cls, size: int) -> BitSet:
        """Create a bitset with all bits set to 1."""
        return cls(size, (1 << size) - 1)

    @classmethod
    def from_indices(cls, size: int, indices: Sequence[int]) -> BitSet:
        """Create a bitset with specific bit positions set."""
        bits = 0
        for idx in indices:
            if 0 <= idx < size:
                bits |= (1 << idx)
        return cls(size, bits)

    @property
    def size(self) -> int:
        return self._size

    @property
    def bits(self) -> int:
        return self._bits

    def is_empty(self) -> bool:
        """Check if no bits are set."""
        return self._bits == 0

    def count(self) -> int:
        """Count the number of set bits."""
        return bin(self._bits).count('1')

    def contains(self, index: int) -> bool:
        """Check if a specific bit is set."""
        if index < 0 or index >= self._size:
            return False
        return bool(self._bits & (1 << index))

    def set_bit(self, index: int) -> None:
        """Set a specific bit."""
        if 0 <= index < self._size:
            self._bits |= (1 << index)

    def clear_bit(self, index: int) -> None:
        """Clear a specific bit."""
        if 0 <= index < self._size:
            self._bits &= ~(1 << index)

    def intersect(self, other: BitSet) -> BitSet:
        """Return intersection (AND) of two bitsets."""
        return BitSet(self._size, self._bits & other._bits)

    def union(self, other: BitSet) -> BitSet:
        """Return union (OR) of two bitsets."""
        return BitSet(self._size, self._bits | other._bits)

    def complement(self) -> BitSet:
        """Return complement (NOT) of this bitset."""
        mask = (1 << self._size) - 1
        return BitSet(self._size, (~self._bits) & mask)

    def intersect_inplace(self, other: BitSet) -> None:
        """In-place intersection."""
        self._bits &= other._bits

    def union_inplace(self, other: BitSet) -> None:
        """In-place union."""
        self._bits |= other._bits

    def iter_set_bits(self) -> list[int]:
        """Return indices of all set bits."""
        indices = []
        bits = self._bits
        idx = 0
        while bits:
            if bits & 1:
                indices.append(idx)
            bits >>= 1
            idx += 1
        return indices

    def __repr__(self) -> str:
        return f"BitSet(size={self._size}, count={self.count()})"


class CompactTable:
    """Compact Table constraint using bitwise propagation.

    Maintains support bitmasks for each variable-value pair and uses
    bitwise AND operations to efficiently determine which values have
    lost all support.

    Args:
        variables: Variables in the constraint scope.
        tuples: Set of allowed (or forbidden) tuples.
        is_positive: If True, tuples are allowed; if False, tuples are forbidden.
        name: Optional constraint name.
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        tuples: Sequence[tuple[int, ...]],
        is_positive: bool = True,
        name: str = "",
    ) -> None:
        self._variables = list(variables)
        self._tuples = list(tuples)
        self._is_positive = is_positive
        self._name = name or f"CT({'positive' if is_positive else 'negative'})"
        self._num_tuples = len(tuples)
        self._supports: dict[int, dict[int, BitSet]] = {}
        self._current_valid: BitSet = BitSet.all_ones(self._num_tuples)
        self._build_supports()

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_tuples(self) -> int:
        return self._num_tuples

    @property
    def is_positive(self) -> bool:
        return self._is_positive

    @property
    def variables(self) -> list[Variable]:
        return self._variables

    @property
    def valid_tuple_count(self) -> int:
        """Number of currently valid tuples."""
        return self._current_valid.count()

    def _build_supports(self) -> None:
        """Build support bitmasks for each variable-value pair.

        For each variable index i and value v, supports[i][v] is a bitset
        where bit j is set if tuple j has value v at position i.
        """
        for var_idx in range(len(self._variables)):
            self._supports[var_idx] = {}

        for tuple_idx, tup in enumerate(self._tuples):
            for var_idx, value in enumerate(tup):
                if value not in self._supports[var_idx]:
                    self._supports[var_idx][value] = BitSet(self._num_tuples)
                self._supports[var_idx][value].set_bit(tuple_idx)

    def propagate(self, assignment: dict[Variable, int]) -> dict[Variable, list[int]]:
        """Run CT propagation and return pruned values per variable.

        Updates the valid tuple set based on current domains and assignment,
        then removes values that have no supporting valid tuple.
        """
        self._update_valid_tuples(assignment)

        if self._is_positive:
            return self._propagate_positive()
        else:
            return self._propagate_negative()

    def _update_valid_tuples(self, assignment: dict[Variable, int]) -> None:
        """Update the set of valid tuples based on current domains."""
        self._current_valid = BitSet.all_ones(self._num_tuples)

        for var_idx, var in enumerate(self._variables):
            if var in assignment:
                value = assignment[var]
                support = self._supports[var_idx].get(value)
                if support is not None:
                    self._current_valid.intersect_inplace(support)
                else:
                    self._current_valid = BitSet(self._num_tuples, 0)
                    return
            else:
                var_support = BitSet(self._num_tuples, 0)
                for value in var.domain:
                    support = self._supports[var_idx].get(value)
                    if support is not None:
                        var_support.union_inplace(support)
                self._current_valid.intersect_inplace(var_support)

    def _propagate_positive(self) -> dict[Variable, list[int]]:
        """Propagate for positive table: remove values with no valid support."""
        pruned: dict[Variable, list[int]] = {}

        for var_idx, var in enumerate(self._variables):
            if var.is_assigned:
                continue

            removed: list[int] = []
            for value in list(var.domain):
                support = self._supports[var_idx].get(value)
                if support is None:
                    if var.domain.remove(value):
                        removed.append(value)
                else:
                    intersection = support.intersect(self._current_valid)
                    if intersection.is_empty():
                        if var.domain.remove(value):
                            removed.append(value)

            if removed:
                pruned[var] = removed

        return pruned

    def _propagate_negative(self) -> dict[Variable, list[int]]:
        """Propagate for negative table: remove values supported only by forbidden tuples."""
        pruned: dict[Variable, list[int]] = {}

        forbidden_valid = self._current_valid

        for var_idx, var in enumerate(self._variables):
            if var.is_assigned:
                continue

            removed: list[int] = []
            for value in list(var.domain):
                support = self._supports[var_idx].get(value)
                if support is None:
                    continue

                forbidden_support = support.intersect(forbidden_valid)
                if not forbidden_support.is_empty():
                    all_tuples_for_value = support
                    non_forbidden = BitSet(
                        self._num_tuples,
                        all_tuples_for_value.bits & ~forbidden_valid.bits
                    )
                    if non_forbidden.is_empty():
                        if var.domain.remove(value):
                            removed.append(value)

            if removed:
                pruned[var] = removed

        return pruned

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        """Check if the assignment satisfies the table constraint."""
        values = []
        for var in self._variables:
            if var not in assignment:
                return True
            values.append(assignment[var])

        tup = tuple(values)
        if self._is_positive:
            return tup in set(self._tuples)
        else:
            return tup not in set(self._tuples)

    def get_supported_values(self, var: Variable, assignment: dict[Variable, int]) -> set[int]:
        """Get values for var that have support in the table."""
        var_idx = self._variables.index(var)
        self._update_valid_tuples(assignment)

        supported: set[int] = set()
        for value in var.domain:
            support = self._supports[var_idx].get(value)
            if support is None:
                if not self._is_positive:
                    supported.add(value)
                continue

            intersection = support.intersect(self._current_valid)
            if self._is_positive:
                if not intersection.is_empty():
                    supported.add(value)
            else:
                non_forbidden = BitSet(
                    self._num_tuples,
                    support.bits & ~self._current_valid.bits
                )
                if not non_forbidden.is_empty() or intersection.is_empty():
                    supported.add(value)

        return supported

    def reset(self) -> None:
        """Reset the valid tuple set to all tuples."""
        self._current_valid = BitSet.all_ones(self._num_tuples)

    def get_valid_tuples(self) -> list[tuple[int, ...]]:
        """Return the list of currently valid tuples."""
        valid_indices = self._current_valid.iter_set_bits()
        return [self._tuples[i] for i in valid_indices]

    def statistics(self) -> dict[str, int]:
        """Return statistics about the compact table."""
        return {
            "num_variables": len(self._variables),
            "num_tuples": self._num_tuples,
            "valid_tuples": self._current_valid.count(),
            "is_positive": int(self._is_positive),
        }
