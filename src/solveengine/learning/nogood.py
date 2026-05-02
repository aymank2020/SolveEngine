"""Nogood recording and storage.

A nogood is a partial assignment that is known to lead to failure.
Recording nogoods prevents the solver from exploring the same dead-end
subtrees multiple times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from solveengine.core.variable import Variable


@dataclass(frozen=True)
class Nogood:
    """An assignment combination known to be infeasible.

    A nogood (x1=v1, x2=v2, ...) means that this combination of
    assignments will always lead to failure.
    """

    assignments: frozenset[tuple[int, int]]  # (variable_index, value)
    learned_at_depth: int = 0

    @staticmethod
    def from_dict(assignment: dict[Variable, int], depth: int = 0) -> Nogood:
        """Create a nogood from a variable->value mapping."""
        entries = frozenset((var.index, val) for var, val in assignment.items())
        return Nogood(assignments=entries, learned_at_depth=depth)

    @property
    def size(self) -> int:
        """Number of variable assignments in this nogood."""
        return len(self.assignments)

    def subsumes(self, other: Nogood) -> bool:
        """Check if this nogood subsumes (is more general than) other.

        A nogood N1 subsumes N2 if N1 is a subset of N2.
        """
        return self.assignments.issubset(other.assignments)

    def is_violated_by(self, assignment: dict[Variable, int]) -> bool:
        """Check if the current assignment matches this nogood."""
        for var_idx, val in self.assignments:
            found = False
            for var, assigned_val in assignment.items():
                if var.index == var_idx and assigned_val == val:
                    found = True
                    break
            if not found:
                return False
        return True

    def get_asserting_variable(self, assignment: dict[Variable, int]) -> int | None:
        """Find the single unassigned variable in this nogood (unit propagation).

        If exactly one variable in the nogood is unassigned, that variable
        can be pruned (its value in the nogood is impossible).
        Returns the variable index, or None if not unit.
        """
        assigned_indices = {var.index for var in assignment}
        unassigned_in_nogood = []
        for var_idx, _ in self.assignments:
            if var_idx not in assigned_indices:
                unassigned_in_nogood.append(var_idx)
        if len(unassigned_in_nogood) == 1:
            return unassigned_in_nogood[0]
        return None


class NogoodStore:
    """Storage and management of learned nogoods.

    Provides efficient lookup and subsumption checking.
    Implements a bounded store with LRU eviction when capacity is reached.
    """

    def __init__(self, capacity: int = 10000) -> None:
        self._nogoods: list[Nogood] = []
        self._capacity = capacity
        self._total_added: int = 0
        self._total_used: int = 0

    @property
    def size(self) -> int:
        return len(self._nogoods)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_added(self) -> int:
        return self._total_added

    @property
    def total_used(self) -> int:
        return self._total_used

    def add(self, nogood: Nogood) -> bool:
        """Add a nogood to the store.

        Returns False if the nogood is subsumed by an existing one.
        Removes any existing nogoods subsumed by the new one.
        """
        # Check if subsumed by existing
        for existing in self._nogoods:
            if existing.subsumes(nogood):
                return False

        # Remove nogoods subsumed by the new one
        self._nogoods = [ng for ng in self._nogoods if not nogood.subsumes(ng)]

        # Evict oldest if at capacity
        if len(self._nogoods) >= self._capacity:
            self._nogoods.pop(0)

        self._nogoods.append(nogood)
        self._total_added += 1
        return True

    def check_conflict(self, assignment: dict[Variable, int]) -> Nogood | None:
        """Check if the current assignment violates any stored nogood.

        Returns the violated nogood, or None if no conflict.
        """
        for nogood in self._nogoods:
            if nogood.is_violated_by(assignment):
                self._total_used += 1
                return nogood
        return None

    def get_unit_nogoods(self, assignment: dict[Variable, int]) -> list[tuple[Nogood, int, int]]:
        """Find nogoods that become unit under the current assignment.

        Returns list of (nogood, variable_index, forbidden_value) tuples.
        """
        units = []
        assigned_indices = {var.index for var in assignment}

        for nogood in self._nogoods:
            unassigned_entries = []
            all_assigned_match = True

            for var_idx, val in nogood.assignments:
                if var_idx in assigned_indices:
                    # Check if the assigned value matches
                    matches = False
                    for var, assigned_val in assignment.items():
                        if var.index == var_idx:
                            matches = (assigned_val == val)
                            break
                    if not matches:
                        all_assigned_match = False
                        break
                else:
                    unassigned_entries.append((var_idx, val))

            if all_assigned_match and len(unassigned_entries) == 1:
                var_idx, val = unassigned_entries[0]
                units.append((nogood, var_idx, val))

        return units

    def clear(self) -> None:
        """Remove all stored nogoods."""
        self._nogoods.clear()

    def __iter__(self) -> Iterator[Nogood]:
        return iter(self._nogoods)

    def __len__(self) -> int:
        return len(self._nogoods)
