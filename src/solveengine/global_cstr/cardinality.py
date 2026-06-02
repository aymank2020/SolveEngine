"""Cardinality (Global Cardinality Constraint - GCC).

Ensures that each value in a specified set appears a certain number of
times across the variables. Useful for scheduling and assignment problems.
"""

from __future__ import annotations

from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class CardinalityConstraint(Constraint):
    """Global Cardinality Constraint.

    For each value v in the value set, the number of variables assigned to v
    must be between lower[v] and upper[v].

    Args:
        variables: The scope variables.
        value_counts: Dict mapping value -> (lower_bound, upper_bound).
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        value_counts: dict[int, tuple[int, int]],
        name: str = "",
    ) -> None:
        super().__init__(list(variables), name or "GCC")
        self._value_counts = dict(value_counts)

    @property
    def value_counts(self) -> dict[int, tuple[int, int]]:
        return self._value_counts

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        # Only fully check when all variables are assigned
        assigned_vars = [v for v in self._variables if v in assignment]
        unassigned_count = len(self._variables) - len(assigned_vars)

        if unassigned_count > 0:
            # Partial check: ensure no upper bound is already exceeded
            counts: dict[int, int] = {}
            for var in assigned_vars:
                val = assignment[var]
                counts[val] = counts.get(val, 0) + 1

            for val, count in counts.items():
                if val in self._value_counts:
                    _, upper = self._value_counts[val]
                    if count > upper:
                        return False
            return True

        # Full assignment: check all bounds
        counts: dict[int, int] = {}
        for var in self._variables:
            val = assignment[var]
            counts[val] = counts.get(val, 0) + 1

        for val, (lower, upper) in self._value_counts.items():
            actual = counts.get(val, 0)
            if actual < lower or actual > upper:
                return False

        return True

    def get_supported_values(self, var: Variable, assignment: dict[Variable, int]) -> set[int]:
        """Get values for var that don't violate upper bounds."""
        counts: dict[int, int] = {}
        for other in self._variables:
            if other is var:
                continue
            if other in assignment:
                val = assignment[other]
                counts[val] = counts.get(val, 0) + 1

        supported = set()
        for val in var.domain.values():
            current_count = counts.get(val, 0)
            if val in self._value_counts:
                _, upper = self._value_counts[val]
                if current_count < upper:
                    supported.add(val)
            else:
                # Value not in cardinality spec — always allowed
                supported.add(val)

        return supported

    def check_feasibility(self, assignment: dict[Variable, int]) -> bool:
        """Check if the constraint can still be satisfied.

        Verifies that lower bounds can still be met given remaining
        unassigned variables and their domains.
        """
        assigned_counts: dict[int, int] = {}
        for var in self._variables:
            if var in assignment:
                val = assignment[var]
                assigned_counts[val] = assigned_counts.get(val, 0) + 1

        unassigned = [v for v in self._variables if v not in assignment]

        for val, (lower, upper) in self._value_counts.items():
            current = assigned_counts.get(val, 0)
            if current > upper:
                return False
            # Check if lower bound can still be met
            needed = lower - current
            if needed > 0:
                can_provide = sum(
                    1 for v in unassigned if v.domain.contains(val)
                )
                if can_provide < needed:
                    return False

        return True
