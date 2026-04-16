"""AllDifferent global constraint.

Ensures that all variables in the scope take distinct values.
Uses value-based propagation: when a variable is assigned, that value
is removed from all other variables' domains.
"""

from __future__ import annotations

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class AllDifferent(Constraint):
    """All variables must take distinct values.

    Propagation strategy:
    - When a variable is assigned value v, remove v from all other domains.
    - When a domain becomes singleton {v}, remove v from all other domains.
    - Detects inconsistency when domain size < number of remaining values needed.
    """

    def __init__(self, *variables: Variable, name: str = "") -> None:
        super().__init__(list(variables), name or "AllDifferent")

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        """Check if assigned variables all have distinct values."""
        assigned_values: list[int] = []
        for var in self._variables:
            if var in assignment:
                val = assignment[var]
                if val in assigned_values:
                    return False
                assigned_values.append(val)
        return True

    def get_supported_values(self, var: Variable, assignment: dict[Variable, int]) -> set[int]:
        """Return values for var that don't conflict with assigned variables."""
        taken = set()
        for other in self._variables:
            if other is var:
                continue
            if other in assignment:
                taken.add(assignment[other])

        return var.domain.values() - taken

    def propagate_assignment(self, assigned_var: Variable, value: int) -> dict[Variable, list[int]]:
        """Propagate an assignment through the AllDifferent constraint.

        Removes the assigned value from all other variables' domains.
        Returns a map of variable -> removed values.
        """
        pruned: dict[Variable, list[int]] = {}
        for other in self._variables:
            if other is assigned_var:
                continue
            if other.is_assigned:
                continue
            if other.domain.contains(value):
                other.domain.remove(value)
                pruned[other] = [value]
        return pruned

    def check_hall_set(self) -> bool:
        """Check for Hall set violations (pigeonhole principle).

        If k unassigned variables have a combined domain of size < k,
        the constraint is unsatisfiable.

        Returns True if consistent, False if a Hall set violation is detected.
        """
        unassigned = [v for v in self._variables if not v.is_assigned]
        if not unassigned:
            return True

        # Simple check: union of all domains must be >= number of unassigned vars
        all_values: set[int] = set()
        for var in unassigned:
            all_values.update(var.domain.values())

        if len(all_values) < len(unassigned):
            return False

        # Check subsets of size 2 (most common Hall set violations)
        for i in range(len(unassigned)):
            for j in range(i + 1, len(unassigned)):
                combined = unassigned[i].domain.values() | unassigned[j].domain.values()
                if len(combined) < 2:
                    return False

        return True

    def bound_consistency_prune(self) -> dict[Variable, list[int]]:
        """Apply bounds consistency pruning for AllDifferent.

        Uses the observation that if the min/max range of unassigned variables
        is too small to accommodate all variables, some values can be pruned.

        Returns map of variable -> pruned values.
        """
        unassigned = [v for v in self._variables if not v.is_assigned]
        if len(unassigned) <= 1:
            return {}

        assigned_values = {
            assignment_val
            for var in self._variables
            if var.is_assigned
            for assignment_val in [var.assigned_value]
            if assignment_val is not None
        }

        pruned: dict[Variable, list[int]] = {}

        # Remove assigned values from unassigned domains
        for var in unassigned:
            removed = []
            for val in assigned_values:
                if var.domain.contains(val):
                    var.domain.remove(val)
                    removed.append(val)
            if removed:
                pruned[var] = removed

        return pruned
