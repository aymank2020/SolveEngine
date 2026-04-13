"""Forward checking implementation.

Forward checking is a look-ahead technique that, after each assignment,
removes inconsistent values from the domains of future (unassigned)
variables. It detects failures earlier than simple backtracking by
checking constraints between the assigned variable and each unassigned
variable.

This module provides a standalone forward checker that can be used
independently of the full AC-3 propagator for lighter-weight propagation.
"""

from __future__ import annotations

from dataclasses import dataclass

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


@dataclass
class ForwardCheckResult:
    """Result of forward checking after an assignment."""

    consistent: bool
    pruned: dict[Variable, list[int]]
    wipeout_variable: Variable | None = None
    wipeout_constraint: Constraint | None = None

    @property
    def total_pruned(self) -> int:
        return sum(len(vals) for vals in self.pruned.values())


class ForwardChecker:
    """Forward checking propagator.

    After assigning a value to a variable, checks all constraints between
    that variable and unassigned variables. Removes values from unassigned
    variables that are inconsistent with the new assignment.

    Lighter than full arc consistency but catches many failures early.
    """

    def __init__(self, variables: list[Variable], constraints: list[Constraint]) -> None:
        self._variables = variables
        self._constraints = constraints
        self._constraint_map: dict[int, list[Constraint]] = {}
        self._build_map()

    def _build_map(self) -> None:
        """Build variable -> constraints index."""
        for var in self._variables:
            self._constraint_map[var.index] = []
        for cstr in self._constraints:
            for var in cstr.variables:
                if var.index in self._constraint_map:
                    self._constraint_map[var.index].append(cstr)

    def check(
        self,
        assigned_var: Variable,
        assigned_value: int,
        assignment: dict[Variable, int],
    ) -> ForwardCheckResult:
        """Perform forward checking after assigning assigned_var = assigned_value.

        For each constraint involving assigned_var, check all unassigned
        variables in that constraint and remove unsupported values.

        Args:
            assigned_var: The variable that was just assigned.
            assigned_value: The value assigned to it.
            assignment: The current complete assignment (including the new one).

        Returns:
            ForwardCheckResult with consistency status and pruned values.
        """
        pruned: dict[Variable, list[int]] = {}

        for cstr in self._constraint_map.get(assigned_var.index, []):
            for other_var in cstr.other_variables(assigned_var):
                if other_var.is_assigned:
                    continue

                # Find values of other_var that are consistent
                supported = cstr.get_supported_values(other_var, assignment)
                removed = []

                for val in list(other_var.domain):
                    if val not in supported:
                        if other_var.domain.remove(val):
                            removed.append(val)

                if removed:
                    if other_var not in pruned:
                        pruned[other_var] = []
                    pruned[other_var].extend(removed)

                # Check for domain wipeout
                if other_var.domain.is_empty:
                    return ForwardCheckResult(
                        consistent=False,
                        pruned=pruned,
                        wipeout_variable=other_var,
                        wipeout_constraint=cstr,
                    )

        return ForwardCheckResult(consistent=True, pruned=pruned)

    def check_all_constraints(
        self,
        assignment: dict[Variable, int],
    ) -> ForwardCheckResult:
        """Check all constraints against the current assignment.

        Used for initial propagation before search begins.
        """
        pruned: dict[Variable, list[int]] = {}

        for cstr in self._constraints:
            for var in cstr.variables:
                if var.is_assigned:
                    continue

                supported = cstr.get_supported_values(var, assignment)
                removed = []

                for val in list(var.domain):
                    if val not in supported:
                        if var.domain.remove(val):
                            removed.append(val)

                if removed:
                    if var not in pruned:
                        pruned[var] = []
                    pruned[var].extend(removed)

                if var.domain.is_empty:
                    return ForwardCheckResult(
                        consistent=False,
                        pruned=pruned,
                        wipeout_variable=var,
                        wipeout_constraint=cstr,
                    )

        return ForwardCheckResult(consistent=True, pruned=pruned)

    def get_future_constraints(self, var: Variable) -> list[Constraint]:
        """Get constraints between var and unassigned variables."""
        result = []
        for cstr in self._constraint_map.get(var.index, []):
            has_future = any(
                not v.is_assigned for v in cstr.other_variables(var)
            )
            if has_future:
                result.append(cstr)
        return result

    def estimate_pruning_power(self, var: Variable, value: int) -> int:
        """Estimate how many values would be pruned by assigning var=value.

        Used by value ordering heuristics (LCV) to prefer less constraining values.
        """
        total_pruned = 0
        test_assignment = {var: value}

        for cstr in self._constraint_map.get(var.index, []):
            for other_var in cstr.other_variables(var):
                if other_var.is_assigned:
                    continue
                supported = cstr.get_supported_values(other_var, test_assignment)
                current_size = other_var.domain_size
                total_pruned += current_size - len(supported & other_var.domain.values())

        return total_pruned
