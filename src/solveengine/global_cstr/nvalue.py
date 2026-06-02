"""NValue global constraint for CSP problems.

The NValue constraint restricts the number of distinct values used by a
set of variables. Given variables x1, ..., xn and a count variable (or
bounds), it ensures that the number of distinct values in the assignment
is within the specified range.

This is useful for workforce scheduling (limiting the number of different
shifts used), resource allocation, and diversity constraints.

Propagation uses lower and upper bound reasoning:
- Lower bound: at least L distinct values must be used (pigeonhole)
- Upper bound: at most U distinct values can be used (value merging)
"""

from __future__ import annotations

from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class NValueConstraint(Constraint):
    """Constrains the number of distinct values used by a set of variables.

    Ensures that: lower_bound <= |{x1, x2, ..., xn}| <= upper_bound
    where |{...}| denotes the number of distinct values in the set.

    Args:
        variables: The variables whose distinct value count is constrained.
        lower_bound: Minimum number of distinct values required.
        upper_bound: Maximum number of distinct values allowed.
        name: Optional constraint name.
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        lower_bound: int = 1,
        upper_bound: int | None = None,
        name: str = "",
    ) -> None:
        n = len(variables)
        if n < 1:
            raise ValueError("NValue constraint requires at least 1 variable")
        if lower_bound < 1:
            raise ValueError("Lower bound must be at least 1")
        if upper_bound is None:
            upper_bound = n
        if upper_bound < lower_bound:
            raise ValueError(
                f"Upper bound ({upper_bound}) must be >= lower bound ({lower_bound})"
            )
        super().__init__(list(variables), name or "NValue")
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._n = n

    @property
    def lower_bound(self) -> int:
        """Minimum number of distinct values required."""
        return self._lower_bound

    @property
    def upper_bound(self) -> int:
        """Maximum number of distinct values allowed."""
        return self._upper_bound

    @property
    def num_variables(self) -> int:
        """Number of variables in the constraint."""
        return self._n

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        """Check if the constraint is satisfied.

        For partial assignments, checks that the bounds can still be met
        given the assigned values and remaining domain possibilities.
        """
        assigned_values: set[int] = set()
        unassigned_count = 0

        for var in self._variables:
            if var in assignment:
                assigned_values.add(assignment[var])
            else:
                unassigned_count += 1

        current_distinct = len(assigned_values)

        if unassigned_count == 0:
            # Complete assignment: exact check
            return self._lower_bound <= current_distinct <= self._upper_bound

        # Partial assignment: check feasibility
        # Lower bound check: can we still reach the minimum?
        # Maximum possible distinct = current + number of new values unassigned can add
        all_possible_values = set(assigned_values)
        for var in self._variables:
            if var not in assignment:
                all_possible_values.update(var.domain.values())
        max_possible_distinct = len(all_possible_values)

        if max_possible_distinct < self._lower_bound:
            return False

        # Upper bound check: are we already over?
        if current_distinct > self._upper_bound:
            return False

        return True

    def get_supported_values(
        self, var: Variable, assignment: dict[Variable, int]
    ) -> set[int]:
        """Get values for var that don't violate the NValue bounds.

        A value is supported if assigning it keeps the distinct count
        feasible with respect to both bounds.
        """
        assigned_values: set[int] = set()
        unassigned_vars: list[Variable] = []

        for v in self._variables:
            if v is var:
                continue
            if v in assignment:
                assigned_values.add(assignment[v])
            else:
                unassigned_vars.append(v)

        supported: set[int] = set()
        remaining_after = len(unassigned_vars)  # excluding var itself

        for value in var.domain.values():
            test_values = set(assigned_values)
            test_values.add(value)
            current_distinct = len(test_values)

            # Check upper bound: if already at max, remaining must use existing values
            if current_distinct > self._upper_bound:
                continue

            # Check lower bound feasibility: can remaining vars add enough new values?
            possible_new_values: set[int] = set()
            for uvar in unassigned_vars:
                for uval in uvar.domain.values():
                    if uval not in test_values:
                        possible_new_values.add(uval)

            max_achievable = current_distinct + len(possible_new_values)
            if max_achievable < self._lower_bound:
                continue

            # Check upper bound feasibility: can remaining vars stay within limit?
            # Minimum distinct after remaining = current_distinct (if all use existing)
            # This is always feasible for upper bound if current <= upper
            supported.add(value)

        return supported

    def propagate(self, assignment: dict[Variable, int]) -> dict[Variable, list[int]]:
        """Propagate the NValue constraint using bound reasoning.

        Applies both lower-bound and upper-bound propagation rules.
        Returns a mapping from variable to list of pruned values.
        """
        pruned: dict[Variable, list[int]] = {}

        # Gather current state
        assigned_values: set[int] = set()
        unassigned_vars: list[Variable] = []

        for var in self._variables:
            if var in assignment:
                assigned_values.add(assignment[var])
            else:
                unassigned_vars.append(var)

        if not unassigned_vars:
            return pruned

        current_distinct = len(assigned_values)

        # Upper bound propagation: if at the limit, force remaining to use existing values
        upper_pruned = self._propagate_upper_bound(
            assigned_values, unassigned_vars, current_distinct
        )
        for var, removed in upper_pruned.items():
            pruned.setdefault(var, []).extend(removed)

        # Lower bound propagation: if we need more distinct values, force diversity
        lower_pruned = self._propagate_lower_bound(
            assigned_values, unassigned_vars, current_distinct
        )
        for var, removed in lower_pruned.items():
            pruned.setdefault(var, []).extend(removed)

        return pruned

    def _propagate_upper_bound(
        self,
        assigned_values: set[int],
        unassigned_vars: list[Variable],
        current_distinct: int,
    ) -> dict[Variable, list[int]]:
        """Upper bound propagation: limit new distinct values.

        If current_distinct == upper_bound, all unassigned variables
        must take values from the already-used set.
        """
        pruned: dict[Variable, list[int]] = {}

        if current_distinct >= self._upper_bound:
            # No new values allowed
            for var in unassigned_vars:
                removed: list[int] = []
                for value in list(var.domain.values()):
                    if value not in assigned_values:
                        if var.domain.remove(value):
                            removed.append(value)
                if removed:
                    pruned[var] = removed
        elif current_distinct == self._upper_bound - 1:
            # At most one new value can be introduced
            # Find values that appear in only one unassigned variable's domain
            # If introducing that value would exceed the limit, prune it
            new_value_vars: dict[int, list[Variable]] = {}
            for var in unassigned_vars:
                for value in var.domain.values():
                    if value not in assigned_values:
                        new_value_vars.setdefault(value, []).append(var)

            # Count how many new values each variable could introduce
            # If a variable can ONLY take new values, and there are too many
            # new values needed, we have a conflict (but don't prune here)
            pass

        return pruned

    def _propagate_lower_bound(
        self,
        assigned_values: set[int],
        unassigned_vars: list[Variable],
        current_distinct: int,
    ) -> dict[Variable, list[int]]:
        """Lower bound propagation: ensure enough distinct values.

        Uses Hall's theorem reasoning: if a subset of variables can only
        take values from a set smaller than the lower bound requirement,
        we need to ensure other variables provide the missing diversity.
        """
        pruned: dict[Variable, list[int]] = {}

        # Compute the maximum number of distinct values achievable
        all_possible: set[int] = set(assigned_values)
        for var in unassigned_vars:
            all_possible.update(var.domain.values())

        max_possible = len(all_possible)
        if max_possible < self._lower_bound:
            # Infeasible - but we don't prune here, just report
            return pruned

        # If a value appears in exactly one unassigned variable's domain,
        # and we need that value to meet the lower bound, it must be assigned
        needed_new = self._lower_bound - current_distinct
        if needed_new <= 0:
            return pruned

        # Find values that are only available in one variable
        value_to_vars: dict[int, list[Variable]] = {}
        for var in unassigned_vars:
            for value in var.domain.values():
                if value not in assigned_values:
                    value_to_vars.setdefault(value, []).append(var)

        # Unique values (only one variable can provide them)
        unique_providers: dict[Variable, set[int]] = {}
        for value, providers in value_to_vars.items():
            if len(providers) == 1:
                unique_providers.setdefault(providers[0], set()).add(value)

        # If a variable is the sole provider of a needed new value,
        # and we need more distinct values, that variable cannot take
        # an already-used value (it must contribute a new one)
        new_values_available = len(value_to_vars)
        if new_values_available <= needed_new:
            # Every new value is needed - variables that can provide new values
            # should not take already-used values if they're the only provider
            for var, unique_vals in unique_providers.items():
                if len(unique_vals) > 0 and new_values_available <= needed_new:
                    removed: list[int] = []
                    # If this var has both old and new values, and new values
                    # are critically needed, remove old values
                    has_new = any(
                        v not in assigned_values for v in var.domain.values()
                    )
                    if has_new and len(unique_vals) >= 1:
                        for value in list(var.domain.values()):
                            if value in assigned_values:
                                # Only prune if the variable MUST provide a unique value
                                other_new = [
                                    v for v in var.domain.values()
                                    if v not in assigned_values and v != value
                                ]
                                if len(other_new) == 0:
                                    continue
                                # Don't prune if it would empty the domain
                                if var.domain.size <= 1:
                                    continue
                                if var.domain.remove(value):
                                    removed.append(value)
                    if removed:
                        pruned[var] = removed

        return pruned

    def compute_distinct_bounds(
        self, assignment: dict[Variable, int]
    ) -> tuple[int, int]:
        """Compute achievable lower and upper bounds on distinct values.

        Returns (min_distinct, max_distinct) that can be achieved given
        the current assignment and remaining domains.
        """
        assigned_values: set[int] = set()
        unassigned_vars: list[Variable] = []

        for var in self._variables:
            if var in assignment:
                assigned_values.add(assignment[var])
            else:
                unassigned_vars.append(var)

        current_distinct = len(assigned_values)

        # Maximum: current + all new values available in unassigned domains
        all_new_values: set[int] = set()
        for var in unassigned_vars:
            for value in var.domain.values():
                if value not in assigned_values:
                    all_new_values.add(value)
        max_distinct = current_distinct + len(all_new_values)

        # Minimum: current distinct (if all unassigned can use existing values)
        # But if some variables can ONLY take new values, minimum increases
        min_distinct = current_distinct
        for var in unassigned_vars:
            all_old = all(
                value in assigned_values for value in var.domain.values()
            )
            if not all_old:
                # Check if variable is forced to take a new value
                has_old = any(
                    value in assigned_values for value in var.domain.values()
                )
                if not has_old:
                    # This variable MUST introduce a new value
                    min_distinct += 1

        return (min_distinct, max_distinct)

    def get_value_frequencies(
        self, assignment: dict[Variable, int]
    ) -> dict[int, int]:
        """Count how many variables are assigned each value.

        Useful for understanding the current value distribution.
        """
        frequencies: dict[int, int] = {}
        for var in self._variables:
            if var in assignment:
                val = assignment[var]
                frequencies[val] = frequencies.get(val, 0) + 1
        return frequencies

    def get_potential_values(self) -> set[int]:
        """Get the union of all values across all variable domains.

        This is the maximum set of distinct values that could appear.
        """
        all_values: set[int] = set()
        for var in self._variables:
            all_values.update(var.domain.values())
        return all_values
