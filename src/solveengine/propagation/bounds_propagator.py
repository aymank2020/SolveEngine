"""Bounds propagation for integer CSP variables.

Maintains lower and upper bounds for each variable and propagates
bound changes through constraints. This is more efficient than full
arc consistency for problems with large domains (e.g., scheduling)
where maintaining explicit domain sets is impractical.

Bounds consistency ensures that for each variable, the minimum and
maximum values in its domain have support in all constraints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


@dataclass
class BoundsChange:
    """Records a change to a variable's bounds."""

    variable: Variable
    old_lower: int
    old_upper: int
    new_lower: int
    new_upper: int

    @property
    def lower_tightened(self) -> bool:
        return self.new_lower > self.old_lower

    @property
    def upper_tightened(self) -> bool:
        return self.new_upper < self.old_upper

    @property
    def is_wipeout(self) -> bool:
        return self.new_lower > self.new_upper


@dataclass
class BoundsPropagationResult:
    """Result of bounds propagation."""

    consistent: bool
    changes: list[BoundsChange] = field(default_factory=list)
    iterations: int = 0
    wipeout_variable: Variable | None = None

    @property
    def total_tightening(self) -> int:
        """Total number of bound tightenings performed."""
        count = 0
        for change in self.changes:
            if change.lower_tightened:
                count += 1
            if change.upper_tightened:
                count += 1
        return count


class VariableBounds:
    """Tracks lower and upper bounds for a variable with undo support."""

    __slots__ = ("_variable", "_lower", "_upper", "_history")

    def __init__(self, variable: Variable) -> None:
        self._variable = variable
        self._lower = variable.domain.min_value
        self._upper = variable.domain.max_value
        self._history: list[tuple[int, int]] = []

    @property
    def variable(self) -> Variable:
        return self._variable

    @property
    def lower(self) -> int:
        return self._lower

    @property
    def upper(self) -> int:
        return self._upper

    @property
    def span(self) -> int:
        """Size of the bounds interval."""
        return max(0, self._upper - self._lower + 1)

    @property
    def is_fixed(self) -> bool:
        """Whether the bounds define a single value."""
        return self._lower == self._upper

    @property
    def is_empty(self) -> bool:
        """Whether the bounds are inconsistent."""
        return self._lower > self._upper

    def tighten_lower(self, new_lower: int) -> bool:
        """Tighten the lower bound. Returns True if changed."""
        if new_lower <= self._lower:
            return False
        self._history.append((self._lower, self._upper))
        self._lower = new_lower
        return True

    def tighten_upper(self, new_upper: int) -> bool:
        """Tighten the upper bound. Returns True if changed."""
        if new_upper >= self._upper:
            return False
        self._history.append((self._lower, self._upper))
        self._upper = new_upper
        return True

    def save_state(self) -> int:
        """Save current state and return checkpoint index."""
        self._history.append((self._lower, self._upper))
        return len(self._history) - 1

    def restore_to(self, checkpoint: int) -> None:
        """Restore bounds to a previous checkpoint."""
        if checkpoint < len(self._history):
            self._lower, self._upper = self._history[checkpoint]
            self._history = self._history[:checkpoint]

    def reset(self) -> None:
        """Reset bounds to the variable's current domain extremes."""
        self._lower = self._variable.domain.min_value
        self._upper = self._variable.domain.max_value
        self._history.clear()


class BoundsConstraint:
    """A constraint that can propagate bounds.

    Wraps a standard Constraint and provides bounds propagation logic.
    Subclasses can override propagate_bounds for specialized propagation.
    """

    def __init__(self, constraint: Constraint) -> None:
        self._constraint = constraint

    @property
    def constraint(self) -> Constraint:
        return self._constraint

    @property
    def variables(self) -> tuple[Variable, ...]:
        return self._constraint.variables

    def propagate_bounds(
        self, bounds: dict[Variable, VariableBounds]
    ) -> list[BoundsChange]:
        """Propagate bounds through this constraint.

        Default implementation checks each variable's bounds against
        the constraint's supported values. Override for efficiency.
        """
        changes: list[BoundsChange] = []

        for var in self._constraint.variables:
            var_bounds = bounds.get(var)
            if var_bounds is None or var_bounds.is_fixed:
                continue

            partial_assignment: dict[Variable, int] = {}
            for other in self._constraint.variables:
                if other is var:
                    continue
                other_bounds = bounds.get(other)
                if other_bounds is not None and other_bounds.is_fixed:
                    partial_assignment[other] = other_bounds.lower

            supported = self._constraint.get_supported_values(var, partial_assignment)
            if not supported:
                changes.append(BoundsChange(
                    variable=var,
                    old_lower=var_bounds.lower,
                    old_upper=var_bounds.upper,
                    new_lower=var_bounds.upper + 1,
                    new_upper=var_bounds.lower - 1,
                ))
                continue

            bounded_supported = {
                v for v in supported
                if var_bounds.lower <= v <= var_bounds.upper
            }

            if not bounded_supported:
                changes.append(BoundsChange(
                    variable=var,
                    old_lower=var_bounds.lower,
                    old_upper=var_bounds.upper,
                    new_lower=var_bounds.upper + 1,
                    new_upper=var_bounds.lower - 1,
                ))
                continue

            new_lower = min(bounded_supported)
            new_upper = max(bounded_supported)

            if new_lower > var_bounds.lower or new_upper < var_bounds.upper:
                old_lower = var_bounds.lower
                old_upper = var_bounds.upper
                var_bounds.tighten_lower(new_lower)
                var_bounds.tighten_upper(new_upper)
                changes.append(BoundsChange(
                    variable=var,
                    old_lower=old_lower,
                    old_upper=old_upper,
                    new_lower=new_lower,
                    new_upper=new_upper,
                ))

        return changes


class BoundsPropagator:
    """Bounds consistency propagator.

    Maintains bounds for all variables and propagates changes through
    constraints until a fixed point is reached or inconsistency is detected.

    Args:
        variables: All CSP variables.
        constraints: All constraints to propagate through.
        max_iterations: Maximum propagation iterations before stopping.
    """

    def __init__(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        max_iterations: int = 1000,
    ) -> None:
        self._variables = variables
        self._constraints = constraints
        self._max_iterations = max_iterations
        self._bounds: dict[Variable, VariableBounds] = {}
        self._bounds_constraints: list[BoundsConstraint] = []
        self._var_to_constraints: dict[Variable, list[BoundsConstraint]] = {}
        self._initialize()

    def _initialize(self) -> None:
        """Initialize bounds and constraint mappings."""
        for var in self._variables:
            self._bounds[var] = VariableBounds(var)

        for cstr in self._constraints:
            bc = BoundsConstraint(cstr)
            self._bounds_constraints.append(bc)
            for var in cstr.variables:
                if var not in self._var_to_constraints:
                    self._var_to_constraints[var] = []
                self._var_to_constraints[var].append(bc)

    def get_bounds(self, var: Variable) -> tuple[int, int]:
        """Get current (lower, upper) bounds for a variable."""
        b = self._bounds.get(var)
        if b is None:
            return (var.domain.min_value, var.domain.max_value)
        return (b.lower, b.upper)

    def propagate(
        self, trigger_var: Variable | None = None
    ) -> BoundsPropagationResult:
        """Run bounds propagation to fixed point.

        Args:
            trigger_var: If provided, only propagate constraints involving
                        this variable initially. Otherwise propagate all.

        Returns:
            BoundsPropagationResult with consistency status and changes.
        """
        all_changes: list[BoundsChange] = []
        iterations = 0

        queue: deque[BoundsConstraint] = deque()
        if trigger_var is not None:
            for bc in self._var_to_constraints.get(trigger_var, []):
                queue.append(bc)
        else:
            for bc in self._bounds_constraints:
                queue.append(bc)

        in_queue: set[int] = {id(bc) for bc in queue}

        while queue and iterations < self._max_iterations:
            iterations += 1
            bc = queue.popleft()
            in_queue.discard(id(bc))

            changes = bc.propagate_bounds(self._bounds)

            for change in changes:
                if change.is_wipeout:
                    all_changes.append(change)
                    return BoundsPropagationResult(
                        consistent=False,
                        changes=all_changes,
                        iterations=iterations,
                        wipeout_variable=change.variable,
                    )

                if change.lower_tightened or change.upper_tightened:
                    all_changes.append(change)
                    for affected_bc in self._var_to_constraints.get(change.variable, []):
                        if affected_bc is not bc and id(affected_bc) not in in_queue:
                            queue.append(affected_bc)
                            in_queue.add(id(affected_bc))

        return BoundsPropagationResult(
            consistent=True,
            changes=all_changes,
            iterations=iterations,
        )

    def save_state(self) -> dict[Variable, int]:
        """Save current bounds state. Returns checkpoint indices per variable."""
        checkpoints: dict[Variable, int] = {}
        for var, bounds in self._bounds.items():
            checkpoints[var] = bounds.save_state()
        return checkpoints

    def restore_state(self, checkpoints: dict[Variable, int]) -> None:
        """Restore bounds to a saved state."""
        for var, checkpoint in checkpoints.items():
            bounds = self._bounds.get(var)
            if bounds is not None:
                bounds.restore_to(checkpoint)

    def reset(self) -> None:
        """Reset all bounds to initial domain values."""
        for bounds in self._bounds.values():
            bounds.reset()
