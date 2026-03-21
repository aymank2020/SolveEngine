"""AC-3 arc consistency algorithm.

Maintains a worklist of arcs (variable, constraint) pairs and iteratively
removes unsupported values until a fixed point is reached or a domain wipeout
is detected.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint

if TYPE_CHECKING:
    from solveengine.propagation.queue_policy import QueuePolicy


class PropagationResult:
    """Result of a propagation pass."""

    __slots__ = ("_consistent", "_pruned", "_revisions")

    def __init__(self, consistent: bool, pruned: dict[Variable, list[int]], revisions: int) -> None:
        self._consistent = consistent
        self._pruned = pruned
        self._revisions = revisions

    @property
    def consistent(self) -> bool:
        """Whether the propagation found the problem still consistent."""
        return self._consistent

    @property
    def pruned(self) -> dict[Variable, list[int]]:
        """Map of variable -> list of values removed during propagation."""
        return self._pruned

    @property
    def revisions(self) -> int:
        """Number of arc revisions performed."""
        return self._revisions

    def __repr__(self) -> str:
        status = "consistent" if self._consistent else "WIPEOUT"
        total_pruned = sum(len(v) for v in self._pruned.values())
        return f"PropagationResult({status}, pruned={total_pruned}, revisions={self._revisions})"


class AC3Propagator:
    """AC-3 arc consistency propagator.

    Enforces arc consistency by maintaining a queue of arcs to revise.
    When a domain is reduced, all arcs pointing to that variable are
    re-enqueued for revision.
    """

    def __init__(self, variables: list[Variable], constraints: list[Constraint]) -> None:
        self._variables = variables
        self._constraints = constraints
        self._constraint_map: dict[Variable, list[Constraint]] = {}
        self._build_constraint_map()

    def _build_constraint_map(self) -> None:
        """Build mapping from variable to constraints involving it."""
        for var in self._variables:
            self._constraint_map[var] = []
        for cstr in self._constraints:
            for var in cstr.variables:
                if var in self._constraint_map:
                    self._constraint_map[var].append(cstr)

    def propagate(
        self,
        assignment: dict[Variable, int],
        trigger_var: Variable | None = None,
    ) -> PropagationResult:
        """Run AC-3 propagation.

        Args:
            assignment: Current partial assignment.
            trigger_var: If provided, only enqueue arcs affected by this variable.
                        If None, enqueue all arcs (initial propagation).

        Returns:
            PropagationResult with consistency status and pruned values.
        """
        queue: deque[tuple[Variable, Constraint]] = deque()
        pruned: dict[Variable, list[int]] = {}
        revisions = 0

        if trigger_var is not None:
            # Only enqueue arcs affected by the trigger variable
            for cstr in self._constraint_map.get(trigger_var, []):
                for var in cstr.other_variables(trigger_var):
                    if not var.is_assigned:
                        queue.append((var, cstr))
        else:
            # Initial propagation: enqueue all arcs
            for cstr in self._constraints:
                for var in cstr.variables:
                    if not var.is_assigned:
                        queue.append((var, cstr))

        in_queue: set[tuple[int, int]] = {(id(v), id(c)) for v, c in queue}

        while queue:
            var, cstr = queue.popleft()
            in_queue.discard((id(var), id(cstr)))
            revisions += 1

            removed = self._revise(var, cstr, assignment)

            if removed:
                cstr.record_propagation()
                if var not in pruned:
                    pruned[var] = []
                pruned[var].extend(removed)

                if var.domain.is_empty:
                    # Domain wipeout — increment weights for dom/wdeg
                    for v in cstr.variables:
                        v.increment_weight()
                    return PropagationResult(False, pruned, revisions)

                # Re-enqueue arcs pointing to var (from other constraints)
                for other_cstr in self._constraint_map.get(var, []):
                    if other_cstr is cstr:
                        continue
                    for other_var in other_cstr.other_variables(var):
                        if not other_var.is_assigned:
                            key = (id(other_var), id(other_cstr))
                            if key not in in_queue:
                                queue.append((other_var, other_cstr))
                                in_queue.add(key)

        return PropagationResult(True, pruned, revisions)

    def _revise(
        self,
        var: Variable,
        cstr: Constraint,
        assignment: dict[Variable, int],
    ) -> list[int]:
        """Revise the domain of var with respect to constraint cstr.

        Removes values from var's domain that have no support.
        Returns list of removed values.
        """
        supported = cstr.get_supported_values(var, assignment)
        current = var.domain.values()
        to_remove = current - supported

        removed = []
        for val in to_remove:
            if var.domain.remove(val):
                removed.append(val)

        return removed

    def get_constraints_for(self, var: Variable) -> list[Constraint]:
        """Return all constraints involving the given variable."""
        return self._constraint_map.get(var, [])
