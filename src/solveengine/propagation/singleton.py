"""Singleton arc consistency (SAC).

SAC is stronger than arc consistency. A value v in domain(x) is SAC
if, after tentatively assigning x=v and enforcing arc consistency,
no domain becomes empty. Values that fail this test can be removed.

SAC is more expensive than AC but can prune more values, especially
for hard problems near the phase transition.
"""

from __future__ import annotations

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.propagation.ac3 import AC3Propagator, PropagationResult


class SACPropagator:
    """Singleton Arc Consistency propagator.

    For each unassigned variable x and each value v in domain(x):
    1. Tentatively assign x = v
    2. Run AC-3 propagation
    3. If any domain becomes empty, remove v from domain(x)
    4. Restore all domains

    This is expensive (O(n * d * AC3_cost)) but very effective for
    hard problems.
    """

    def __init__(self, variables: list[Variable], constraints: list[Constraint]) -> None:
        self._variables = variables
        self._constraints = constraints

    def propagate(self, assignment: dict[Variable, int]) -> PropagationResult:
        """Run SAC propagation.

        Returns PropagationResult with all values pruned by SAC.
        """
        pruned: dict[Variable, list[int]] = {}
        total_revisions = 0
        changed = True

        while changed:
            changed = False
            for var in self._variables:
                if var.is_assigned:
                    continue
                if var.domain.is_empty:
                    return PropagationResult(False, pruned, total_revisions)

                values_to_remove = []
                for val in list(var.domain):
                    if not self._is_sac_consistent(var, val, assignment):
                        values_to_remove.append(val)

                for val in values_to_remove:
                    if var.domain.remove(val):
                        if var not in pruned:
                            pruned[var] = []
                        pruned[var].append(val)
                        changed = True
                        total_revisions += 1

                if var.domain.is_empty:
                    return PropagationResult(False, pruned, total_revisions)

        return PropagationResult(True, pruned, total_revisions)

    def _is_sac_consistent(
        self,
        var: Variable,
        value: int,
        assignment: dict[Variable, int],
    ) -> bool:
        """Check if assigning var=value is SAC-consistent.

        Tentatively assigns, runs AC-3, checks for wipeout, then restores.
        """
        # Save all domain generations
        generations = {v: v.domain.generation for v in self._variables}
        for v in self._variables:
            v.mark_generation()

        # Tentatively assign
        test_assignment = dict(assignment)
        test_assignment[var] = value
        var.domain.assign(value)

        # Run AC-3
        propagator = AC3Propagator(self._variables, self._constraints)
        result = propagator.propagate(test_assignment, trigger_var=var)

        # Restore all domains
        for v in self._variables:
            v.restore_to(generations[v])

        return result.consistent

    def propagate_incremental(
        self,
        var: Variable,
        assignment: dict[Variable, int],
    ) -> PropagationResult:
        """Run SAC only for variables connected to var.

        More efficient than full SAC when only one variable changed.
        """
        pruned: dict[Variable, list[int]] = {}
        revisions = 0

        # Get neighbors of var in constraint graph
        neighbors: set[Variable] = set()
        for cstr in self._constraints:
            if cstr.involves(var):
                for other in cstr.other_variables(var):
                    if not other.is_assigned:
                        neighbors.add(other)

        for neighbor in neighbors:
            if neighbor.domain.is_empty:
                return PropagationResult(False, pruned, revisions)

            values_to_remove = []
            for val in list(neighbor.domain):
                if not self._is_sac_consistent(neighbor, val, assignment):
                    values_to_remove.append(val)

            for val in values_to_remove:
                if neighbor.domain.remove(val):
                    if neighbor not in pruned:
                        pruned[neighbor] = []
                    pruned[neighbor].append(val)
                    revisions += 1

            if neighbor.domain.is_empty:
                return PropagationResult(False, pruned, revisions)

        return PropagationResult(True, pruned, revisions)
