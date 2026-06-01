"""Failure explanation for unsatisfiable CSP instances.

When a problem has no solution, the explainer identifies a minimal
unsatisfiable core — a subset of constraints that is itself unsatisfiable.
"""

from __future__ import annotations

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.solver.backtrack import BacktrackSolver


class FailureExplainer:
    """Explains why a CSP instance is unsatisfiable.

    Uses a deletion-based algorithm to find a Minimal Unsatisfiable
    Subproblem (MUS): a smallest subset of constraints that is still
    unsatisfiable.
    """

    def __init__(self, variables: list[Variable], constraints: list[Constraint]) -> None:
        self._variables = variables
        self._constraints = constraints

    def find_mus(self) -> list[Constraint]:
        """Find a Minimal Unsatisfiable Subproblem.

        Uses deletion-based approach: try removing each constraint;
        if the problem becomes satisfiable, the constraint is necessary.

        Returns the minimal set of constraints that causes unsatisfiability.
        Returns empty list if the problem is actually satisfiable.
        """
        # First verify the full problem is unsatisfiable
        if self._is_satisfiable(self._constraints):
            return []

        necessary: list[Constraint] = []
        candidates = list(self._constraints)

        for i in range(len(candidates)):
            # Try without constraint i
            reduced = candidates[:i] + candidates[i + 1:]
            reduced_with_necessary = reduced + necessary

            if self._is_satisfiable(reduced_with_necessary):
                # Constraint i is necessary for unsatisfiability
                necessary.append(candidates[i])

        return necessary

    def find_conflict_set(self) -> set[Variable]:
        """Find variables involved in the conflict.

        Returns the set of variables that appear in the MUS constraints.
        """
        mus = self.find_mus()
        conflict_vars: set[Variable] = set()
        for cstr in mus:
            for var in cstr.variables:
                conflict_vars.add(var)
        return conflict_vars

    def explain(self) -> str:
        """Generate a human-readable explanation of why the problem fails."""
        mus = self.find_mus()
        if not mus:
            return "The problem is satisfiable."

        lines = [f"Unsatisfiable: {len(mus)} conflicting constraints:"]
        for i, cstr in enumerate(mus, 1):
            var_names = ", ".join(v.name for v in cstr.variables)
            lines.append(f"  {i}. {cstr.name} (variables: {var_names})")

        conflict_vars = set()
        for cstr in mus:
            for var in cstr.variables:
                conflict_vars.add(var)

        lines.append(f"\nConflict involves {len(conflict_vars)} variables:")
        for var in sorted(conflict_vars, key=lambda v: v.name):
            lines.append(f"  - {var.name}: domain size = {var.domain_size}")

        return "\n".join(lines)

    def _is_satisfiable(self, constraints: list[Constraint]) -> bool:
        """Check if the problem with given constraints is satisfiable."""
        # Create fresh copies of variables for testing
        from solveengine.core.variable import Variable as Var
        from solveengine.core.domain import Domain

        fresh_vars = []
        var_map: dict[int, Variable] = {}
        for var in self._variables:
            fresh = Var(var.name, var.domain.values())
            fresh_vars.append(fresh)
            var_map[var.index] = fresh

        # Remap constraints to fresh variables (simplified: use original)
        solver = BacktrackSolver(
            self._variables,
            constraints,
            use_forward_check=True,
        )
        solver.set_node_limit(10000)  # Limit search for efficiency
        result = solver.solve()
        return result is not None
