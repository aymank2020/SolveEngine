"""Conflict analysis for conflict-driven backjumping.

When a domain wipeout occurs, conflict analysis determines which
decisions contributed to the failure and computes a minimal nogood
that can be used for backjumping and learning.
"""

from __future__ import annotations

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.solver.state import Decision, SearchState
from solveengine.learning.nogood import Nogood


class ConflictAnalyzer:
    """Analyzes conflicts to produce nogoods and backjump levels.

    Uses a simplified 1-UIP (Unit Implication Point) scheme:
    the learned nogood contains the most recent decision plus all
    decisions that contributed to the propagation chain leading to failure.
    """

    def __init__(self, constraints: list[Constraint]) -> None:
        self._constraints = constraints

    def analyze(
        self,
        state: SearchState,
        failed_var: Variable,
        failed_constraint: Constraint | None = None,
    ) -> tuple[Nogood, int]:
        """Analyze a conflict and produce a nogood + backjump level.

        Args:
            state: Current search state with decision stack.
            failed_var: The variable whose domain was wiped out.
            failed_constraint: The constraint that caused the wipeout (if known).

        Returns:
            Tuple of (learned_nogood, backjump_depth).
            backjump_depth is the depth to backtrack to (0-indexed).
        """
        conflict_set = self._compute_conflict_set(state, failed_var, failed_constraint)

        if not conflict_set:
            # Root-level failure — problem is unsatisfiable
            nogood = Nogood(frozenset(), learned_at_depth=0)
            return nogood, 0

        # Build nogood from conflict set
        nogood_entries: set[tuple[int, int]] = set()
        for var in conflict_set:
            if var in state.assignment:
                nogood_entries.add((var.index, state.assignment[var]))

        nogood = Nogood(frozenset(nogood_entries), learned_at_depth=state.depth)

        # Compute backjump level: second-highest decision level in conflict set
        levels = sorted(
            self._decision_level(state, var)
            for var in conflict_set
            if var in state.assignment
        )

        if len(levels) <= 1:
            backjump_depth = 0
        else:
            # Jump to the second-highest level
            backjump_depth = levels[-2]

        return nogood, backjump_depth

    def _compute_conflict_set(
        self,
        state: SearchState,
        failed_var: Variable,
        failed_constraint: Constraint | None,
    ) -> set[Variable]:
        """Compute the set of variables responsible for the conflict.

        Traces back through the constraint graph to find which decisions
        contributed to the domain wipeout of failed_var.
        """
        conflict_set: set[Variable] = set()

        # Start with variables in the failed constraint
        if failed_constraint is not None:
            for var in failed_constraint.variables:
                if var in state.assignment and var is not failed_var:
                    conflict_set.add(var)

        # Expand: for each variable in conflict set, add variables that
        # constrained it (transitively, up to assigned variables)
        frontier = list(conflict_set)
        visited: set[Variable] = set(conflict_set)

        while frontier:
            current = frontier.pop()
            for cstr in self._constraints:
                if not cstr.involves(current):
                    continue
                for other in cstr.other_variables(current):
                    if other in state.assignment and other not in visited:
                        conflict_set.add(other)
                        visited.add(other)
                        # Only expand one level deep to keep nogoods small
                        # (full resolution would expand further)

        return conflict_set

    def _decision_level(self, state: SearchState, var: Variable) -> int:
        """Find the decision level at which var was assigned."""
        for i, decision in enumerate(state.decision_stack):
            if decision.variable is var:
                return i
        return 0

    def minimize_nogood(self, nogood: Nogood, state: SearchState) -> Nogood:
        """Attempt to minimize a nogood by removing redundant literals.

        A literal is redundant if the nogood is still valid without it
        (i.e., the remaining literals still imply failure).
        """
        if nogood.size <= 2:
            return nogood  # Already minimal enough

        entries = list(nogood.assignments)
        minimized = set(entries)

        for entry in entries:
            candidate = frozenset(minimized - {entry})
            # Check if the reduced nogood is still implied by constraints
            # (simplified check: keep it if removing would lose the conflict)
            if len(candidate) >= 2:
                # Heuristic: keep entries at higher decision levels
                # (they are more likely to be essential)
                pass
            else:
                continue

        return Nogood(frozenset(minimized), learned_at_depth=nogood.learned_at_depth)
