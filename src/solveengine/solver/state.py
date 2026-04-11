"""Search state management for the backtracking solver.

Tracks the current assignment, decision stack, and solver statistics.
Provides checkpoint/restore for backtracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from solveengine.core.variable import Variable


@dataclass
class Decision:
    """A single decision in the search tree."""

    variable: Variable
    value: int
    generation: int  # Domain generation at time of decision
    depth: int
    conflict_set: set[Variable] = field(default_factory=set)


@dataclass
class SolverStats:
    """Statistics collected during solving."""

    nodes_explored: int = 0
    backtracks: int = 0
    propagations: int = 0
    solutions_found: int = 0
    max_depth: int = 0
    domain_wipeouts: int = 0
    constraint_checks: int = 0

    def reset(self) -> None:
        self.nodes_explored = 0
        self.backtracks = 0
        self.propagations = 0
        self.solutions_found = 0
        self.max_depth = 0
        self.domain_wipeouts = 0
        self.constraint_checks = 0

    def __repr__(self) -> str:
        return (
            f"SolverStats(nodes={self.nodes_explored}, backtracks={self.backtracks}, "
            f"solutions={self.solutions_found})"
        )


class SearchState:
    """Manages the search state during backtracking.

    Maintains the decision stack, current assignment, and provides
    methods for making/undoing decisions.
    """

    def __init__(self, variables: list[Variable]) -> None:
        self._variables = variables
        self._assignment: dict[Variable, int] = {}
        self._decision_stack: list[Decision] = []
        self._stats = SolverStats()

    @property
    def assignment(self) -> dict[Variable, int]:
        return self._assignment

    @property
    def depth(self) -> int:
        return len(self._decision_stack)

    @property
    def stats(self) -> SolverStats:
        return self._stats

    @property
    def decision_stack(self) -> list[Decision]:
        return self._decision_stack

    @property
    def is_complete(self) -> bool:
        """True if all variables are assigned."""
        return len(self._assignment) == len(self._variables)

    def unassigned_variables(self) -> list[Variable]:
        """Return list of variables not yet assigned."""
        return [v for v in self._variables if v not in self._assignment]

    def make_decision(self, var: Variable, value: int) -> Decision:
        """Assign a value to a variable and record the decision.

        Marks a new generation on the variable's domain for undo.
        """
        generation = var.mark_generation()
        var.assign(value)
        self._assignment[var] = value
        self._stats.nodes_explored += 1

        decision = Decision(
            variable=var,
            value=value,
            generation=generation,
            depth=self.depth,
        )
        self._decision_stack.append(decision)

        if self.depth > self._stats.max_depth:
            self._stats.max_depth = self.depth

        return decision

    def undo_decision(self) -> Decision | None:
        """Undo the most recent decision.

        Restores the variable's domain to the generation before the decision.
        Returns the undone decision, or None if the stack is empty.
        """
        if not self._decision_stack:
            return None

        decision = self._decision_stack.pop()
        var = decision.variable

        # Restore domain to pre-decision state
        var.restore_to(decision.generation)
        var.unassign()
        del self._assignment[var]

        self._stats.backtracks += 1
        return decision

    def undo_to_depth(self, target_depth: int) -> list[Decision]:
        """Undo decisions until reaching target_depth.

        Returns the list of undone decisions (most recent first).
        """
        undone = []
        while self.depth > target_depth:
            decision = self.undo_decision()
            if decision is not None:
                undone.append(decision)
        return undone

    def current_decision(self) -> Decision | None:
        """Return the most recent decision without removing it."""
        if self._decision_stack:
            return self._decision_stack[-1]
        return None

    def get_decision_at_depth(self, depth: int) -> Decision | None:
        """Return the decision at the given depth (0-indexed)."""
        if 0 <= depth < len(self._decision_stack):
            return self._decision_stack[depth]
        return None
