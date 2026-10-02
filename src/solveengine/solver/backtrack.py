"""Backtracking solver with forward checking and conflict-directed backjumping.

This is the main solver engine. It combines:
- Variable ordering heuristics (from heuristics module)
- Value ordering heuristics (from heuristics module)
- Arc consistency propagation (from propagation module)
- Conflict set tracking for intelligent backjumping
"""

from __future__ import annotations

from typing import Iterator

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.propagation.ac3 import AC3Propagator, PropagationResult
from solveengine.heuristics.variable_ordering import VariableSelector, MRVSelector
from solveengine.heuristics.value_ordering import ValueOrderer, AscendingOrderer
from solveengine.solver.state import SearchState, SolverStats


class BacktrackSolver:
    """CSP solver using backtracking with MAC (Maintaining Arc Consistency).

    Combines chronological backtracking with arc consistency propagation
    after each assignment. Supports conflict-directed backjumping when
    a conflict set is available.
    """

    def __init__(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        var_selector: VariableSelector | None = None,
        val_orderer: ValueOrderer | None = None,
        use_forward_check: bool = True,
        use_backjumping: bool = False,
    ) -> None:
        self._variables = variables
        self._constraints = constraints
        self._var_selector = var_selector or MRVSelector()
        self._val_orderer = val_orderer or AscendingOrderer()
        self._use_forward_check = use_forward_check
        self._use_backjumping = use_backjumping
        self._propagator = AC3Propagator(variables, constraints)
        self._state = SearchState(variables)
        self._node_limit: int | None = None
        self._node_limit_reached = False

    @property
    def stats(self) -> SolverStats:
        return self._state.stats

    @property
    def node_limit_reached(self) -> bool:
        """Whether search stopped before proving completeness."""
        return self._node_limit_reached

    def set_node_limit(self, limit: int) -> None:
        """Set maximum number of nodes to explore before giving up."""
        self._node_limit = limit

    def solve(self) -> dict[Variable, int] | None:
        """Find a single solution or return None if unsatisfiable.

        Returns a complete assignment if a solution exists.
        """
        self._node_limit_reached = False
        # Initial propagation
        result = self._propagator.propagate(self._state.assignment)
        if not result.consistent:
            return None

        solution = self._search()
        return solution

    def solve_all(self) -> list[dict[Variable, int]]:
        """Find all solutions."""
        self._node_limit_reached = False
        solutions = list(self._iter_solutions())
        return solutions

    def _iter_solutions(self) -> Iterator[dict[Variable, int]]:
        """Iterate over all solutions using backtracking."""
        result = self._propagator.propagate(self._state.assignment)
        if not result.consistent:
            return

        yield from self._search_all()

    def _search(self) -> dict[Variable, int] | None:
        """Recursive backtracking search for a single solution."""
        if self._state.is_complete:
            return dict(self._state.assignment)

        if self._node_limit and self._state.stats.nodes_explored >= self._node_limit:
            self._node_limit_reached = True
            return None

        unassigned = self._state.unassigned_variables()
        var = self._var_selector.select(unassigned, self._constraints, self._state.assignment)
        values = self._val_orderer.order(var, self._constraints, self._state.assignment)

        conflict_set: set[Variable] = set()

        for value in values:
            if self._node_limit_reached:
                return
            if not var.domain.contains(value):
                continue  # Value may have been pruned by earlier propagation

            # Save domain generations for all variables (for restoration)
            generations = {v: v.domain.generation for v in self._variables}

            # Mark new generation on ALL variables before decision
            for v in self._variables:
                v.mark_generation()

            decision = self._state.make_decision(var, value)

            consistent = True
            if self._use_forward_check:
                prop_result = self._propagator.propagate(self._state.assignment, trigger_var=var)
                self._state.stats.propagations += 1
                if not prop_result.consistent:
                    consistent = False
                    self._state.stats.domain_wipeouts += 1
                    # Collect conflict set from the wipeout
                    for wipeout_var in prop_result.pruned:
                        for cstr in self._propagator.get_constraints_for(wipeout_var):
                            for cv in cstr.variables:
                                if cv in self._state.assignment and cv is not var:
                                    conflict_set.add(cv)
            else:
                # Simple consistency check without propagation
                consistent = self._check_consistent(var, value)

            if consistent:
                result = self._search()
                if result is not None:
                    return result
                # Collect conflict set from deeper failure
                if self._use_backjumping and self._state.decision_stack:
                    deeper_decision = self._state.current_decision()
                    if deeper_decision and deeper_decision.conflict_set:
                        conflict_set.update(deeper_decision.conflict_set)

            # Undo the decision and restore ALL domains to pre-decision state
            self._state.undo_decision()
            for v in self._variables:
                v.restore_to(generations[v])

        # All values exhausted — record conflict set
        if self._use_backjumping and self._state.decision_stack:
            current = self._state.current_decision()
            if current:
                current.conflict_set = conflict_set

        return None

    def _search_all(self) -> Iterator[dict[Variable, int]]:
        """Search for all solutions."""
        if self._state.is_complete:
            yield dict(self._state.assignment)
            return

        if self._node_limit and self._state.stats.nodes_explored >= self._node_limit:
            self._node_limit_reached = True
            return

        unassigned = self._state.unassigned_variables()
        var = self._var_selector.select(unassigned, self._constraints, self._state.assignment)
        values = self._val_orderer.order(var, self._constraints, self._state.assignment)

        for value in values:
            if self._node_limit_reached:
                return
            if not var.domain.contains(value):
                continue
            generations = {v: v.domain.generation for v in self._variables}
            for v in self._variables:
                v.mark_generation()
            decision = self._state.make_decision(var, value)

            consistent = True
            if self._use_forward_check:
                prop_result = self._propagator.propagate(self._state.assignment, trigger_var=var)
                self._state.stats.propagations += 1
                if not prop_result.consistent:
                    consistent = False
                    self._state.stats.domain_wipeouts += 1
            else:
                consistent = self._check_consistent(var, value)

            if consistent:
                yield from self._search_all()

            self._state.undo_decision()
            for v in self._variables:
                v.restore_to(generations[v])

    def _check_consistent(self, var: Variable, value: int) -> bool:
        """Check if assigning value to var is consistent with current assignment."""
        for cstr in self._constraints:
            if not cstr.involves(var):
                continue
            if not cstr.is_satisfied(self._state.assignment):
                self._state.stats.constraint_checks += 1
                return False
            self._state.stats.constraint_checks += 1
        return True
