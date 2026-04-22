"""Depth-first search strategy.

Standard DFS explores the search tree by always expanding the deepest
unexplored node first. This is the default strategy for backtracking solvers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.heuristics.variable_ordering import VariableSelector, MRVSelector
from solveengine.heuristics.value_ordering import ValueOrderer, AscendingOrderer


class SearchStrategy(ABC):
    """Abstract base for search strategies."""

    @abstractmethod
    def explore(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> Iterator[dict[Variable, int]]:
        """Generate solutions by exploring the search space."""
        ...


class DFSStrategy(SearchStrategy):
    """Standard depth-first search.

    Explores the leftmost branch first, backtracks on failure.
    Memory efficient (O(depth)) but may explore large subtrees.
    """

    def __init__(
        self,
        var_selector: VariableSelector | None = None,
        val_orderer: ValueOrderer | None = None,
        depth_limit: int | None = None,
    ) -> None:
        self._var_selector = var_selector or MRVSelector()
        self._val_orderer = val_orderer or AscendingOrderer()
        self._depth_limit = depth_limit
        self._nodes_explored = 0

    @property
    def nodes_explored(self) -> int:
        return self._nodes_explored

    def explore(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> Iterator[dict[Variable, int]]:
        self._nodes_explored = 0
        yield from self._dfs(variables, constraints, assignment, depth=0)

    def _dfs(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        assignment: dict[Variable, int],
        depth: int,
    ) -> Iterator[dict[Variable, int]]:
        if len(assignment) == len(variables):
            yield dict(assignment)
            return

        if self._depth_limit is not None and depth >= self._depth_limit:
            return

        unassigned = [v for v in variables if v not in assignment]
        var = self._var_selector.select(unassigned, constraints, assignment)
        values = self._val_orderer.order(var, constraints, assignment)

        for value in values:
            self._nodes_explored += 1
            assignment[var] = value

            if self._is_consistent(var, constraints, assignment):
                yield from self._dfs(variables, constraints, assignment, depth + 1)

            del assignment[var]

    def _is_consistent(
        self,
        var: Variable,
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> bool:
        """Check if the current assignment is consistent with all constraints."""
        for cstr in constraints:
            if cstr.involves(var):
                if not cstr.is_satisfied(assignment):
                    return False
        return True
