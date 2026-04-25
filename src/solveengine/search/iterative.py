"""Iterative deepening search strategy.

Combines the completeness of BFS with the memory efficiency of DFS
by repeatedly running depth-limited DFS with increasing depth limits.
"""

from __future__ import annotations

from typing import Iterator

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.heuristics.variable_ordering import VariableSelector, MRVSelector
from solveengine.heuristics.value_ordering import ValueOrderer, AscendingOrderer
from solveengine.search.dfs import SearchStrategy, DFSStrategy


class IterativeDeepeningStrategy(SearchStrategy):
    """Iterative deepening depth-first search.

    Runs DFS with depth limit 1, then 2, then 3, etc.
    Guarantees finding the shallowest solution first.
    """

    def __init__(
        self,
        var_selector: VariableSelector | None = None,
        val_orderer: ValueOrderer | None = None,
        max_depth: int | None = None,
    ) -> None:
        self._var_selector = var_selector or MRVSelector()
        self._val_orderer = val_orderer or AscendingOrderer()
        self._max_depth = max_depth
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
        max_d = self._max_depth or len(variables)

        for depth_limit in range(1, max_d + 1):
            dfs = DFSStrategy(
                var_selector=self._var_selector,
                val_orderer=self._val_orderer,
                depth_limit=depth_limit,
            )
            for solution in dfs.explore(variables, constraints, assignment):
                self._nodes_explored += dfs.nodes_explored
                yield solution
                return  # Found at this depth
            self._nodes_explored += dfs.nodes_explored
