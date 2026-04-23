"""Limited Discrepancy Search (LDS).

LDS explores the search tree by limiting the number of times the
heuristic's preferred value is not chosen (discrepancies). It first
tries the path with 0 discrepancies (all heuristic choices), then
allows 1 discrepancy, then 2, etc.

This is effective when the heuristic is good but not perfect.
"""

from __future__ import annotations

from typing import Iterator

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.heuristics.variable_ordering import VariableSelector, MRVSelector
from solveengine.heuristics.value_ordering import ValueOrderer, AscendingOrderer
from solveengine.search.dfs import SearchStrategy


class LDSStrategy(SearchStrategy):
    """Limited Discrepancy Search.

    Iteratively increases the allowed number of discrepancies from
    the heuristic ordering. A discrepancy occurs when a non-preferred
    value is chosen.
    """

    def __init__(
        self,
        var_selector: VariableSelector | None = None,
        val_orderer: ValueOrderer | None = None,
        max_discrepancy: int | None = None,
    ) -> None:
        self._var_selector = var_selector or MRVSelector()
        self._val_orderer = val_orderer or AscendingOrderer()
        self._max_discrepancy = max_discrepancy
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
        max_disc = self._max_discrepancy or len(variables)

        for allowed_disc in range(max_disc + 1):
            yield from self._lds_search(
                variables, constraints, assignment, allowed_disc, 0
            )

    def _lds_search(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        assignment: dict[Variable, int],
        remaining_disc: int,
        depth: int,
    ) -> Iterator[dict[Variable, int]]:
        if len(assignment) == len(variables):
            yield dict(assignment)
            return

        unassigned = [v for v in variables if v not in assignment]
        var = self._var_selector.select(unassigned, constraints, assignment)
        values = self._val_orderer.order(var, constraints, assignment)

        for i, value in enumerate(values):
            discrepancy_cost = 0 if i == 0 else 1

            if discrepancy_cost > remaining_disc:
                continue

            self._nodes_explored += 1
            assignment[var] = value

            if self._is_consistent(var, constraints, assignment):
                yield from self._lds_search(
                    variables,
                    constraints,
                    assignment,
                    remaining_disc - discrepancy_cost,
                    depth + 1,
                )

            del assignment[var]

    def _is_consistent(
        self,
        var: Variable,
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> bool:
        for cstr in constraints:
            if cstr.involves(var):
                if not cstr.is_satisfied(assignment):
                    return False
        return True
