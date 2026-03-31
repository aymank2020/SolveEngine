"""Variable ordering heuristics for CSP search.

The choice of which variable to assign next has a dramatic effect on
search efficiency. These heuristics implement the "fail-first" principle:
choose the variable most likely to cause a failure, so that failures are
detected early and the search tree is pruned.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class VariableSelector(ABC):
    """Abstract base for variable selection heuristics."""

    @abstractmethod
    def select(
        self,
        unassigned: list[Variable],
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> Variable:
        """Select the next variable to assign from the unassigned list."""
        ...


class MRVSelector(VariableSelector):
    """Minimum Remaining Values (MRV) heuristic.

    Selects the variable with the smallest current domain.
    Ties are broken by variable index (deterministic).
    """

    def select(
        self,
        unassigned: list[Variable],
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> Variable:
        if not unassigned:
            raise ValueError("No unassigned variables")
        return min(unassigned, key=lambda v: (v.domain_size, v.index))


class DegreeSelector(VariableSelector):
    """Degree heuristic.

    Selects the variable involved in the most constraints with other
    unassigned variables. Used as a tiebreaker for MRV.
    """

    def select(
        self,
        unassigned: list[Variable],
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> Variable:
        if not unassigned:
            raise ValueError("No unassigned variables")

        unassigned_set = set(unassigned)

        def degree(var: Variable) -> int:
            count = 0
            for cstr in constraints:
                if not cstr.involves(var):
                    continue
                for other in cstr.other_variables(var):
                    if other in unassigned_set:
                        count += 1
                        break
            return count

        return max(unassigned, key=lambda v: (degree(v), -v.index))


class DomWdegSelector(VariableSelector):
    """Domain-over-Weighted-Degree (dom/wdeg) heuristic.

    Selects the variable with the smallest ratio of domain size to
    weighted degree. The weight of a constraint is incremented each
    time it causes a domain wipeout during propagation.

    This is one of the most effective generic heuristics for CSP solving.
    """

    def select(
        self,
        unassigned: list[Variable],
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> Variable:
        if not unassigned:
            raise ValueError("No unassigned variables")

        unassigned_set = set(unassigned)

        def weighted_degree(var: Variable) -> float:
            total = 0.0
            for cstr in constraints:
                if not cstr.involves(var):
                    continue
                has_unassigned_other = any(
                    other in unassigned_set for other in cstr.other_variables(var)
                )
                if has_unassigned_other:
                    total += cstr.propagation_count + 1
            return max(total, 1.0)

        def score(var: Variable) -> float:
            return var.domain_size / weighted_degree(var)

        return min(unassigned, key=lambda v: (score(v), v.index))


class MRVDegreeSelector(VariableSelector):
    """MRV with degree as tiebreaker.

    First selects by minimum domain size, then breaks ties using
    the degree heuristic (most constraints with unassigned variables).
    """

    def __init__(self) -> None:
        self._mrv = MRVSelector()
        self._degree = DegreeSelector()

    def select(
        self,
        unassigned: list[Variable],
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> Variable:
        if not unassigned:
            raise ValueError("No unassigned variables")

        min_size = min(v.domain_size for v in unassigned)
        tied = [v for v in unassigned if v.domain_size == min_size]

        if len(tied) == 1:
            return tied[0]

        return self._degree.select(tied, constraints, assignment)
