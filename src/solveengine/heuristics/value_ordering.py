"""Value ordering heuristics for CSP search.

Once a variable is selected, the order in which its values are tried
can affect how quickly a solution is found. The "succeed-first" principle
suggests trying values most likely to lead to a solution.
"""

from __future__ import annotations

import random as _random
from abc import ABC, abstractmethod

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class ValueOrderer(ABC):
    """Abstract base for value ordering heuristics."""

    @abstractmethod
    def order(
        self,
        var: Variable,
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> list[int]:
        """Return the domain values of var in the order they should be tried."""
        ...


class AscendingOrderer(ValueOrderer):
    """Try values in ascending numerical order (deterministic baseline)."""

    def order(
        self,
        var: Variable,
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> list[int]:
        return sorted(var.domain)


class RandomOrderer(ValueOrderer):
    """Try values in random order (useful for randomized restarts)."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = _random.Random(seed)

    def order(
        self,
        var: Variable,
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> list[int]:
        values = list(var.domain)
        self._rng.shuffle(values)
        return values


class LCVOrderer(ValueOrderer):
    """Least Constraining Value (LCV) heuristic.

    Orders values by how many values they eliminate from neighboring
    variables' domains. Values that eliminate fewer options are tried first,
    maximizing the chance of finding a solution without backtracking.
    """

    def order(
        self,
        var: Variable,
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> list[int]:
        relevant_constraints = [c for c in constraints if c.involves(var)]
        neighbors = set()
        for cstr in relevant_constraints:
            for other in cstr.other_variables(var):
                if not other.is_assigned:
                    neighbors.add(other)

        def elimination_count(value: int) -> int:
            """Count how many values this assignment would eliminate from neighbors."""
            total = 0
            test_assignment = dict(assignment)
            test_assignment[var] = value

            for cstr in relevant_constraints:
                for neighbor in cstr.other_variables(var):
                    if neighbor.is_assigned or neighbor not in neighbors:
                        continue
                    supported = cstr.get_supported_values(neighbor, test_assignment)
                    current_size = neighbor.domain_size
                    total += current_size - len(supported & neighbor.domain.values())

            return total

        values = list(var.domain)
        values.sort(key=lambda v: (elimination_count(v), v))
        return values


class MiddleOutOrderer(ValueOrderer):
    """Try values from the middle of the domain outward.

    Useful for problems where solutions tend to be near the center
    of variable domains (e.g., scheduling, resource allocation).
    """

    def order(
        self,
        var: Variable,
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> list[int]:
        values = sorted(var.domain)
        if not values:
            return []
        mid = len(values) // 2
        result = [values[mid]]
        left, right = mid - 1, mid + 1
        while left >= 0 or right < len(values):
            if right < len(values):
                result.append(values[right])
                right += 1
            if left >= 0:
                result.append(values[left])
                left -= 1
        return result
