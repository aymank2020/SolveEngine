"""Work splitting for parallel/distributed CSP solving.

Splits a CSP problem into independent sub-problems that can be solved
in parallel. Uses domain splitting on a selected variable to create
disjoint sub-problems whose union covers the entire search space.
"""

from __future__ import annotations

from dataclasses import dataclass

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.decomposition.graph import ConstraintGraph


@dataclass
class SubProblem:
    """A sub-problem created by domain splitting."""

    problem_id: int
    split_variable: Variable
    split_values: frozenset[int]
    estimated_difficulty: float

    @property
    def domain_fraction(self) -> float:
        """Fraction of the original domain assigned to this sub-problem."""
        return len(self.split_values) / max(1, self.split_variable.domain_size)


class WorkSplitter:
    """Splits CSP problems into parallel sub-problems.

    Strategy: select the variable with the largest domain and split
    its domain into roughly equal parts. Each part becomes a sub-problem
    where that variable's domain is restricted.
    """

    def __init__(self, variables: list[Variable], constraints: list[Constraint]) -> None:
        self._variables = variables
        self._constraints = constraints
        self._graph = ConstraintGraph(variables, constraints)

    def split(self, num_parts: int = 4) -> list[SubProblem]:
        """Split the problem into num_parts sub-problems.

        Selects the best variable to split on and divides its domain.
        """
        split_var = self._select_split_variable()
        if split_var is None:
            return [SubProblem(
                problem_id=0,
                split_variable=self._variables[0],
                split_values=self._variables[0].domain.values(),
                estimated_difficulty=1.0,
            )]

        values = sorted(split_var.domain)
        parts = self._partition_values(values, num_parts)

        sub_problems = []
        for i, part in enumerate(parts):
            difficulty = self._estimate_difficulty(split_var, frozenset(part))
            sub_problems.append(SubProblem(
                problem_id=i,
                split_variable=split_var,
                split_values=frozenset(part),
                estimated_difficulty=difficulty,
            ))

        return sub_problems

    def split_balanced(self, num_parts: int = 4) -> list[SubProblem]:
        """Split trying to balance estimated difficulty across parts.

        Uses constraint tightness to estimate which values are harder.
        """
        split_var = self._select_split_variable()
        if split_var is None:
            return self.split(num_parts)

        values = sorted(split_var.domain)
        # Score each value by how many constraints it participates in
        value_scores: dict[int, float] = {}
        for val in values:
            score = 0.0
            for cstr in self._constraints:
                if cstr.involves(split_var):
                    supported = cstr.get_supported_values(split_var, {})
                    if val in supported:
                        score += 1.0 / max(1, len(supported))
            value_scores[val] = score

        # Sort by score and distribute round-robin for balance
        sorted_by_score = sorted(values, key=lambda v: value_scores.get(v, 0))
        parts: list[list[int]] = [[] for _ in range(num_parts)]
        for i, val in enumerate(sorted_by_score):
            parts[i % num_parts].append(val)

        sub_problems = []
        for i, part in enumerate(parts):
            if not part:
                continue
            difficulty = self._estimate_difficulty(split_var, frozenset(part))
            sub_problems.append(SubProblem(
                problem_id=i,
                split_variable=split_var,
                split_values=frozenset(part),
                estimated_difficulty=difficulty,
            ))

        return sub_problems

    def _select_split_variable(self) -> Variable | None:
        """Select the best variable to split on.

        Prefers variables with large domains and high constraint degree.
        """
        unassigned = [v for v in self._variables if not v.is_assigned]
        if not unassigned:
            return None

        def score(var: Variable) -> float:
            domain_score = var.domain_size
            degree_score = self._graph.degree(var)
            return domain_score * (1 + degree_score * 0.1)

        return max(unassigned, key=score)

    def _partition_values(self, values: list[int], num_parts: int) -> list[list[int]]:
        """Partition values into roughly equal parts."""
        parts: list[list[int]] = [[] for _ in range(num_parts)]
        for i, val in enumerate(values):
            parts[i % num_parts].append(val)
        return [p for p in parts if p]

    def _estimate_difficulty(self, var: Variable, values: frozenset[int]) -> float:
        """Estimate relative difficulty of a sub-problem.

        Based on the fraction of the domain and constraint tightness.
        """
        fraction = len(values) / max(1, var.domain_size)
        # More constrained values are harder
        constraint_factor = 1.0
        for cstr in self._constraints:
            if cstr.involves(var):
                supported = cstr.get_supported_values(var, {})
                overlap = len(values & supported)
                if overlap < len(values):
                    constraint_factor *= 1.2
        return fraction * constraint_factor
