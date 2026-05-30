"""Beam search strategy for constraint satisfaction problems.

Beam search maintains the k best partial solutions at each level of the
search tree. At each step, all k partial solutions are expanded by trying
all possible values for the next variable, then pruned back to the k best
candidates based on a scoring function.

This provides a middle ground between DFS (k=1) and BFS (k=∞), trading
completeness for efficiency on large problems.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Iterator

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.heuristics.variable_ordering import VariableSelector, MRVSelector


@dataclass(order=True)
class BeamCandidate:
    """A candidate partial solution in the beam.

    Ordered by score (lower is better for min-heap usage).
    """

    score: float
    assignment: dict[Variable, int] = field(compare=False)
    depth: int = field(compare=False, default=0)
    violations: int = field(compare=False, default=0)

    def copy(self) -> BeamCandidate:
        """Create an independent copy of this candidate."""
        return BeamCandidate(
            score=self.score,
            assignment=dict(self.assignment),
            depth=self.depth,
            violations=self.violations,
        )


class BeamScorer:
    """Scores partial assignments for beam search ranking.

    The default scoring function combines:
    - Number of constraint violations (weighted heavily)
    - Sum of remaining domain sizes (smaller is more constrained)
    - Depth (deeper assignments are preferred)
    """

    def __init__(
        self,
        violation_weight: float = 100.0,
        domain_weight: float = 1.0,
        depth_bonus: float = -5.0,
    ) -> None:
        self._violation_weight = violation_weight
        self._domain_weight = domain_weight
        self._depth_bonus = depth_bonus

    def score(
        self,
        assignment: dict[Variable, int],
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> float:
        """Compute a score for the partial assignment (lower is better)."""
        violations = self._count_violations(assignment, constraints)
        remaining_domain = self._remaining_domain_sum(assignment, variables)
        depth = len(assignment)

        return (
            self._violation_weight * violations
            + self._domain_weight * remaining_domain
            + self._depth_bonus * depth
        )

    def _count_violations(
        self, assignment: dict[Variable, int], constraints: list[Constraint]
    ) -> int:
        """Count constraints violated by the current partial assignment."""
        count = 0
        for cstr in constraints:
            assigned_vars = [v for v in cstr.variables if v in assignment]
            if len(assigned_vars) == cstr.arity:
                if not cstr.is_satisfied(assignment):
                    count += 1
        return count

    def _remaining_domain_sum(
        self, assignment: dict[Variable, int], variables: list[Variable]
    ) -> float:
        """Sum of domain sizes for unassigned variables."""
        total = 0.0
        for var in variables:
            if var not in assignment:
                total += var.domain_size
        return total


class BeamSearch:
    """Beam search strategy for CSP solving.

    Maintains k best partial solutions and expands them level by level.
    At each level, one variable is selected and all candidates are expanded
    with all possible values, then pruned back to the k best.

    Args:
        variables: List of CSP variables.
        constraints: List of constraints.
        beam_width: Number of candidates to keep at each level (k).
        var_selector: Heuristic for choosing which variable to assign next.
        scorer: Scoring function for ranking candidates.
        max_expansions: Maximum total expansions before giving up.
    """

    def __init__(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        beam_width: int = 10,
        var_selector: VariableSelector | None = None,
        scorer: BeamScorer | None = None,
        max_expansions: int = 100000,
    ) -> None:
        self._variables = variables
        self._constraints = constraints
        self._beam_width = max(1, beam_width)
        self._var_selector = var_selector or MRVSelector()
        self._scorer = scorer or BeamScorer()
        self._max_expansions = max_expansions
        self._expansions: int = 0
        self._levels_explored: int = 0

    @property
    def beam_width(self) -> int:
        """Current beam width."""
        return self._beam_width

    @property
    def expansions(self) -> int:
        """Total number of candidate expansions performed."""
        return self._expansions

    @property
    def levels_explored(self) -> int:
        """Number of levels (variables) explored."""
        return self._levels_explored

    def solve(self) -> dict[Variable, int] | None:
        """Find a solution using beam search.

        Returns a complete consistent assignment, or None if no solution
        is found within the beam width and expansion limits.
        """
        self._expansions = 0
        self._levels_explored = 0

        initial_score = self._scorer.score({}, self._variables, self._constraints)
        beam: list[BeamCandidate] = [
            BeamCandidate(score=initial_score, assignment={}, depth=0, violations=0)
        ]

        for level in range(len(self._variables)):
            if not beam:
                return None

            self._levels_explored = level + 1
            next_beam: list[BeamCandidate] = []

            for candidate in beam:
                if self._expansions >= self._max_expansions:
                    return self._best_complete_solution(beam)

                expansions = self._expand_candidate(candidate)
                for expanded in expansions:
                    self._expansions += 1
                    if self._is_complete(expanded):
                        if self._is_consistent(expanded.assignment):
                            return expanded.assignment
                    next_beam.append(expanded)

            beam = self._prune_beam(next_beam)

        return self._best_complete_solution(beam)

    def solve_top_k(self, k: int = 5) -> list[dict[Variable, int]]:
        """Find up to k solutions using beam search with wider beam.

        Returns a list of complete consistent assignments found.
        """
        self._expansions = 0
        self._levels_explored = 0
        solutions: list[dict[Variable, int]] = []

        initial_score = self._scorer.score({}, self._variables, self._constraints)
        beam: list[BeamCandidate] = [
            BeamCandidate(score=initial_score, assignment={}, depth=0, violations=0)
        ]

        for level in range(len(self._variables)):
            if not beam:
                break

            self._levels_explored = level + 1
            next_beam: list[BeamCandidate] = []

            for candidate in beam:
                if self._expansions >= self._max_expansions:
                    break
                expansions = self._expand_candidate(candidate)
                for expanded in expansions:
                    self._expansions += 1
                    if self._is_complete(expanded):
                        if self._is_consistent(expanded.assignment):
                            solutions.append(expanded.assignment)
                            if len(solutions) >= k:
                                return solutions
                    else:
                        next_beam.append(expanded)

            beam = self._prune_beam(next_beam)

        return solutions

    def _expand_candidate(self, candidate: BeamCandidate) -> list[BeamCandidate]:
        """Expand a candidate by assigning all possible values to the next variable."""
        unassigned = [v for v in self._variables if v not in candidate.assignment]
        if not unassigned:
            return []

        var = self._var_selector.select(unassigned, self._constraints, candidate.assignment)
        expansions: list[BeamCandidate] = []

        for value in var.domain:
            new_assignment = dict(candidate.assignment)
            new_assignment[var] = value

            if not self._is_locally_consistent(var, value, new_assignment):
                continue

            score = self._scorer.score(new_assignment, self._variables, self._constraints)
            violations = self._count_full_violations(new_assignment)

            expansions.append(BeamCandidate(
                score=score,
                assignment=new_assignment,
                depth=candidate.depth + 1,
                violations=violations,
            ))

        return expansions

    def _prune_beam(self, candidates: list[BeamCandidate]) -> list[BeamCandidate]:
        """Prune candidates to the beam width, keeping the k best."""
        if len(candidates) <= self._beam_width:
            return candidates
        return heapq.nsmallest(self._beam_width, candidates)

    def _is_complete(self, candidate: BeamCandidate) -> bool:
        """Check if a candidate is a complete assignment."""
        return candidate.depth == len(self._variables)

    def _is_consistent(self, assignment: dict[Variable, int]) -> bool:
        """Check if a complete assignment satisfies all constraints."""
        for cstr in self._constraints:
            if not cstr.is_satisfied(assignment):
                return False
        return True

    def _is_locally_consistent(
        self, var: Variable, value: int, assignment: dict[Variable, int]
    ) -> bool:
        """Check if assigning value to var is consistent with current assignment."""
        for cstr in self._constraints:
            if not cstr.involves(var):
                continue
            if not cstr.is_satisfied(assignment):
                return False
        return True

    def _count_full_violations(self, assignment: dict[Variable, int]) -> int:
        """Count total constraint violations in the assignment."""
        count = 0
        for cstr in self._constraints:
            assigned_vars = [v for v in cstr.variables if v in assignment]
            if len(assigned_vars) == cstr.arity:
                if not cstr.is_satisfied(assignment):
                    count += 1
        return count

    def _best_complete_solution(
        self, beam: list[BeamCandidate]
    ) -> dict[Variable, int] | None:
        """Find the best complete and consistent solution in the beam."""
        complete = [
            c for c in beam
            if self._is_complete(c) and self._is_consistent(c.assignment)
        ]
        if complete:
            best = min(complete, key=lambda c: c.score)
            return best.assignment
        return None
