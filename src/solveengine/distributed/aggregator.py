"""Result aggregation for distributed solving.

Collects results from multiple sub-problem solvers and combines them
into a unified result. Handles partial results, timeouts, and
solution deduplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from solveengine.core.variable import Variable


class SubProblemStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SOLVED = "solved"
    UNSATISFIABLE = "unsatisfiable"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class SubProblemResult:
    """Result from solving a single sub-problem."""

    problem_id: int
    status: SubProblemStatus
    solution: dict[Variable, int] | None = None
    nodes_explored: int = 0
    time_ms: float = 0.0
    error_message: str = ""


@dataclass
class AggregatedResult:
    """Combined result from all sub-problems."""

    solutions: list[dict[Variable, int]] = field(default_factory=list)
    total_nodes: int = 0
    total_time_ms: float = 0.0
    sub_results: list[SubProblemResult] = field(default_factory=list)

    @property
    def is_solved(self) -> bool:
        return len(self.solutions) > 0

    @property
    def is_proven_unsat(self) -> bool:
        """True if all sub-problems are unsatisfiable (proves global unsat)."""
        return all(
            r.status == SubProblemStatus.UNSATISFIABLE for r in self.sub_results
        )

    @property
    def num_completed(self) -> int:
        return sum(
            1 for r in self.sub_results
            if r.status in (SubProblemStatus.SOLVED, SubProblemStatus.UNSATISFIABLE)
        )

    @property
    def num_pending(self) -> int:
        return sum(
            1 for r in self.sub_results
            if r.status in (SubProblemStatus.PENDING, SubProblemStatus.RUNNING)
        )


class ResultAggregator:
    """Aggregates results from distributed sub-problem solving.

    Maintains state across multiple sub-problem completions and
    provides a unified view of the solving progress.
    """

    def __init__(self, num_sub_problems: int) -> None:
        self._num_sub_problems = num_sub_problems
        self._results: dict[int, SubProblemResult] = {}
        self._solutions: list[dict[Variable, int]] = []
        self._first_solution_at: int | None = None

    @property
    def num_sub_problems(self) -> int:
        return self._num_sub_problems

    @property
    def is_complete(self) -> bool:
        """True if all sub-problems have reported results."""
        return len(self._results) == self._num_sub_problems

    @property
    def has_solution(self) -> bool:
        return len(self._solutions) > 0

    @property
    def first_solution_index(self) -> int | None:
        """Index of the sub-problem that found the first solution."""
        return self._first_solution_at

    def submit_result(self, result: SubProblemResult) -> None:
        """Submit a result from a completed sub-problem."""
        self._results[result.problem_id] = result

        if result.status == SubProblemStatus.SOLVED and result.solution is not None:
            self._solutions.append(result.solution)
            if self._first_solution_at is None:
                self._first_solution_at = result.problem_id

    def get_aggregated(self) -> AggregatedResult:
        """Get the current aggregated result."""
        total_nodes = sum(r.nodes_explored for r in self._results.values())
        total_time = max((r.time_ms for r in self._results.values()), default=0.0)

        return AggregatedResult(
            solutions=list(self._solutions),
            total_nodes=total_nodes,
            total_time_ms=total_time,
            sub_results=list(self._results.values()),
        )

    def should_cancel_remaining(self) -> bool:
        """Check if remaining sub-problems can be cancelled.

        Returns True if we already have a solution and only need one.
        """
        return self.has_solution

    def progress(self) -> float:
        """Return solving progress as a fraction [0, 1]."""
        if self._num_sub_problems == 0:
            return 1.0
        completed = sum(
            1 for r in self._results.values()
            if r.status != SubProblemStatus.PENDING
        )
        return completed / self._num_sub_problems

    def reset(self) -> None:
        """Reset the aggregator for a new solving session."""
        self._results.clear()
        self._solutions.clear()
        self._first_solution_at = None
