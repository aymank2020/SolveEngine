"""Adaptive heuristic selection based on solver performance.

Monitors solver behavior during search and dynamically switches between
heuristics based on which one is performing better. Uses a multi-armed
bandit approach to balance exploration and exploitation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.heuristics.variable_ordering import (
    VariableSelector,
    MRVSelector,
    DegreeSelector,
    DomWdegSelector,
)


@dataclass
class HeuristicStats:
    """Performance statistics for a single heuristic."""

    name: str
    selections: int = 0
    total_nodes: int = 0
    total_backtracks: int = 0
    solutions_found: int = 0
    wipeouts_caused: int = 0

    @property
    def avg_nodes_per_solution(self) -> float:
        if self.solutions_found == 0:
            return float("inf")
        return self.total_nodes / self.solutions_found

    @property
    def backtrack_ratio(self) -> float:
        if self.total_nodes == 0:
            return 0.0
        return self.total_backtracks / self.total_nodes

    @property
    def score(self) -> float:
        """UCB1-style score for bandit selection."""
        if self.selections == 0:
            return float("inf")
        exploitation = 1.0 / (1.0 + self.backtrack_ratio)
        exploration = math.sqrt(2.0 * math.log(self.selections + 1) / (self.selections + 1))
        return exploitation + exploration


class AdaptiveHeuristic(VariableSelector):
    """Adaptive variable selector that switches between heuristics.

    Uses a UCB1 (Upper Confidence Bound) bandit algorithm to select
    which heuristic to use at each decision point. Heuristics that
    lead to fewer backtracks are preferred.
    """

    def __init__(
        self,
        heuristics: list[VariableSelector] | None = None,
        window_size: int = 50,
    ) -> None:
        self._heuristics = heuristics or [
            MRVSelector(),
            DegreeSelector(),
            DomWdegSelector(),
        ]
        self._stats: list[HeuristicStats] = [
            HeuristicStats(name=type(h).__name__) for h in self._heuristics
        ]
        self._window_size = window_size
        self._current_idx: int = 0
        self._decisions_since_switch: int = 0
        self._total_decisions: int = 0

    @property
    def current_heuristic(self) -> VariableSelector:
        return self._heuristics[self._current_idx]

    @property
    def stats(self) -> list[HeuristicStats]:
        return self._stats

    def select(
        self,
        unassigned: list[Variable],
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> Variable:
        """Select a variable using the currently active heuristic."""
        self._total_decisions += 1
        self._decisions_since_switch += 1

        # Periodically re-evaluate which heuristic to use
        if self._decisions_since_switch >= self._window_size:
            self._switch_heuristic()
            self._decisions_since_switch = 0

        self._stats[self._current_idx].selections += 1
        return self._heuristics[self._current_idx].select(
            unassigned, constraints, assignment
        )

    def record_backtrack(self) -> None:
        """Record a backtrack event for the current heuristic."""
        self._stats[self._current_idx].total_backtracks += 1

    def record_node(self) -> None:
        """Record a node exploration for the current heuristic."""
        self._stats[self._current_idx].total_nodes += 1

    def record_solution(self) -> None:
        """Record a solution found under the current heuristic."""
        self._stats[self._current_idx].solutions_found += 1

    def record_wipeout(self) -> None:
        """Record a domain wipeout under the current heuristic."""
        self._stats[self._current_idx].wipeouts_caused += 1

    def _switch_heuristic(self) -> None:
        """Select the best heuristic using UCB1 scores."""
        best_idx = 0
        best_score = -1.0

        for i, stat in enumerate(self._stats):
            score = stat.score
            if score > best_score:
                best_score = score
                best_idx = i

        self._current_idx = best_idx

    def get_best_heuristic(self) -> tuple[str, float]:
        """Return the name and score of the best-performing heuristic."""
        best = max(self._stats, key=lambda s: s.score)
        return best.name, best.score

    def reset_stats(self) -> None:
        """Reset all heuristic statistics."""
        for stat in self._stats:
            stat.selections = 0
            stat.total_nodes = 0
            stat.total_backtracks = 0
            stat.solutions_found = 0
            stat.wipeouts_caused = 0
        self._current_idx = 0
        self._decisions_since_switch = 0
        self._total_decisions = 0
