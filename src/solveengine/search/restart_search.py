"""Search with restarts integration.

Combines the backtracking solver with restart policies to escape
heavy-tailed runtime distributions. After each restart, the solver
begins a new search with potentially different variable/value orderings
informed by the weights learned during previous attempts.
"""

from __future__ import annotations

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.solver.state import SolverStats
from solveengine.optimizer.restarts import RestartPolicy, LubyRestart
from solveengine.heuristics.variable_ordering import VariableSelector, DomWdegSelector
from solveengine.heuristics.value_ordering import ValueOrderer, LCVOrderer


class RestartSearchResult:
    """Result from restart-based search."""

    def __init__(
        self,
        solution: dict[Variable, int] | None,
        num_restarts: int,
        total_nodes: int,
        stats: SolverStats,
    ) -> None:
        self._solution = solution
        self._num_restarts = num_restarts
        self._total_nodes = total_nodes
        self._stats = stats

    @property
    def solution(self) -> dict[Variable, int] | None:
        return self._solution

    @property
    def is_solved(self) -> bool:
        return self._solution is not None

    @property
    def num_restarts(self) -> int:
        return self._num_restarts

    @property
    def total_nodes(self) -> int:
        return self._total_nodes

    @property
    def stats(self) -> SolverStats:
        return self._stats


class RestartSearch:
    """Solver with restart support.

    Runs the backtracking solver with a node budget determined by the
    restart policy. After each restart:
    - Domain weights are preserved (dom/wdeg learning persists)
    - Domains are restored to their initial state
    - A new search begins with the updated heuristic weights
    """

    def __init__(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        restart_policy: RestartPolicy | None = None,
        var_selector: VariableSelector | None = None,
        val_orderer: ValueOrderer | None = None,
        max_restarts: int = 100,
        total_node_limit: int = 1000000,
    ) -> None:
        self._variables = variables
        self._constraints = constraints
        self._policy = restart_policy or LubyRestart(unit=100)
        self._var_selector = var_selector or DomWdegSelector()
        self._val_orderer = val_orderer or LCVOrderer()
        self._max_restarts = max_restarts
        self._total_node_limit = total_node_limit

    def solve(self) -> RestartSearchResult:
        """Run search with restarts until solution or limit reached."""
        total_nodes = 0
        combined_stats = SolverStats()

        for restart_num in range(self._max_restarts + 1):
            # Compute node budget for this attempt
            budget = self._policy.next_cutoff()
            remaining = self._total_node_limit - total_nodes
            if remaining <= 0:
                break
            budget = min(budget, remaining)

            # Restore domains for fresh search
            # (weights are preserved for dom/wdeg learning)
            self._restore_domains()

            # Create solver with current budget
            solver = BacktrackSolver(
                self._variables,
                self._constraints,
                var_selector=self._var_selector,
                val_orderer=self._val_orderer,
                use_forward_check=True,
            )
            solver.set_node_limit(budget)

            result = solver.solve()
            stats = solver.stats
            total_nodes += stats.nodes_explored
            combined_stats.nodes_explored += stats.nodes_explored
            combined_stats.backtracks += stats.backtracks
            combined_stats.propagations += stats.propagations

            if result is not None:
                combined_stats.solutions_found = 1
                return RestartSearchResult(
                    solution=result,
                    num_restarts=restart_num,
                    total_nodes=total_nodes,
                    stats=combined_stats,
                )

            # Record restart and continue
            self._policy.record_restart(stats.nodes_explored)

        return RestartSearchResult(
            solution=None,
            num_restarts=self._policy.restart_count,
            total_nodes=total_nodes,
            stats=combined_stats,
        )

    def _restore_domains(self) -> None:
        """Restore all variable domains to initial state.

        Preserves weights for dom/wdeg heuristic learning.
        """
        for var in self._variables:
            var.restore_to(0)
            var.unassign()
