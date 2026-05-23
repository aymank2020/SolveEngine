"""Portfolio solver: runs multiple solver configurations in parallel.

A portfolio solver allocates a time budget across multiple solver
configurations and returns the first solution found. This is effective
when the best configuration is unknown a priori.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.solver.state import SolverStats
from solveengine.heuristics.variable_ordering import (
    VariableSelector,
    MRVSelector,
    DegreeSelector,
    DomWdegSelector,
)
from solveengine.heuristics.value_ordering import (
    ValueOrderer,
    AscendingOrderer,
    LCVOrderer,
    RandomOrderer,
)


@dataclass
class SolverConfig:
    """Configuration for a single solver in the portfolio."""

    name: str
    var_selector: VariableSelector
    val_orderer: ValueOrderer
    use_forward_check: bool = True
    node_budget: int = 10000


class PortfolioResult:
    """Result from portfolio solving."""

    def __init__(
        self,
        solution: dict[Variable, int] | None,
        winning_config: str | None,
        configs_tried: int,
        total_nodes: int,
    ) -> None:
        self._solution = solution
        self._winning_config = winning_config
        self._configs_tried = configs_tried
        self._total_nodes = total_nodes

    @property
    def solution(self) -> dict[Variable, int] | None:
        return self._solution

    @property
    def is_solved(self) -> bool:
        return self._solution is not None

    @property
    def winning_config(self) -> str | None:
        return self._winning_config

    @property
    def configs_tried(self) -> int:
        return self._configs_tried

    @property
    def total_nodes(self) -> int:
        return self._total_nodes

    def __repr__(self) -> str:
        if self._solution:
            return f"PortfolioResult(solved by {self._winning_config})"
        return f"PortfolioResult(unsolved, tried {self._configs_tried} configs)"


class PortfolioSolver:
    """Portfolio solver that tries multiple configurations sequentially.

    Each configuration gets a node budget. If it doesn't find a solution
    within its budget, the next configuration is tried. The first
    configuration to find a solution wins.
    """

    def __init__(self, configs: list[SolverConfig] | None = None) -> None:
        self._configs = configs or self._default_configs()
        self._results: list[tuple[str, SolverStats]] = []

    @property
    def configs(self) -> list[SolverConfig]:
        return self._configs

    @property
    def results(self) -> list[tuple[str, SolverStats]]:
        """Results from each configuration tried in the last solve."""
        return self._results

    def solve(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> PortfolioResult:
        """Try each configuration until one finds a solution."""
        self._results.clear()
        total_nodes = 0

        for i, config in enumerate(self._configs):
            # Create fresh variable copies for each attempt
            fresh_vars = self._copy_variables(variables)
            fresh_constraints = self._remap_constraints(
                constraints, variables, fresh_vars
            )

            solver = BacktrackSolver(
                fresh_vars,
                fresh_constraints,
                var_selector=config.var_selector,
                val_orderer=config.val_orderer,
                use_forward_check=config.use_forward_check,
            )
            solver.set_node_limit(config.node_budget)

            result = solver.solve()
            self._results.append((config.name, solver.stats))
            total_nodes += solver.stats.nodes_explored

            if result is not None:
                # Map back to original variables
                solution = self._map_solution(result, fresh_vars, variables)
                return PortfolioResult(
                    solution=solution,
                    winning_config=config.name,
                    configs_tried=i + 1,
                    total_nodes=total_nodes,
                )

        return PortfolioResult(
            solution=None,
            winning_config=None,
            configs_tried=len(self._configs),
            total_nodes=total_nodes,
        )

    def _copy_variables(self, variables: list[Variable]) -> list[Variable]:
        """Create fresh copies of variables with restored domains."""
        copies = []
        for var in variables:
            fresh = Variable(var.name, var.domain.values())
            copies.append(fresh)
        return copies

    def _remap_constraints(
        self,
        constraints: list[Constraint],
        original_vars: list[Variable],
        fresh_vars: list[Variable],
    ) -> list[Constraint]:
        """Remap constraints to use fresh variable copies.

        Note: This only works for constraints that can be reconstructed.
        For lambda-based constraints, we keep the originals (they reference
        original variables which won't be modified).
        """
        # Simple approach: return original constraints
        # (works because we check satisfaction against assignment dict)
        return constraints

    def _map_solution(
        self,
        result: dict[Variable, int],
        fresh_vars: list[Variable],
        original_vars: list[Variable],
    ) -> dict[Variable, int]:
        """Map solution from fresh variables back to originals."""
        solution = {}
        name_to_original = {v.name: v for v in original_vars}
        for var, val in result.items():
            if var.name in name_to_original:
                solution[name_to_original[var.name]] = val
        return solution

    @staticmethod
    def _default_configs() -> list[SolverConfig]:
        """Default portfolio: 4 diverse configurations."""
        return [
            SolverConfig(
                name="dom_wdeg+lcv",
                var_selector=DomWdegSelector(),
                val_orderer=LCVOrderer(),
                node_budget=50000,
            ),
            SolverConfig(
                name="mrv+ascending",
                var_selector=MRVSelector(),
                val_orderer=AscendingOrderer(),
                node_budget=50000,
            ),
            SolverConfig(
                name="degree+random",
                var_selector=DegreeSelector(),
                val_orderer=RandomOrderer(seed=42),
                node_budget=50000,
            ),
            SolverConfig(
                name="mrv+random",
                var_selector=MRVSelector(),
                val_orderer=RandomOrderer(seed=123),
                node_budget=100000,
            ),
        ]
