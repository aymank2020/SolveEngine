"""Randomized search strategies for constraint satisfaction problems.

Provides randomized variants of variable ordering, value selection, and
restart strategies. Randomization helps escape heavy-tailed runtime
distributions and explore diverse regions of the search space.

Strategies include:
- Random variable ordering with configurable probability
- Random value selection with bias toward promising values
- Random restarts with different seeds for portfolio-style solving
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.heuristics.variable_ordering import VariableSelector, MRVSelector
from solveengine.heuristics.value_ordering import ValueOrderer, AscendingOrderer
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.solver.state import SolverStats


class RandomVariableSelector(VariableSelector):
    """Variable selector that randomly perturbs a base heuristic.

    With probability `randomness`, selects a random unassigned variable.
    Otherwise, defers to the base heuristic (default: MRV).

    This introduces diversity in the search while still benefiting from
    the base heuristic most of the time.

    Args:
        base_selector: The deterministic heuristic to use when not randomizing.
        randomness: Probability of selecting a random variable (0.0 to 1.0).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        base_selector: VariableSelector | None = None,
        randomness: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self._base = base_selector or MRVSelector()
        self._randomness = max(0.0, min(1.0, randomness))
        self._rng = random.Random(seed)
        self._selections_made: int = 0
        self._random_selections: int = 0

    @property
    def randomness(self) -> float:
        """Current randomness probability."""
        return self._randomness

    @property
    def selections_made(self) -> int:
        """Total number of selections made."""
        return self._selections_made

    @property
    def random_fraction(self) -> float:
        """Fraction of selections that were random."""
        if self._selections_made == 0:
            return 0.0
        return self._random_selections / self._selections_made

    def set_randomness(self, value: float) -> None:
        """Update the randomness probability."""
        self._randomness = max(0.0, min(1.0, value))

    def select(
        self,
        unassigned: list[Variable],
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> Variable:
        """Select a variable, randomly with probability `randomness`."""
        if not unassigned:
            raise ValueError("No unassigned variables")

        self._selections_made += 1

        if self._rng.random() < self._randomness:
            self._random_selections += 1
            return self._rng.choice(unassigned)

        return self._base.select(unassigned, constraints, assignment)


class WeightedRandomVariableSelector(VariableSelector):
    """Variable selector using weighted random sampling.

    Variables are selected with probability proportional to a weight
    function. By default, variables with smaller domains have higher
    weight (soft MRV), but custom weight functions can be provided.

    Args:
        weight_fn: Function mapping (variable, constraints, assignment) to weight.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        weight_fn: Callable[[Variable, list[Constraint], dict[Variable, int]], float] | None = None,
        seed: int | None = None,
    ) -> None:
        self._weight_fn = weight_fn or self._default_weight
        self._rng = random.Random(seed)

    @staticmethod
    def _default_weight(
        var: Variable,
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> float:
        """Default weight: inverse of domain size (smaller domain = higher weight)."""
        if var.domain_size == 0:
            return 0.0
        return 1.0 / var.domain_size

    def select(
        self,
        unassigned: list[Variable],
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> Variable:
        """Select a variable using weighted random sampling."""
        if not unassigned:
            raise ValueError("No unassigned variables")

        if len(unassigned) == 1:
            return unassigned[0]

        weights = [
            self._weight_fn(var, constraints, assignment)
            for var in unassigned
        ]

        total_weight = sum(weights)
        if total_weight <= 0:
            return self._rng.choice(unassigned)

        # Weighted random selection
        threshold = self._rng.random() * total_weight
        cumulative = 0.0
        for var, weight in zip(unassigned, weights):
            cumulative += weight
            if cumulative >= threshold:
                return var

        return unassigned[-1]


class RandomValueOrderer(ValueOrderer):
    """Value orderer that shuffles values with optional bias.

    Can bias toward values that satisfy more constraints (soft LCV)
    while still introducing randomness for diversity.

    Args:
        bias_strength: How strongly to bias toward good values (0.0 = pure random,
            1.0 = deterministic best-first). Default 0.3.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        bias_strength: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self._bias_strength = max(0.0, min(1.0, bias_strength))
        self._rng = random.Random(seed)

    def order(
        self,
        var: Variable,
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> list[int]:
        """Order values with randomized bias toward constraint-satisfying values."""
        values = list(var.domain.values())
        if len(values) <= 1:
            return values

        if self._bias_strength == 0.0:
            self._rng.shuffle(values)
            return values

        # Score each value by how many constraints it satisfies
        scored: list[tuple[float, int]] = []
        for value in values:
            score = self._score_value(var, value, constraints, assignment)
            # Add random noise scaled by (1 - bias_strength)
            noise = self._rng.random() * (1.0 - self._bias_strength)
            scored.append((score * self._bias_strength + noise, value))

        scored.sort(reverse=True)
        return [value for _, value in scored]

    def _score_value(
        self,
        var: Variable,
        value: int,
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> float:
        """Score a value by how many constraints it satisfies."""
        test_assignment = dict(assignment)
        test_assignment[var] = value

        satisfied = 0
        total = 0
        for cstr in constraints:
            if not cstr.involves(var):
                continue
            total += 1
            if cstr.is_satisfied(test_assignment):
                satisfied += 1

        if total == 0:
            return 1.0
        return satisfied / total


@dataclass
class RandomRestartResult:
    """Result from a random restart search run."""

    solution: dict[Variable, int] | None
    num_restarts: int
    total_nodes: int
    best_seed: int | None
    stats_per_restart: list[SolverStats]

    @property
    def is_solved(self) -> bool:
        return self.solution is not None


class RandomRestartSearch:
    """Search with random restarts using different seeds.

    Each restart uses a different random seed, producing a different
    search trajectory. This is effective against heavy-tailed runtime
    distributions where some seeds lead to quick solutions.

    Args:
        variables: List of CSP variables.
        constraints: List of constraints.
        max_restarts: Maximum number of restarts to attempt.
        nodes_per_restart: Node budget for each restart attempt.
        base_seed: Starting seed (subsequent restarts use base_seed + i).
        randomness: Probability of random variable selection per restart.
        value_bias: Bias strength for value ordering (0=random, 1=deterministic).
    """

    def __init__(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        max_restarts: int = 50,
        nodes_per_restart: int = 10000,
        base_seed: int = 42,
        randomness: float = 0.15,
        value_bias: float = 0.4,
    ) -> None:
        self._variables = variables
        self._constraints = constraints
        self._max_restarts = max_restarts
        self._nodes_per_restart = nodes_per_restart
        self._base_seed = base_seed
        self._randomness = randomness
        self._value_bias = value_bias

    def solve(self) -> RandomRestartResult:
        """Run random restart search.

        Tries multiple restarts with different seeds. Returns the first
        solution found, or None if all restarts are exhausted.
        """
        total_nodes = 0
        stats_list: list[SolverStats] = []

        for restart_idx in range(self._max_restarts):
            seed = self._base_seed + restart_idx

            # Restore all domains before each restart
            self._restore_all_domains()

            # Create randomized heuristics with this seed
            var_selector = RandomVariableSelector(
                base_selector=MRVSelector(),
                randomness=self._randomness,
                seed=seed,
            )
            val_orderer = RandomValueOrderer(
                bias_strength=self._value_bias,
                seed=seed,
            )

            solver = BacktrackSolver(
                self._variables,
                self._constraints,
                var_selector=var_selector,
                val_orderer=val_orderer,
                use_forward_check=True,
            )
            solver.set_node_limit(self._nodes_per_restart)

            result = solver.solve()
            stats = solver.stats
            total_nodes += stats.nodes_explored
            stats_list.append(stats)

            if result is not None:
                return RandomRestartResult(
                    solution=result,
                    num_restarts=restart_idx + 1,
                    total_nodes=total_nodes,
                    best_seed=seed,
                    stats_per_restart=stats_list,
                )

        return RandomRestartResult(
            solution=None,
            num_restarts=self._max_restarts,
            total_nodes=total_nodes,
            best_seed=None,
            stats_per_restart=stats_list,
        )

    def solve_parallel_seeds(self, num_seeds: int = 4) -> RandomRestartResult:
        """Simulate parallel solving with multiple seeds.

        Runs each seed sequentially but returns the result from the seed
        that would have finished first (fewest nodes explored).
        """
        best_result: dict[Variable, int] | None = None
        best_nodes = float("inf")
        best_seed_val: int | None = None
        total_nodes = 0
        stats_list: list[SolverStats] = []

        for i in range(num_seeds):
            seed = self._base_seed + i * 1000
            self._restore_all_domains()

            var_selector = RandomVariableSelector(
                base_selector=MRVSelector(),
                randomness=self._randomness,
                seed=seed,
            )
            val_orderer = RandomValueOrderer(
                bias_strength=self._value_bias,
                seed=seed,
            )

            solver = BacktrackSolver(
                self._variables,
                self._constraints,
                var_selector=var_selector,
                val_orderer=val_orderer,
                use_forward_check=True,
            )
            solver.set_node_limit(self._nodes_per_restart)

            result = solver.solve()
            stats = solver.stats
            total_nodes += stats.nodes_explored
            stats_list.append(stats)

            if result is not None and stats.nodes_explored < best_nodes:
                best_result = result
                best_nodes = stats.nodes_explored
                best_seed_val = seed

        return RandomRestartResult(
            solution=best_result,
            num_restarts=num_seeds,
            total_nodes=total_nodes,
            best_seed=best_seed_val,
            stats_per_restart=stats_list,
        )

    def _restore_all_domains(self) -> None:
        """Restore all variable domains to their initial state."""
        for var in self._variables:
            var.restore_to(0)
            var.unassign()
