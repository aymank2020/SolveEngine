"""Solution counting for constraint satisfaction problems.

Provides methods to count (or estimate) the number of solutions without
fully enumerating them. Techniques include:
- Exact counting via exhaustive search (for small problems)
- Inclusion-Exclusion for independent components
- Upper/lower bound estimation using constraint tightness
- Sampling-based estimation for large problems

Solution counting is useful for:
- Measuring problem difficulty
- Comparing constraint formulations
- Validating solver correctness
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.decomposition.graph import ConstraintGraph
from solveengine.heuristics.variable_ordering import MRVSelector
from solveengine.heuristics.value_ordering import AscendingOrderer


@dataclass
class CountResult:
    """Result of solution counting.

    Attributes:
        exact_count: Exact number of solutions (if computed).
        lower_bound: Lower bound on solution count.
        upper_bound: Upper bound on solution count.
        is_exact: Whether the count is exact.
        components_counted: Number of independent components processed.
        nodes_explored: Total search nodes explored during counting.
    """

    exact_count: int | None = None
    lower_bound: int = 0
    upper_bound: int = 0
    is_exact: bool = False
    components_counted: int = 0
    nodes_explored: int = 0

    @property
    def count(self) -> int:
        """Best available count (exact if available, else upper bound)."""
        if self.exact_count is not None:
            return self.exact_count
        return self.upper_bound

    @property
    def has_solutions(self) -> bool:
        """Whether at least one solution exists."""
        if self.exact_count is not None:
            return self.exact_count > 0
        return self.lower_bound > 0


class SolutionCounter:
    """Counts solutions to a CSP using various techniques.

    For small problems, performs exact counting. For larger problems,
    uses decomposition and estimation to provide bounds.

    Args:
        variables: List of CSP variables.
        constraints: List of constraints.
        exact_limit: Maximum problem size for exact counting (product of domain sizes).
        sample_size: Number of samples for estimation.
    """

    def __init__(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        exact_limit: int = 1000000,
        sample_size: int = 1000,
    ) -> None:
        self._variables = variables
        self._constraints = constraints
        self._exact_limit = exact_limit
        self._sample_size = sample_size
        self._graph = ConstraintGraph(variables, constraints)

    def count(self) -> CountResult:
        """Count solutions using the best available method.

        Automatically selects between exact counting, component
        decomposition, and estimation based on problem size.
        """
        # Check if problem is small enough for exact counting
        search_space = self._compute_search_space_size()

        if search_space <= self._exact_limit:
            return self._exact_count()

        # Try component decomposition
        components = self._graph.connected_components()
        if len(components) > 1:
            return self._count_by_components(components)

        # Fall back to bounds estimation
        return self._estimate_bounds()

    def exact_count(self) -> int:
        """Perform exact solution counting via exhaustive search.

        Warning: This can be very slow for large problems.
        """
        result = self._exact_count()
        return result.exact_count if result.exact_count is not None else 0

    def _exact_count(self) -> CountResult:
        """Count all solutions exactly using backtracking."""
        self._restore_domains()
        solver = BacktrackSolver(
            self._variables,
            self._constraints,
            var_selector=MRVSelector(),
            val_orderer=AscendingOrderer(),
            use_forward_check=True,
        )
        solutions = solver.solve_all()
        count = len(solutions)
        nodes = solver.stats.nodes_explored

        return CountResult(
            exact_count=count,
            lower_bound=count,
            upper_bound=count,
            is_exact=True,
            components_counted=1,
            nodes_explored=nodes,
        )

    def _count_by_components(
        self, components: list[list[Variable]]
    ) -> CountResult:
        """Count solutions using Inclusion-Exclusion on independent components.

        For independent components, the total solution count is the product
        of solution counts for each component.
        """
        total_count = 1
        total_nodes = 0
        all_exact = True

        for component_vars in components:
            # Get constraints for this component
            component_set = set(component_vars)
            component_constraints = [
                c for c in self._constraints
                if all(v in component_set for v in c.variables)
            ]

            # Count solutions for this component
            component_counter = SolutionCounter(
                component_vars,
                component_constraints,
                exact_limit=self._exact_limit,
                sample_size=self._sample_size,
            )
            component_result = component_counter.count()
            total_nodes += component_result.nodes_explored

            if component_result.is_exact and component_result.exact_count is not None:
                if component_result.exact_count == 0:
                    return CountResult(
                        exact_count=0,
                        lower_bound=0,
                        upper_bound=0,
                        is_exact=True,
                        components_counted=len(components),
                        nodes_explored=total_nodes,
                    )
                total_count *= component_result.exact_count
            else:
                all_exact = False
                total_count *= component_result.upper_bound

        return CountResult(
            exact_count=total_count if all_exact else None,
            lower_bound=1 if total_count > 0 else 0,
            upper_bound=total_count,
            is_exact=all_exact,
            components_counted=len(components),
            nodes_explored=total_nodes,
        )

    def _estimate_bounds(self) -> CountResult:
        """Estimate upper and lower bounds on solution count.

        Upper bound: product of domain sizes divided by constraint tightness.
        Lower bound: sampling-based estimation.
        """
        upper = self._compute_upper_bound()
        lower = self._compute_lower_bound()

        return CountResult(
            exact_count=None,
            lower_bound=lower,
            upper_bound=upper,
            is_exact=False,
            components_counted=1,
            nodes_explored=0,
        )

    def _compute_upper_bound(self) -> int:
        """Compute an upper bound using constraint tightness.

        For each constraint, estimate the fraction of tuples it allows.
        The upper bound is the product of domain sizes times the product
        of allowed fractions.
        """
        # Start with unconstrained count (product of domain sizes)
        unconstrained = 1
        for var in self._variables:
            unconstrained *= var.domain_size

        if unconstrained == 0:
            return 0

        # Reduce by estimated constraint tightness
        reduction_factor = 1.0
        for cstr in self._constraints:
            tightness = self._estimate_constraint_tightness(cstr)
            allowed_fraction = 1.0 - tightness
            if allowed_fraction <= 0:
                return 0
            reduction_factor *= allowed_fraction

        upper = int(unconstrained * reduction_factor)
        return max(upper, 0)

    def _compute_lower_bound(self) -> int:
        """Compute a lower bound using random sampling.

        Tries to find solutions using random assignments and counts
        successes. The lower bound is at least 1 if any solution is found.
        """
        self._restore_domains()

        # Try to find at least one solution
        solver = BacktrackSolver(
            self._variables,
            self._constraints,
            var_selector=MRVSelector(),
            val_orderer=AscendingOrderer(),
            use_forward_check=True,
        )
        solver.set_node_limit(10000)
        solution = solver.solve()

        if solution is None:
            return 0

        # Found at least one solution
        # Try sampling to estimate density
        successes = self._sample_solutions()
        if successes > 0:
            # Estimate: (successes / samples) * search_space
            search_space = self._compute_search_space_size()
            estimated = int((successes / self._sample_size) * search_space)
            return max(estimated, 1)

        return 1  # At least the one solution we found

    def _sample_solutions(self) -> int:
        """Count how many random complete assignments satisfy all constraints."""
        rng = random.Random(42)
        successes = 0

        for _ in range(self._sample_size):
            assignment: dict[Variable, int] = {}
            for var in self._variables:
                values = list(var.domain.values())
                assignment[var] = rng.choice(values)

            if self._is_consistent(assignment):
                successes += 1

        return successes

    def _is_consistent(self, assignment: dict[Variable, int]) -> bool:
        """Check if a complete assignment satisfies all constraints."""
        for cstr in self._constraints:
            if not cstr.is_satisfied(assignment):
                return False
        return True

    def _estimate_constraint_tightness(self, cstr: Constraint) -> float:
        """Estimate the tightness of a single constraint.

        Tightness = fraction of tuples that violate the constraint.
        For large domains, uses sampling.
        """
        vars_in = cstr.variables
        if len(vars_in) == 0:
            return 0.0

        # Compute total possible tuples
        total_tuples = 1
        for var in vars_in:
            total_tuples *= var.domain_size

        if total_tuples == 0:
            return 1.0

        # For small spaces, check exhaustively
        if total_tuples <= 10000:
            violations = 0
            checked = 0
            self._enumerate_and_check(cstr, vars_in, 0, {}, violations_counter=[0], total_counter=[0])
            violations = self._last_violations
            checked = self._last_checked
            if checked == 0:
                return 0.0
            return violations / checked

        # For large spaces, sample
        rng = random.Random(123)
        sample_count = min(1000, total_tuples)
        violations = 0

        for _ in range(sample_count):
            assignment: dict[Variable, int] = {}
            for var in vars_in:
                values = list(var.domain.values())
                assignment[var] = rng.choice(values)
            if not cstr.is_satisfied(assignment):
                violations += 1

        return violations / sample_count

    def _enumerate_and_check(
        self,
        cstr: Constraint,
        variables: tuple[Variable, ...],
        idx: int,
        assignment: dict[Variable, int],
        violations_counter: list[int],
        total_counter: list[int],
    ) -> None:
        """Recursively enumerate tuples and count violations."""
        if idx == len(variables):
            total_counter[0] += 1
            if not cstr.is_satisfied(assignment):
                violations_counter[0] += 1
            self._last_violations = violations_counter[0]
            self._last_checked = total_counter[0]
            return

        var = variables[idx]
        for value in var.domain.values():
            assignment[var] = value
            self._enumerate_and_check(
                cstr, variables, idx + 1, assignment,
                violations_counter, total_counter,
            )
        if var in assignment:
            del assignment[var]

    def _compute_search_space_size(self) -> int:
        """Compute the total search space size (product of domain sizes)."""
        size = 1
        for var in self._variables:
            size *= var.domain_size
        return size

    def _restore_domains(self) -> None:
        """Restore all variable domains to initial state."""
        for var in self._variables:
            var.restore_to(0)
            var.unassign()


def count_solutions(
    variables: list[Variable], constraints: list[Constraint]
) -> int:
    """Convenience function to count solutions exactly.

    Warning: May be slow for large problems.
    """
    counter = SolutionCounter(variables, constraints)
    return counter.exact_count()


def estimate_solution_count(
    variables: list[Variable],
    constraints: list[Constraint],
    samples: int = 10000,
) -> tuple[int, int]:
    """Estimate solution count bounds.

    Returns (lower_bound, upper_bound) tuple.
    """
    counter = SolutionCounter(
        variables, constraints, sample_size=samples
    )
    result = counter.count()
    return (result.lower_bound, result.upper_bound)
