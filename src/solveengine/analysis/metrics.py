"""Graph metrics for constraint problem analysis.

Computes structural metrics that predict problem difficulty and
guide solver configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.decomposition.graph import ConstraintGraph


@dataclass
class ProblemMetrics:
    """Structural metrics for a CSP instance."""

    num_variables: int
    num_constraints: int
    avg_domain_size: float
    max_domain_size: int
    min_domain_size: int
    graph_density: float
    avg_degree: float
    max_degree: int
    num_components: int
    constraint_tightness: float  # Average fraction of forbidden tuples

    @property
    def kappa(self) -> float:
        """Constrainedness parameter (kappa).

        Higher values indicate more constrained (harder) problems.
        kappa ≈ 1 is the phase transition region.
        """
        if self.num_variables == 0 or self.avg_domain_size <= 1:
            return 0.0
        import math
        n = self.num_variables
        d = self.avg_domain_size
        e = self.num_constraints
        t = self.constraint_tightness
        if t <= 0 or d <= 0:
            return 0.0
        return e * (-math.log(1 - t)) / (n * math.log(d))


class GraphMetrics:
    """Computes structural metrics for a CSP problem."""

    def __init__(self, variables: list[Variable], constraints: list[Constraint]) -> None:
        self._variables = variables
        self._constraints = constraints
        self._graph = ConstraintGraph(variables, constraints)

    def compute(self) -> ProblemMetrics:
        """Compute all metrics for the problem."""
        n = len(self._variables)
        if n == 0:
            return ProblemMetrics(
                num_variables=0, num_constraints=0, avg_domain_size=0,
                max_domain_size=0, min_domain_size=0, graph_density=0,
                avg_degree=0, max_degree=0, num_components=0,
                constraint_tightness=0,
            )

        domain_sizes = [v.domain_size for v in self._variables]
        degrees = [self._graph.degree(v) for v in self._variables]
        components = self._graph.connected_components()

        return ProblemMetrics(
            num_variables=n,
            num_constraints=len(self._constraints),
            avg_domain_size=sum(domain_sizes) / n,
            max_domain_size=max(domain_sizes),
            min_domain_size=min(domain_sizes),
            graph_density=self._graph.density(),
            avg_degree=sum(degrees) / n,
            max_degree=max(degrees) if degrees else 0,
            num_components=len(components),
            constraint_tightness=self._estimate_tightness(),
        )

    def _estimate_tightness(self) -> float:
        """Estimate average constraint tightness by sampling.

        Tightness = fraction of value pairs that violate the constraint.
        """
        if not self._constraints:
            return 0.0

        total_tightness = 0.0
        count = 0

        for cstr in self._constraints:
            if cstr.arity != 2:
                continue
            vars_in = cstr.variables
            v1, v2 = vars_in[0], vars_in[1]
            total_pairs = v1.domain_size * v2.domain_size
            if total_pairs == 0:
                continue

            violations = 0
            for val1 in v1.domain:
                for val2 in v2.domain:
                    if not cstr.is_satisfied({v1: val1, v2: val2}):
                        violations += 1

            total_tightness += violations / total_pairs
            count += 1

        return total_tightness / count if count > 0 else 0.0

    def suggest_heuristic(self) -> str:
        """Suggest a variable ordering heuristic based on problem structure."""
        metrics = self.compute()

        if metrics.num_components > 1:
            return "component_decomposition"
        if metrics.kappa > 0.8:
            return "dom_wdeg"  # Hard problems benefit from learning
        if metrics.graph_density > 0.5:
            return "mrv"  # Dense graphs: domain size matters most
        if metrics.avg_degree > 5:
            return "dom_wdeg"
        return "mrv_degree"  # Default: MRV with degree tiebreaker
