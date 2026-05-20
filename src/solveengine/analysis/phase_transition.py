"""Phase transition analysis for constraint satisfaction problems.

Studies the relationship between problem constrainedness and satisfiability.
CSP problems exhibit a phase transition: below a critical constraint density,
almost all instances are satisfiable; above it, almost all are unsatisfiable.
The hardest instances cluster near this critical point.

This module provides tools to:
- Estimate the constrainedness (kappa) of a CSP instance
- Predict satisfiability probability based on constraint density
- Identify whether a problem is in the easy-SAT, hard, or easy-UNSAT region
- Compute the critical constraint density for random CSP models
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint, BinaryConstraint


@dataclass
class PhaseTransitionResult:
    """Results of phase transition analysis."""

    constrainedness: float
    predicted_sat_probability: float
    constraint_density: float
    critical_density: float
    region: str  # "easy-sat", "critical", "easy-unsat"
    tightness: float
    domain_size_avg: float
    num_variables: int
    num_constraints: int

    @property
    def is_critical(self) -> bool:
        """Whether the problem is near the phase transition."""
        return self.region == "critical"

    @property
    def distance_from_critical(self) -> float:
        """Normalized distance from the critical point."""
        if self.critical_density == 0:
            return 0.0
        return (self.constraint_density - self.critical_density) / self.critical_density

    def summary(self) -> str:
        """Human-readable summary of the analysis."""
        lines = [
            f"Phase Transition Analysis:",
            f"  Variables: {self.num_variables}, Constraints: {self.num_constraints}",
            f"  Avg domain size: {self.domain_size_avg:.1f}",
            f"  Constraint density: {self.constraint_density:.4f}",
            f"  Critical density: {self.critical_density:.4f}",
            f"  Constrainedness (κ): {self.constrainedness:.4f}",
            f"  Predicted P(SAT): {self.predicted_sat_probability:.4f}",
            f"  Region: {self.region}",
        ]
        return "\n".join(lines)


class PhaseTransitionAnalyzer:
    """Analyzes CSP instances for phase transition properties.

    Uses statistical properties of the constraint graph to estimate
    where the instance falls relative to the satisfiability threshold.

    Args:
        variables: All CSP variables.
        constraints: All constraints.
    """

    def __init__(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> None:
        self._variables = variables
        self._constraints = constraints
        self._n = len(variables)
        self._e = len(constraints)

    def analyze(self) -> PhaseTransitionResult:
        """Perform full phase transition analysis."""
        domain_sizes = [v.domain_size for v in self._variables]
        avg_domain = sum(domain_sizes) / max(1, len(domain_sizes))
        tightness = self._compute_average_tightness()
        density = self._compute_constraint_density()
        critical = self._estimate_critical_density(avg_domain, tightness)
        kappa = self._compute_constrainedness(avg_domain, tightness)
        sat_prob = self._predict_sat_probability(kappa)
        region = self._classify_region(density, critical)

        return PhaseTransitionResult(
            constrainedness=kappa,
            predicted_sat_probability=sat_prob,
            constraint_density=density,
            critical_density=critical,
            region=region,
            tightness=tightness,
            domain_size_avg=avg_domain,
            num_variables=self._n,
            num_constraints=self._e,
        )

    def _compute_constraint_density(self) -> float:
        """Compute constraint density: ratio of constraints to max possible.

        For binary CSPs: density = e / (n choose 2)
        """
        if self._n < 2:
            return 0.0
        max_constraints = self._n * (self._n - 1) / 2
        return self._e / max_constraints

    def _compute_average_tightness(self) -> float:
        """Compute average constraint tightness.

        Tightness is the fraction of value pairs that are forbidden
        by a constraint. For non-binary constraints, we approximate.
        """
        if not self._constraints:
            return 0.0

        total_tightness = 0.0
        count = 0

        for cstr in self._constraints:
            tightness = self._constraint_tightness(cstr)
            if tightness >= 0:
                total_tightness += tightness
                count += 1

        return total_tightness / max(1, count)

    def _constraint_tightness(self, cstr: Constraint) -> float:
        """Compute tightness of a single constraint.

        Tightness = 1 - (allowed pairs / total pairs)
        """
        if cstr.arity != 2:
            return self._estimate_nary_tightness(cstr)

        vars = cstr.variables
        dom1 = vars[0].domain.values()
        dom2 = vars[1].domain.values()
        total_pairs = len(dom1) * len(dom2)

        if total_pairs == 0:
            return 1.0

        allowed = 0
        for v1 in dom1:
            for v2 in dom2:
                assignment = {vars[0]: v1, vars[1]: v2}
                if cstr.is_satisfied(assignment):
                    allowed += 1

        return 1.0 - (allowed / total_pairs)

    def _estimate_nary_tightness(self, cstr: Constraint) -> float:
        """Estimate tightness for n-ary constraints using sampling."""
        vars = cstr.variables
        if not vars:
            return 0.0

        sample_size = min(100, self._max_combinations(vars))
        satisfied = 0

        import random
        rng = random.Random(42)

        for _ in range(sample_size):
            assignment: dict[Variable, int] = {}
            for var in vars:
                values = list(var.domain)
                if values:
                    assignment[var] = rng.choice(values)
            if cstr.is_satisfied(assignment):
                satisfied += 1

        return 1.0 - (satisfied / max(1, sample_size))

    def _max_combinations(self, vars: tuple[Variable, ...]) -> int:
        """Compute total number of value combinations for variables."""
        product = 1
        for var in vars:
            product *= var.domain_size
            if product > 10000:
                return 10000
        return product

    def _compute_constrainedness(self, avg_domain: float, tightness: float) -> float:
        """Compute the constrainedness parameter κ (kappa).

        κ = 1 - (1-p)^(e/n) where p is tightness and e/n is constraint ratio.
        When κ ≈ 1, the problem is at the phase transition.
        κ < 1 means under-constrained (likely SAT).
        κ > 1 means over-constrained (likely UNSAT).
        """
        if self._n == 0 or avg_domain <= 1:
            return 0.0

        if tightness >= 1.0:
            return float('inf')
        if tightness <= 0.0:
            return 0.0

        constraint_ratio = self._e / max(1, self._n)
        log_sol_space = self._n * math.log(avg_domain)

        if log_sol_space == 0:
            return 0.0

        expected_log_solutions = log_sol_space + self._e * math.log(1.0 - tightness)
        kappa = 1.0 - (expected_log_solutions / log_sol_space)

        return max(0.0, kappa)

    def _estimate_critical_density(self, avg_domain: float, tightness: float) -> float:
        """Estimate the critical constraint density.

        For binary CSPs with domain size d and tightness p:
        critical density ≈ ln(d) / (-ln(1-p)) / (n-1) * 2

        This is derived from the expected number of solutions = 1 condition.
        """
        if avg_domain <= 1 or tightness <= 0 or tightness >= 1:
            return 0.5

        log_domain = math.log(avg_domain)
        neg_log_pass = -math.log(1.0 - tightness)

        if neg_log_pass == 0:
            return 0.5

        critical_edges = log_domain / neg_log_pass
        max_edges = max(1, self._n * (self._n - 1) / 2)
        critical_density = critical_edges * self._n / (2 * max_edges)

        return min(1.0, max(0.0, critical_density))

    def _predict_sat_probability(self, kappa: float) -> float:
        """Predict satisfiability probability from constrainedness.

        Uses a sigmoid-like function centered at κ = 1.
        P(SAT) ≈ 1 / (1 + exp(steepness * (κ - 1)))
        """
        steepness = 5.0 * math.sqrt(max(1, self._n))
        exponent = steepness * (kappa - 1.0)
        exponent = max(-500, min(500, exponent))
        return 1.0 / (1.0 + math.exp(exponent))

    def _classify_region(self, density: float, critical: float) -> str:
        """Classify the problem into a phase transition region."""
        if critical == 0:
            return "easy-sat"

        ratio = density / critical
        if ratio < 0.8:
            return "easy-sat"
        elif ratio > 1.2:
            return "easy-unsat"
        else:
            return "critical"

    def estimate_search_difficulty(self) -> float:
        """Estimate relative search difficulty on a 0-1 scale.

        Problems near the phase transition are hardest (score ≈ 1).
        Problems far from it are easier (score ≈ 0).
        """
        result = self.analyze()
        distance = abs(result.distance_from_critical)
        difficulty = math.exp(-2.0 * distance)
        return min(1.0, max(0.0, difficulty))

    def compute_backbone_estimate(self) -> float:
        """Estimate the backbone fraction of the problem.

        The backbone is the set of variables that take the same value
        in all solutions. Problems with large backbones are typically
        harder to solve. We estimate this from constrainedness.
        """
        result = self.analyze()
        kappa = result.constrainedness

        if kappa < 0.5:
            return 0.0
        elif kappa > 1.5:
            return 1.0
        else:
            return (kappa - 0.5) / 1.0
