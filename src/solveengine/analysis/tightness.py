"""Constraint tightness analysis.

Analyzes how tight individual constraints are and identifies
bottleneck constraints that are likely to cause failures.
"""

from __future__ import annotations

from dataclasses import dataclass

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


@dataclass
class ConstraintTightness:
    """Tightness information for a single constraint."""

    constraint: Constraint
    tightness: float  # Fraction of forbidden tuples [0, 1]
    support_ratio: float  # Average support per variable value
    is_bottleneck: bool  # True if tightness > threshold


class TightnessAnalyzer:
    """Analyzes constraint tightness to identify bottlenecks.

    Tight constraints (high fraction of forbidden tuples) are more
    likely to cause failures and should be prioritized in propagation.
    """

    def __init__(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        bottleneck_threshold: float = 0.7,
    ) -> None:
        self._variables = variables
        self._constraints = constraints
        self._threshold = bottleneck_threshold

    def analyze_all(self) -> list[ConstraintTightness]:
        """Compute tightness for all constraints."""
        results = []
        for cstr in self._constraints:
            tightness = self._compute_tightness(cstr)
            support = self._compute_support_ratio(cstr)
            results.append(ConstraintTightness(
                constraint=cstr,
                tightness=tightness,
                support_ratio=support,
                is_bottleneck=tightness >= self._threshold,
            ))
        return results

    def get_bottlenecks(self) -> list[Constraint]:
        """Return constraints that are likely bottlenecks."""
        results = self.analyze_all()
        return [r.constraint for r in results if r.is_bottleneck]

    def rank_by_tightness(self) -> list[tuple[Constraint, float]]:
        """Rank constraints from tightest to loosest."""
        results = self.analyze_all()
        ranked = [(r.constraint, r.tightness) for r in results]
        ranked.sort(key=lambda x: -x[1])
        return ranked

    def _compute_tightness(self, cstr: Constraint) -> float:
        """Compute tightness as fraction of forbidden tuples."""
        if cstr.arity == 1:
            var = cstr.variables[0]
            if var.domain_size == 0:
                return 1.0
            forbidden = sum(
                1 for v in var.domain if not cstr.is_satisfied({var: v})
            )
            return forbidden / var.domain_size

        if cstr.arity == 2:
            v1, v2 = cstr.variables[0], cstr.variables[1]
            total = v1.domain_size * v2.domain_size
            if total == 0:
                return 1.0
            forbidden = 0
            for val1 in v1.domain:
                for val2 in v2.domain:
                    if not cstr.is_satisfied({v1: val1, v2: val2}):
                        forbidden += 1
            return forbidden / total

        # For higher arity, sample
        return self._sample_tightness(cstr)

    def _sample_tightness(self, cstr: Constraint, samples: int = 1000) -> float:
        """Estimate tightness by random sampling for high-arity constraints."""
        import random
        rng = random.Random(42)
        violations = 0

        for _ in range(samples):
            assignment: dict[Variable, int] = {}
            for var in cstr.variables:
                values = list(var.domain)
                if values:
                    assignment[var] = rng.choice(values)
            if assignment and not cstr.is_satisfied(assignment):
                violations += 1

        return violations / samples if samples > 0 else 0.0

    def _compute_support_ratio(self, cstr: Constraint) -> float:
        """Compute average number of supports per variable value."""
        if cstr.arity != 2:
            return 0.0

        v1, v2 = cstr.variables[0], cstr.variables[1]
        total_support = 0
        total_values = 0

        for val1 in v1.domain:
            supports = sum(
                1 for val2 in v2.domain
                if cstr.is_satisfied({v1: val1, v2: val2})
            )
            total_support += supports
            total_values += 1

        for val2 in v2.domain:
            supports = sum(
                1 for val1 in v1.domain
                if cstr.is_satisfied({v1: val1, v2: val2})
            )
            total_support += supports
            total_values += 1

        return total_support / total_values if total_values > 0 else 0.0
