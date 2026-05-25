"""Bounds validation for constraint problems.

Validates that domain bounds are consistent with constraints and
detects infeasibility early through bounds reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.global_cstr.linear import SumConstraint, ComparisonOp


@dataclass
class BoundsInfo:
    """Bounds information for a variable."""

    variable: Variable
    current_min: int
    current_max: int
    implied_min: int
    implied_max: int
    can_tighten: bool

    @property
    def tightening_potential(self) -> int:
        """How many values could be pruned by bounds tightening."""
        lower_prune = max(0, self.implied_min - self.current_min)
        upper_prune = max(0, self.current_max - self.implied_max)
        return lower_prune + upper_prune


class BoundsValidator:
    """Validates and tightens variable bounds using constraint reasoning.

    For linear constraints (Sum, ScalarProduct), computes implied bounds
    on each variable and checks if the current domain is consistent.
    """

    def __init__(self, variables: list[Variable], constraints: list[Constraint]) -> None:
        self._variables = variables
        self._constraints = constraints

    def compute_bounds(self) -> list[BoundsInfo]:
        """Compute implied bounds for all variables."""
        results = []
        for var in self._variables:
            if var.domain.is_empty:
                results.append(BoundsInfo(
                    variable=var,
                    current_min=0, current_max=0,
                    implied_min=0, implied_max=0,
                    can_tighten=False,
                ))
                continue

            current_min = var.domain.min_value
            current_max = var.domain.max_value
            implied_min = current_min
            implied_max = current_max

            for cstr in self._constraints:
                if not cstr.involves(var):
                    continue
                if isinstance(cstr, SumConstraint):
                    lo, hi = self._implied_bounds_from_sum(var, cstr)
                    implied_min = max(implied_min, lo)
                    implied_max = min(implied_max, hi)

            can_tighten = implied_min > current_min or implied_max < current_max

            results.append(BoundsInfo(
                variable=var,
                current_min=current_min,
                current_max=current_max,
                implied_min=implied_min,
                implied_max=implied_max,
                can_tighten=can_tighten,
            ))

        return results

    def tighten_all(self) -> dict[Variable, list[int]]:
        """Apply bounds tightening to all variables.

        Returns map of variable -> pruned values.
        """
        pruned: dict[Variable, list[int]] = {}
        bounds_info = self.compute_bounds()

        for info in bounds_info:
            if not info.can_tighten:
                continue
            var = info.variable
            removed = []
            for val in list(var.domain):
                if val < info.implied_min or val > info.implied_max:
                    if var.domain.remove(val):
                        removed.append(val)
            if removed:
                pruned[var] = removed

        return pruned

    def is_feasible(self) -> bool:
        """Quick feasibility check using bounds reasoning.

        Returns False if bounds reasoning proves infeasibility.
        """
        for var in self._variables:
            if var.domain.is_empty:
                return False

        bounds_info = self.compute_bounds()
        for info in bounds_info:
            if info.implied_min > info.implied_max:
                return False
            if info.implied_min > info.current_max:
                return False
            if info.implied_max < info.current_min:
                return False

        return True

    def _implied_bounds_from_sum(
        self, var: Variable, cstr: SumConstraint
    ) -> tuple[int, int]:
        """Compute implied bounds on var from a sum constraint.

        For sum(vars) == target:
          var >= target - sum(max of others)
          var <= target - sum(min of others)
        """
        others = [v for v in cstr.variables if v is not var]
        others_min = sum(v.domain.min_value for v in others if not v.domain.is_empty)
        others_max = sum(v.domain.max_value for v in others if not v.domain.is_empty)

        target = cstr.target

        if cstr.op == ComparisonOp.EQ:
            implied_min = target - others_max
            implied_max = target - others_min
        elif cstr.op == ComparisonOp.LE:
            implied_min = var.domain.min_value  # No lower bound from <=
            implied_max = target - others_min
        elif cstr.op == ComparisonOp.GE:
            implied_min = target - others_max
            implied_max = var.domain.max_value  # No upper bound from >=
        else:
            implied_min = var.domain.min_value
            implied_max = var.domain.max_value

        return implied_min, implied_max
