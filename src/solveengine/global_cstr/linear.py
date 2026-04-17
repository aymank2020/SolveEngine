"""Linear constraints: Sum and Scalar Product.

These constraints enforce linear relationships between variables:
- SumConstraint: sum(vars) == target (or <=, >=)
- ScalarProduct: sum(coeff[i] * var[i]) == target
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class ComparisonOp(Enum):
    EQ = "=="
    LE = "<="
    GE = ">="
    LT = "<"
    GT = ">"
    NE = "!="


class SumConstraint(Constraint):
    """Constraint: sum of variables compared to a target value.

    Example: x + y + z == 15
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        target: int,
        op: ComparisonOp = ComparisonOp.EQ,
        name: str = "",
    ) -> None:
        super().__init__(list(variables), name or f"Sum{op.value}{target}")
        self._target = target
        self._op = op

    @property
    def target(self) -> int:
        return self._target

    @property
    def op(self) -> ComparisonOp:
        return self._op

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        # Only check when all variables are assigned
        for var in self._variables:
            if var not in assignment:
                return True  # Cannot be violated yet

        total = sum(assignment[var] for var in self._variables)
        return self._compare(total)

    def _compare(self, total: int) -> bool:
        if self._op == ComparisonOp.EQ:
            return total == self._target
        elif self._op == ComparisonOp.LE:
            return total <= self._target
        elif self._op == ComparisonOp.GE:
            return total >= self._target
        elif self._op == ComparisonOp.LT:
            return total < self._target
        elif self._op == ComparisonOp.GT:
            return total > self._target
        elif self._op == ComparisonOp.NE:
            return total != self._target
        return False

    def get_supported_values(self, var: Variable, assignment: dict[Variable, int]) -> set[int]:
        """Compute supported values using bounds reasoning.

        For each candidate value of var, check if the remaining variables
        can still satisfy the constraint given their domain bounds.
        """
        others = [v for v in self._variables if v is not var]

        # Check if all others are assigned
        all_others_assigned = all(v in assignment for v in others)

        if all_others_assigned:
            others_sum = sum(assignment[v] for v in others)
            supported = set()
            for val in var.domain.values():
                if self._compare(others_sum + val):
                    supported.add(val)
            return supported

        # Bounds-based filtering
        min_others = 0
        max_others = 0
        for other in others:
            if other in assignment:
                min_others += assignment[other]
                max_others += assignment[other]
            else:
                min_others += other.domain.min_value
                max_others += other.domain.max_value

        supported = set()
        for val in var.domain.values():
            total_min = val + min_others
            total_max = val + max_others
            if self._can_satisfy(total_min, total_max):
                supported.add(val)

        return supported

    def _can_satisfy(self, total_min: int, total_max: int) -> bool:
        """Check if any value in [total_min, total_max] satisfies the comparison."""
        if self._op == ComparisonOp.EQ:
            return total_min <= self._target <= total_max
        elif self._op == ComparisonOp.LE:
            return total_min <= self._target
        elif self._op == ComparisonOp.GE:
            return total_max >= self._target
        elif self._op == ComparisonOp.LT:
            return total_min < self._target
        elif self._op == ComparisonOp.GT:
            return total_max > self._target
        elif self._op == ComparisonOp.NE:
            return not (total_min == total_max == self._target)
        return False


class ScalarProduct(Constraint):
    """Constraint: sum(coeff[i] * var[i]) compared to target.

    Example: 2*x + 3*y - z == 10
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        coefficients: Sequence[int],
        target: int,
        op: ComparisonOp = ComparisonOp.EQ,
        name: str = "",
    ) -> None:
        if len(variables) != len(coefficients):
            raise ValueError("Variables and coefficients must have same length")
        super().__init__(list(variables), name or f"ScalarProduct{op.value}{target}")
        self._coefficients = list(coefficients)
        self._target = target
        self._op = op

    @property
    def coefficients(self) -> list[int]:
        return self._coefficients

    @property
    def target(self) -> int:
        return self._target

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        for var in self._variables:
            if var not in assignment:
                return True

        total = sum(
            c * assignment[v] for c, v in zip(self._coefficients, self._variables)
        )
        return self._compare(total)

    def _compare(self, total: int) -> bool:
        if self._op == ComparisonOp.EQ:
            return total == self._target
        elif self._op == ComparisonOp.LE:
            return total <= self._target
        elif self._op == ComparisonOp.GE:
            return total >= self._target
        elif self._op == ComparisonOp.LT:
            return total < self._target
        elif self._op == ComparisonOp.GT:
            return total > self._target
        elif self._op == ComparisonOp.NE:
            return total != self._target
        return False

    def get_supported_values(self, var: Variable, assignment: dict[Variable, int]) -> set[int]:
        var_idx = list(self._variables).index(var)
        var_coeff = self._coefficients[var_idx]

        others_min = 0
        others_max = 0
        for i, other in enumerate(self._variables):
            if i == var_idx:
                continue
            coeff = self._coefficients[i]
            if other in assignment:
                others_min += coeff * assignment[other]
                others_max += coeff * assignment[other]
            else:
                if coeff >= 0:
                    others_min += coeff * other.domain.min_value
                    others_max += coeff * other.domain.max_value
                else:
                    others_min += coeff * other.domain.max_value
                    others_max += coeff * other.domain.min_value

        supported = set()
        for val in var.domain.values():
            contribution = var_coeff * val
            total_min = contribution + others_min
            total_max = contribution + others_max
            if self._can_satisfy(total_min, total_max):
                supported.add(val)

        return supported

    def _can_satisfy(self, total_min: int, total_max: int) -> bool:
        if self._op == ComparisonOp.EQ:
            return total_min <= self._target <= total_max
        elif self._op == ComparisonOp.LE:
            return total_min <= self._target
        elif self._op == ComparisonOp.GE:
            return total_max >= self._target
        elif self._op == ComparisonOp.LT:
            return total_min < self._target
        elif self._op == ComparisonOp.GT:
            return total_max > self._target
        elif self._op == ComparisonOp.NE:
            return not (total_min == total_max == self._target)
        return False
