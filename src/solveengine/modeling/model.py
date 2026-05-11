"""High-level model for defining CSP problems.

Provides a user-friendly API for creating variables, adding constraints,
and solving problems without directly manipulating low-level primitives.
"""

from __future__ import annotations

from typing import Callable, Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint, UnaryConstraint, Constraint
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.global_cstr.linear import SumConstraint, ComparisonOp
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.heuristics.variable_ordering import VariableSelector, DomWdegSelector
from solveengine.heuristics.value_ordering import ValueOrderer, LCVOrderer


class Model:
    """High-level CSP model.

    Example usage:
        model = Model()
        x = model.int_var("x", 1, 10)
        y = model.int_var("y", 1, 10)
        model.add_not_equal(x, y)
        model.add_sum_eq([x, y], 15)
        solution = model.solve()
    """

    def __init__(self) -> None:
        self._variables: list[Variable] = []
        self._constraints: list[Constraint] = []
        self._var_selector: VariableSelector = DomWdegSelector()
        self._val_orderer: ValueOrderer = LCVOrderer()

    @property
    def num_variables(self) -> int:
        return len(self._variables)

    @property
    def num_constraints(self) -> int:
        return len(self._constraints)

    def int_var(self, name: str, lower: int, upper: int) -> Variable:
        """Create an integer variable with domain [lower, upper]."""
        var = Variable(name, range(lower, upper + 1))
        self._variables.append(var)
        return var

    def int_var_values(self, name: str, values: Sequence[int]) -> Variable:
        """Create an integer variable with explicit domain values."""
        var = Variable(name, values)
        self._variables.append(var)
        return var

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a pre-built constraint to the model."""
        self._constraints.append(constraint)

    def add_binary(
        self, var1: Variable, var2: Variable, predicate: Callable[[int, int], bool], name: str = ""
    ) -> None:
        """Add a binary constraint between two variables."""
        cstr = BinaryConstraint(var1, var2, predicate, name)
        self._constraints.append(cstr)

    def add_unary(self, var: Variable, predicate: Callable[[int], bool], name: str = "") -> None:
        """Add a unary constraint on a single variable."""
        cstr = UnaryConstraint(var, predicate, name)
        self._constraints.append(cstr)

    def add_not_equal(self, var1: Variable, var2: Variable) -> None:
        """Add a not-equal constraint: var1 != var2."""
        self.add_binary(var1, var2, lambda a, b: a != b, f"{var1.name}!={var2.name}")

    def add_less_than(self, var1: Variable, var2: Variable) -> None:
        """Add a less-than constraint: var1 < var2."""
        self.add_binary(var1, var2, lambda a, b: a < b, f"{var1.name}<{var2.name}")

    def add_all_different(self, *variables: Variable) -> None:
        """Add an AllDifferent constraint over the given variables."""
        cstr = AllDifferent(*variables)
        self._constraints.append(cstr)

    def add_sum_eq(self, variables: Sequence[Variable], target: int) -> None:
        """Add constraint: sum(variables) == target."""
        cstr = SumConstraint(variables, target, ComparisonOp.EQ)
        self._constraints.append(cstr)

    def add_sum_le(self, variables: Sequence[Variable], target: int) -> None:
        """Add constraint: sum(variables) <= target."""
        cstr = SumConstraint(variables, target, ComparisonOp.LE)
        self._constraints.append(cstr)

    def add_sum_ge(self, variables: Sequence[Variable], target: int) -> None:
        """Add constraint: sum(variables) >= target."""
        cstr = SumConstraint(variables, target, ComparisonOp.GE)
        self._constraints.append(cstr)

    def set_var_selector(self, selector: VariableSelector) -> None:
        """Set the variable ordering heuristic."""
        self._var_selector = selector

    def set_val_orderer(self, orderer: ValueOrderer) -> None:
        """Set the value ordering heuristic."""
        self._val_orderer = orderer

    def solve(self) -> dict[str, int] | None:
        """Find a solution to the model.

        Returns a dict mapping variable names to values, or None if unsatisfiable.
        """
        solver = BacktrackSolver(
            self._variables,
            self._constraints,
            var_selector=self._var_selector,
            val_orderer=self._val_orderer,
            use_forward_check=True,
        )
        result = solver.solve()
        if result is None:
            return None
        return {var.name: val for var, val in result.items()}

    def solve_all(self) -> list[dict[str, int]]:
        """Find all solutions to the model."""
        solver = BacktrackSolver(
            self._variables,
            self._constraints,
            var_selector=self._var_selector,
            val_orderer=self._val_orderer,
            use_forward_check=True,
        )
        results = solver.solve_all()
        return [{var.name: val for var, val in r.items()} for r in results]
