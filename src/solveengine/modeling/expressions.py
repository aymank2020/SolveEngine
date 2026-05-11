"""Expression building for constraint modeling.

Provides a functional interface for building constraints from
arithmetic expressions over variables.
"""

from __future__ import annotations

from typing import Sequence

from solveengine.core.variable import Variable


class Expr:
    """An arithmetic expression over CSP variables.

    Supports building constraint expressions like:
        Expr(x) + Expr(y) == 10
        Expr(x) * 2 + Expr(y) <= 20
    """

    def __init__(self, var: Variable | None = None, constant: int = 0, terms: list[tuple[int, Variable]] | None = None) -> None:
        if terms is not None:
            self._terms = list(terms)
            self._constant = constant
        elif var is not None:
            self._terms = [(1, var)]
            self._constant = constant
        else:
            self._terms = []
            self._constant = constant

    @property
    def terms(self) -> list[tuple[int, Variable]]:
        """List of (coefficient, variable) pairs."""
        return self._terms

    @property
    def constant(self) -> int:
        return self._constant

    @property
    def variables(self) -> list[Variable]:
        return [var for _, var in self._terms]

    def __add__(self, other: Expr | int) -> Expr:
        if isinstance(other, int):
            return Expr(terms=list(self._terms), constant=self._constant + other)
        return Expr(terms=self._terms + other._terms, constant=self._constant + other._constant)

    def __radd__(self, other: int) -> Expr:
        return Expr(terms=list(self._terms), constant=self._constant + other)

    def __sub__(self, other: Expr | int) -> Expr:
        if isinstance(other, int):
            return Expr(terms=list(self._terms), constant=self._constant - other)
        neg_terms = [(-c, v) for c, v in other._terms]
        return Expr(terms=self._terms + neg_terms, constant=self._constant - other._constant)

    def __mul__(self, scalar: int) -> Expr:
        new_terms = [(c * scalar, v) for c, v in self._terms]
        return Expr(terms=new_terms, constant=self._constant * scalar)

    def __rmul__(self, scalar: int) -> Expr:
        return self.__mul__(scalar)

    def evaluate(self, assignment: dict[Variable, int]) -> int:
        """Evaluate the expression under the given assignment."""
        total = self._constant
        for coeff, var in self._terms:
            if var not in assignment:
                raise ValueError(f"Variable {var.name} not in assignment")
            total += coeff * assignment[var]
        return total

    def __repr__(self) -> str:
        parts = []
        for coeff, var in self._terms:
            if coeff == 1:
                parts.append(var.name)
            elif coeff == -1:
                parts.append(f"-{var.name}")
            else:
                parts.append(f"{coeff}*{var.name}")
        if self._constant != 0:
            parts.append(str(self._constant))
        return " + ".join(parts) if parts else "0"


def Sum(variables: Sequence[Variable]) -> Expr:
    """Create an expression representing the sum of variables."""
    terms = [(1, var) for var in variables]
    return Expr(terms=terms, constant=0)


def Count(variables: Sequence[Variable], value: int) -> Expr:
    """Create a pseudo-expression for counting occurrences of a value.

    Note: This is a modeling helper. The actual counting is done
    during constraint evaluation, not via linear arithmetic.
    """
    # This creates a sum of indicator variables (conceptual)
    # In practice, this would be handled by a specialized constraint
    terms = [(1, var) for var in variables]
    return Expr(terms=terms, constant=0)
