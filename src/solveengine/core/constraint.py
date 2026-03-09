"""Constraint definitions for CSP problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

from solveengine.core.variable import Variable


class Constraint(ABC):
    """Base class for all constraints.

    A constraint defines a relation over a set of variables that must
    be satisfied in any valid solution.
    """

    __slots__ = ("_variables", "_name", "_propagation_count")

    def __init__(self, variables: Sequence[Variable], name: str = "") -> None:
        self._variables = tuple(variables)
        self._name = name or self._default_name()
        self._propagation_count: int = 0

    def _default_name(self) -> str:
        var_names = ", ".join(v.name for v in self._variables)
        return f"{self.__class__.__name__}({var_names})"

    @property
    def variables(self) -> tuple[Variable, ...]:
        return self._variables

    @property
    def arity(self) -> int:
        return len(self._variables)

    @property
    def name(self) -> str:
        return self._name

    @property
    def propagation_count(self) -> int:
        """Number of times this constraint triggered propagation."""
        return self._propagation_count

    def record_propagation(self) -> None:
        self._propagation_count += 1

    @abstractmethod
    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        """Check if the constraint is satisfied under the given assignment.

        Only checks variables that are assigned. Returns True if the
        constraint cannot be violated by the current partial assignment.
        """
        ...

    @abstractmethod
    def get_supported_values(self, var: Variable, assignment: dict[Variable, int]) -> set[int]:
        """Return the set of values for var that are consistent with the assignment.

        Used by arc consistency algorithms to prune domains.
        """
        ...

    def involves(self, var: Variable) -> bool:
        """Check if this constraint involves the given variable."""
        return var in self._variables

    def other_variables(self, var: Variable) -> tuple[Variable, ...]:
        """Return all variables in this constraint except var."""
        return tuple(v for v in self._variables if v is not var)

    def __repr__(self) -> str:
        return self._name


class BinaryConstraint(Constraint):
    """A constraint between exactly two variables defined by a predicate."""

    __slots__ = ("_predicate", "_var1", "_var2")

    def __init__(
        self,
        var1: Variable,
        var2: Variable,
        predicate: Callable[[int, int], bool],
        name: str = "",
    ) -> None:
        super().__init__([var1, var2], name)
        self._predicate = predicate
        self._var1 = var1
        self._var2 = var2

    @property
    def var1(self) -> Variable:
        return self._var1

    @property
    def var2(self) -> Variable:
        return self._var2

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        if self._var1 not in assignment or self._var2 not in assignment:
            return True  # Cannot be violated yet
        return self._predicate(assignment[self._var1], assignment[self._var2])

    def get_supported_values(self, var: Variable, assignment: dict[Variable, int]) -> set[int]:
        """Get values of var consistent with the other variable's domain/assignment."""
        if var is self._var1:
            other = self._var2
            check = lambda v, o: self._predicate(v, o)
        elif var is self._var2:
            other = self._var1
            check = lambda v, o: self._predicate(o, v)
        else:
            raise ValueError(f"Variable {var.name} not in constraint {self._name}")

        supported = set()
        if other in assignment:
            other_vals = {assignment[other]}
        else:
            other_vals = other.domain.values()

        for val in var.domain.values():
            for oval in other_vals:
                if check(val, oval):
                    supported.add(val)
                    break

        return supported


class UnaryConstraint(Constraint):
    """A constraint on a single variable defined by a predicate."""

    __slots__ = ("_predicate", "_var")

    def __init__(
        self,
        var: Variable,
        predicate: Callable[[int], bool],
        name: str = "",
    ) -> None:
        super().__init__([var], name)
        self._predicate = predicate
        self._var = var

    @property
    def var(self) -> Variable:
        return self._var

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        if self._var not in assignment:
            return True
        return self._predicate(assignment[self._var])

    def get_supported_values(self, var: Variable, assignment: dict[Variable, int]) -> set[int]:
        if var is not self._var:
            raise ValueError(f"Variable {var.name} not in constraint {self._name}")
        return {v for v in var.domain.values() if self._predicate(v)}


class TableConstraint(Constraint):
    """A constraint defined by an explicit set of allowed tuples."""

    __slots__ = ("_allowed_tuples",)

    def __init__(
        self,
        variables: Sequence[Variable],
        allowed_tuples: set[tuple[int, ...]],
        name: str = "",
    ) -> None:
        super().__init__(variables, name)
        self._allowed_tuples = frozenset(allowed_tuples)

    @property
    def allowed_tuples(self) -> frozenset[tuple[int, ...]]:
        return self._allowed_tuples

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        values = []
        for var in self._variables:
            if var not in assignment:
                return True  # Partial assignment, cannot be violated
            values.append(assignment[var])
        return tuple(values) in self._allowed_tuples

    def get_supported_values(self, var: Variable, assignment: dict[Variable, int]) -> set[int]:
        var_idx = self._variables.index(var)
        supported = set()

        for tup in self._allowed_tuples:
            consistent = True
            for i, v in enumerate(self._variables):
                if i == var_idx:
                    continue
                if v in assignment and assignment[v] != tup[i]:
                    consistent = False
                    break
                if v not in assignment and tup[i] not in v.domain:
                    consistent = False
                    break
            if consistent and tup[var_idx] in var.domain:
                supported.add(tup[var_idx])

        return supported
