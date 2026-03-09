"""Variable and constraint registry for problem management.

Provides a centralized registry that tracks all variables and constraints
in a problem instance. Supports efficient lookup by name, index, and
constraint type. Used by the solver to quickly find relevant constraints
during propagation.
"""

from __future__ import annotations

from typing import Iterator, Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint, BinaryConstraint, UnaryConstraint


class VariableRegistry:
    """Registry for managing CSP variables.

    Provides O(1) lookup by name and index, and maintains ordering
    for deterministic solver behavior.
    """

    def __init__(self) -> None:
        self._variables: list[Variable] = []
        self._by_name: dict[str, Variable] = {}
        self._by_index: dict[int, Variable] = {}

    @property
    def size(self) -> int:
        return len(self._variables)

    @property
    def variables(self) -> list[Variable]:
        return list(self._variables)

    def register(self, var: Variable) -> None:
        """Register a variable. Raises if name already exists."""
        if var.name in self._by_name:
            raise ValueError(f"Variable '{var.name}' already registered")
        self._variables.append(var)
        self._by_name[var.name] = var
        self._by_index[var.index] = var

    def register_many(self, variables: Sequence[Variable]) -> None:
        """Register multiple variables at once."""
        for var in variables:
            self.register(var)

    def get_by_name(self, name: str) -> Variable | None:
        """Look up a variable by name."""
        return self._by_name.get(name)

    def get_by_index(self, index: int) -> Variable | None:
        """Look up a variable by index."""
        return self._by_index.get(index)

    def get_unassigned(self) -> list[Variable]:
        """Return all unassigned variables."""
        return [v for v in self._variables if not v.is_assigned]

    def get_assigned(self) -> list[Variable]:
        """Return all assigned variables."""
        return [v for v in self._variables if v.is_assigned]

    def total_domain_size(self) -> int:
        """Sum of all variable domain sizes."""
        return sum(v.domain_size for v in self._variables)

    def min_domain_variable(self) -> Variable | None:
        """Return the unassigned variable with smallest domain."""
        unassigned = self.get_unassigned()
        if not unassigned:
            return None
        return min(unassigned, key=lambda v: v.domain_size)

    def __contains__(self, var: Variable) -> bool:
        return var.index in self._by_index

    def __iter__(self) -> Iterator[Variable]:
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)


class ConstraintRegistry:
    """Registry for managing CSP constraints.

    Provides efficient lookup of constraints by variable, type, and arity.
    Maintains adjacency information for the constraint graph.
    """

    def __init__(self) -> None:
        self._constraints: list[Constraint] = []
        self._by_variable: dict[int, list[Constraint]] = {}
        self._binary: list[BinaryConstraint] = []
        self._unary: list[UnaryConstraint] = []
        self._global: list[Constraint] = []

    @property
    def size(self) -> int:
        return len(self._constraints)

    @property
    def constraints(self) -> list[Constraint]:
        return list(self._constraints)

    @property
    def binary_constraints(self) -> list[BinaryConstraint]:
        return list(self._binary)

    @property
    def unary_constraints(self) -> list[UnaryConstraint]:
        return list(self._unary)

    @property
    def global_constraints(self) -> list[Constraint]:
        return list(self._global)

    def register(self, constraint: Constraint) -> None:
        """Register a constraint and update indices."""
        self._constraints.append(constraint)

        # Index by variable
        for var in constraint.variables:
            if var.index not in self._by_variable:
                self._by_variable[var.index] = []
            self._by_variable[var.index].append(constraint)

        # Categorize by type
        if isinstance(constraint, BinaryConstraint):
            self._binary.append(constraint)
        elif isinstance(constraint, UnaryConstraint):
            self._unary.append(constraint)
        elif constraint.arity > 2:
            self._global.append(constraint)

    def register_many(self, constraints: Sequence[Constraint]) -> None:
        """Register multiple constraints at once."""
        for cstr in constraints:
            self.register(cstr)

    def get_for_variable(self, var: Variable) -> list[Constraint]:
        """Get all constraints involving a variable."""
        return self._by_variable.get(var.index, [])

    def get_between(self, var1: Variable, var2: Variable) -> list[Constraint]:
        """Get all constraints between two specific variables."""
        cstrs1 = set(id(c) for c in self._by_variable.get(var1.index, []))
        result = []
        for cstr in self._by_variable.get(var2.index, []):
            if id(cstr) in cstrs1:
                result.append(cstr)
        return result

    def get_neighbors(self, var: Variable) -> set[Variable]:
        """Get all variables that share a constraint with var."""
        neighbors: set[Variable] = set()
        for cstr in self.get_for_variable(var):
            for other in cstr.variables:
                if other is not var:
                    neighbors.add(other)
        return neighbors

    def constraint_degree(self, var: Variable) -> int:
        """Number of constraints involving var."""
        return len(self._by_variable.get(var.index, []))

    def __contains__(self, constraint: Constraint) -> bool:
        return constraint in self._constraints

    def __iter__(self) -> Iterator[Constraint]:
        return iter(self._constraints)

    def __len__(self) -> int:
        return len(self._constraints)
