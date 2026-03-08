"""Variable representation for constraint satisfaction problems."""

from __future__ import annotations

from typing import Iterable

from solveengine.core.domain import Domain


class Variable:
    """A CSP variable with a name and finite domain.

    Variables track their current domain state and assignment status.
    They also maintain a weight counter used by the dom/wdeg heuristic.
    """

    __slots__ = ("_name", "_domain", "_assigned_value", "_weight", "_index")

    _next_index: int = 0

    def __init__(self, name: str, values: Iterable[int]) -> None:
        self._name = name
        self._domain = Domain(values)
        self._assigned_value: int | None = None
        self._weight: float = 1.0
        self._index = Variable._next_index
        Variable._next_index += 1

    @property
    def name(self) -> str:
        return self._name

    @property
    def index(self) -> int:
        """Unique integer index for this variable (creation order)."""
        return self._index

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def domain_size(self) -> int:
        return self._domain.size

    @property
    def is_assigned(self) -> bool:
        return self._assigned_value is not None

    @property
    def assigned_value(self) -> int | None:
        return self._assigned_value

    @property
    def weight(self) -> float:
        """Weight for dom/wdeg heuristic. Incremented on constraint failures."""
        return self._weight

    def increment_weight(self, amount: float = 1.0) -> None:
        """Increase weight (called when a constraint involving this var fails)."""
        self._weight += amount

    def assign(self, value: int) -> list[int]:
        """Assign a value to this variable.

        Restricts the domain to the single assigned value.
        Returns the list of removed domain values.
        """
        if self._assigned_value is not None:
            raise RuntimeError(f"Variable {self._name} already assigned to {self._assigned_value}")
        if not self._domain.contains(value):
            raise ValueError(f"Value {value} not in domain of {self._name}")
        self._assigned_value = value
        return self._domain.assign(value)

    def unassign(self) -> None:
        """Remove the current assignment (domain must be restored separately)."""
        self._assigned_value = None

    def mark_generation(self) -> int:
        """Mark a new generation on the domain for backtracking."""
        return self._domain.mark_generation()

    def restore_to(self, generation: int) -> None:
        """Restore domain to a previous generation and clear assignment if needed."""
        self._domain.restore_to(generation)
        if self._assigned_value is not None and not self._domain.contains(self._assigned_value):
            self._assigned_value = None

    def __repr__(self) -> str:
        if self._assigned_value is not None:
            return f"Variable({self._name}={self._assigned_value})"
        return f"Variable({self._name}, dom_size={self.domain_size})"

    def __hash__(self) -> int:
        return hash(self._index)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Variable):
            return NotImplemented
        return self._index == other._index
