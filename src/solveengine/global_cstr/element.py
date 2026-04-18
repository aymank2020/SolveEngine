"""Element global constraint.

Element(index, array, value) ensures that array[index] == value,
where index and value are CSP variables and array is a list of
integer constants or variables.
"""

from __future__ import annotations

from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class ElementConstraint(Constraint):
    """Element constraint: array[index_var] == value_var.

    The index variable selects an element from the array, and the
    value variable must equal that element.

    Args:
        index_var: Variable whose value is used as the array index (0-based).
        array: List of integer constants.
        value_var: Variable that must equal array[index_var].
    """

    def __init__(
        self,
        index_var: Variable,
        array: Sequence[int],
        value_var: Variable,
        name: str = "",
    ) -> None:
        super().__init__([index_var, value_var], name or "Element")
        self._index_var = index_var
        self._value_var = value_var
        self._array = list(array)

    @property
    def index_var(self) -> Variable:
        return self._index_var

    @property
    def value_var(self) -> Variable:
        return self._value_var

    @property
    def array(self) -> list[int]:
        return self._array

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        if self._index_var not in assignment or self._value_var not in assignment:
            return True
        idx = assignment[self._index_var]
        val = assignment[self._value_var]
        if idx < 0 or idx >= len(self._array):
            return False
        return self._array[idx] == val

    def get_supported_values(self, var: Variable, assignment: dict[Variable, int]) -> set[int]:
        if var is self._index_var:
            return self._get_supported_indices(assignment)
        elif var is self._value_var:
            return self._get_supported_values(assignment)
        raise ValueError(f"Variable {var.name} not in Element constraint")

    def _get_supported_indices(self, assignment: dict[Variable, int]) -> set[int]:
        """Get index values that are consistent."""
        supported = set()
        for idx in self._index_var.domain.values():
            if idx < 0 or idx >= len(self._array):
                continue
            if self._value_var in assignment:
                if self._array[idx] == assignment[self._value_var]:
                    supported.add(idx)
            else:
                if self._array[idx] in self._value_var.domain:
                    supported.add(idx)
        return supported

    def _get_supported_values(self, assignment: dict[Variable, int]) -> set[int]:
        """Get value-var values that are consistent."""
        supported = set()
        if self._index_var in assignment:
            idx = assignment[self._index_var]
            if 0 <= idx < len(self._array):
                val = self._array[idx]
                if val in self._value_var.domain:
                    supported.add(val)
        else:
            for idx in self._index_var.domain.values():
                if 0 <= idx < len(self._array):
                    val = self._array[idx]
                    if val in self._value_var.domain:
                        supported.add(val)
        return supported

    def propagate_index(self) -> dict[Variable, list[int]]:
        """Propagate: remove index values pointing outside array bounds."""
        pruned: dict[Variable, list[int]] = {}
        removed = []
        for idx in list(self._index_var.domain):
            if idx < 0 or idx >= len(self._array):
                if self._index_var.domain.remove(idx):
                    removed.append(idx)
        if removed:
            pruned[self._index_var] = removed
        return pruned
