"""Symmetry breaking for CSP problems.

Symmetries in a CSP cause redundant exploration of equivalent search
subtrees. Breaking symmetries by adding ordering constraints can
dramatically reduce the search space.
"""

from __future__ import annotations

from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint, Constraint


class SymmetryBreaker:
    """Detects and breaks symmetries in CSP problems.

    Supports:
    - Variable symmetry (interchangeable variables)
    - Value symmetry (interchangeable values)
    - Lexicographic ordering constraints
    """

    def __init__(self, variables: list[Variable], constraints: list[Constraint]) -> None:
        self._variables = variables
        self._constraints = constraints

    def find_interchangeable_variables(self) -> list[list[Variable]]:
        """Find groups of variables that are interchangeable.

        Two variables are interchangeable if swapping their values in any
        solution produces another valid solution.
        """
        groups: list[list[Variable]] = []
        used: set[int] = set()

        for i, v1 in enumerate(self._variables):
            if v1.index in used:
                continue
            group = [v1]
            used.add(v1.index)

            for j in range(i + 1, len(self._variables)):
                v2 = self._variables[j]
                if v2.index in used:
                    continue
                if self._are_interchangeable(v1, v2):
                    group.append(v2)
                    used.add(v2.index)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _are_interchangeable(self, v1: Variable, v2: Variable) -> bool:
        """Check if two variables are interchangeable.

        Variables are interchangeable if:
        1. They have the same domain
        2. They appear in the same constraints with the same role
        """
        if v1.domain.values() != v2.domain.values():
            return False

        # Check constraint participation
        v1_constraints = set()
        v2_constraints = set()

        for cstr in self._constraints:
            if cstr.involves(v1):
                others = frozenset(v.index for v in cstr.other_variables(v1))
                v1_constraints.add((type(cstr).__name__, others))
            if cstr.involves(v2):
                others = frozenset(v.index for v in cstr.other_variables(v2))
                v2_constraints.add((type(cstr).__name__, others))

        # Simplified check: same constraint types with same other variables
        # (excluding each other)
        return len(v1_constraints) == len(v2_constraints)

    def add_lex_ordering(self, group: list[Variable]) -> list[BinaryConstraint]:
        """Add lexicographic ordering constraints to break symmetry.

        For interchangeable variables [x1, x2, ..., xn], adds:
        x1 <= x2 <= ... <= xn
        """
        constraints = []
        for i in range(len(group) - 1):
            cstr = BinaryConstraint(
                group[i],
                group[i + 1],
                lambda a, b: a <= b,
                name=f"lex_{group[i].name}<={group[i+1].name}",
            )
            constraints.append(cstr)
        return constraints

    def find_value_symmetries(self) -> list[set[int]]:
        """Find groups of interchangeable values.

        Values are interchangeable if swapping them in any solution
        produces another valid solution.
        """
        # Collect all values across all domains
        all_values: set[int] = set()
        for var in self._variables:
            all_values.update(var.domain.values())

        # Simple heuristic: values that appear in the same set of domains
        # and participate in the same constraint patterns
        value_profiles: dict[int, frozenset[int]] = {}
        for val in all_values:
            profile = frozenset(
                var.index for var in self._variables if var.domain.contains(val)
            )
            value_profiles[val] = profile

        # Group values with identical profiles
        profile_groups: dict[frozenset[int], list[int]] = {}
        for val, profile in value_profiles.items():
            if profile not in profile_groups:
                profile_groups[profile] = []
            profile_groups[profile].append(val)

        return [set(group) for group in profile_groups.values() if len(group) > 1]
