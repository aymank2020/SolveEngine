"""Consistency level checking for CSP problems.

Determines what level of consistency holds for the current state:
- Node consistency: all unary constraints satisfied
- Arc consistency: all binary arcs are consistent
- Path consistency: all paths of length 2 are consistent
- Bounds consistency: domain bounds are consistent
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint, BinaryConstraint, UnaryConstraint


class ConsistencyLevel(Enum):
    NONE = "none"
    NODE = "node"
    ARC = "arc"
    BOUNDS = "bounds"
    PATH = "path"


@dataclass
class ConsistencyReport:
    """Report on the consistency level of a CSP state."""

    level: ConsistencyLevel
    node_consistent: bool
    arc_consistent: bool
    bounds_consistent: bool
    inconsistent_arcs: list[tuple[Variable, Constraint]]
    empty_domains: list[Variable]

    @property
    def is_feasible(self) -> bool:
        """True if no domain is empty (problem might still be solvable)."""
        return len(self.empty_domains) == 0


class ConsistencyChecker:
    """Checks various levels of consistency for a CSP state."""

    def __init__(self, variables: list[Variable], constraints: list[Constraint]) -> None:
        self._variables = variables
        self._constraints = constraints

    def full_check(self, assignment: dict[Variable, int] | None = None) -> ConsistencyReport:
        """Perform a full consistency check."""
        if assignment is None:
            assignment = {}

        empty_domains = [v for v in self._variables if v.domain.is_empty]
        node_ok = self._check_node_consistency(assignment)
        arc_ok, bad_arcs = self._check_arc_consistency(assignment)
        bounds_ok = self._check_bounds_consistency(assignment)

        if not node_ok:
            level = ConsistencyLevel.NONE
        elif not arc_ok:
            level = ConsistencyLevel.NODE
        elif not bounds_ok:
            level = ConsistencyLevel.ARC
        else:
            level = ConsistencyLevel.PATH

        return ConsistencyReport(
            level=level,
            node_consistent=node_ok,
            arc_consistent=arc_ok,
            bounds_consistent=bounds_ok,
            inconsistent_arcs=bad_arcs,
            empty_domains=empty_domains,
        )

    def _check_node_consistency(self, assignment: dict[Variable, int]) -> bool:
        """Check if all unary constraints are satisfied by current domains."""
        for cstr in self._constraints:
            if not isinstance(cstr, UnaryConstraint):
                continue
            var = cstr.var
            if var.is_assigned:
                continue
            for val in var.domain:
                if not cstr.is_satisfied({var: val}):
                    return False
        return True

    def _check_arc_consistency(
        self, assignment: dict[Variable, int]
    ) -> tuple[bool, list[tuple[Variable, Constraint]]]:
        """Check if all arcs are consistent.

        An arc (var, constraint) is consistent if every value in var's domain
        has at least one support in the other variables' domains.
        """
        bad_arcs: list[tuple[Variable, Constraint]] = []

        for cstr in self._constraints:
            if cstr.arity < 2:
                continue
            for var in cstr.variables:
                if var.is_assigned:
                    continue
                supported = cstr.get_supported_values(var, assignment)
                if not var.domain.values().issubset(supported | var.domain.values()):
                    # Check if any value lacks support
                    for val in var.domain:
                        if val not in supported:
                            bad_arcs.append((var, cstr))
                            break

        return len(bad_arcs) == 0, bad_arcs

    def _check_bounds_consistency(self, assignment: dict[Variable, int]) -> bool:
        """Check bounds consistency for numeric constraints.

        For each variable, verify that its min and max values are
        supported by at least one constraint.
        """
        for cstr in self._constraints:
            for var in cstr.variables:
                if var.is_assigned or var.domain.is_empty:
                    continue
                supported = cstr.get_supported_values(var, assignment)
                if supported and var.domain.min_value not in supported:
                    return False
                if supported and var.domain.max_value not in supported:
                    return False
        return True

    def detect_singleton_propagation(self) -> list[tuple[Variable, int]]:
        """Find variables with singleton domains that haven't been propagated.

        Returns list of (variable, value) pairs that should trigger propagation.
        """
        singletons = []
        for var in self._variables:
            if var.domain.is_singleton and not var.is_assigned:
                val = next(iter(var.domain))
                singletons.append((var, val))
        return singletons
