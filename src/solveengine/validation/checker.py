"""Solution validation and verification.

Provides comprehensive checking that a proposed solution actually satisfies
all constraints. Useful for verifying solver correctness and validating
externally-provided solutions.
"""

from __future__ import annotations

from dataclasses import dataclass

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


@dataclass
class ValidationResult:
    """Result of solution validation."""

    is_valid: bool
    violations: list[tuple[Constraint, str]]
    unassigned_variables: list[Variable]
    out_of_domain: list[tuple[Variable, int]]

    @property
    def num_violations(self) -> int:
        return len(self.violations)

    @property
    def is_complete(self) -> bool:
        return len(self.unassigned_variables) == 0

    def summary(self) -> str:
        if self.is_valid:
            return "Valid solution: all constraints satisfied"
        lines = [f"Invalid solution: {self.num_violations} violation(s)"]
        for cstr, msg in self.violations:
            lines.append(f"  - {cstr.name}: {msg}")
        if self.unassigned_variables:
            names = [v.name for v in self.unassigned_variables]
            lines.append(f"  Unassigned: {names}")
        if self.out_of_domain:
            for var, val in self.out_of_domain:
                lines.append(f"  Out of domain: {var.name}={val}")
        return "\n".join(lines)


class SolutionChecker:
    """Validates solutions against problem constraints.

    Checks:
    1. All variables are assigned
    2. All assigned values are within variable domains
    3. All constraints are satisfied
    """

    def __init__(self, variables: list[Variable], constraints: list[Constraint]) -> None:
        self._variables = variables
        self._constraints = constraints

    def validate(self, assignment: dict[Variable, int]) -> ValidationResult:
        """Validate a complete or partial assignment."""
        violations: list[tuple[Constraint, str]] = []
        unassigned: list[Variable] = []
        out_of_domain: list[tuple[Variable, int]] = []

        # Check completeness
        for var in self._variables:
            if var not in assignment:
                unassigned.append(var)

        # Check domain membership
        for var, val in assignment.items():
            if not var.domain.contains(val):
                out_of_domain.append((var, val))

        # Check constraints
        for cstr in self._constraints:
            # Only check if all variables in constraint are assigned
            all_assigned = all(v in assignment for v in cstr.variables)
            if all_assigned:
                if not cstr.is_satisfied(assignment):
                    var_vals = ", ".join(
                        f"{v.name}={assignment[v]}" for v in cstr.variables
                    )
                    violations.append((cstr, f"Violated with {var_vals}"))

        is_valid = (
            len(violations) == 0
            and len(unassigned) == 0
            and len(out_of_domain) == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            unassigned_variables=unassigned,
            out_of_domain=out_of_domain,
        )

    def check_partial(self, assignment: dict[Variable, int]) -> list[Constraint]:
        """Check which constraints are violated by a partial assignment.

        Only checks constraints where all involved variables are assigned.
        Returns list of violated constraints.
        """
        violated = []
        for cstr in self._constraints:
            all_assigned = all(v in assignment for v in cstr.variables)
            if all_assigned and not cstr.is_satisfied(assignment):
                violated.append(cstr)
        return violated

    def is_consistent(self, assignment: dict[Variable, int]) -> bool:
        """Quick check: is the partial assignment consistent?

        Returns True if no constraint is violated (unassigned vars are OK).
        """
        for cstr in self._constraints:
            if not cstr.is_satisfied(assignment):
                return False
        return True
