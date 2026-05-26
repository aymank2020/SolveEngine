"""Model checking for constraint satisfaction problems.

Provides verification utilities for CSP models:
- Consistency checking: verifies that the model has no trivially unsatisfiable
  constraints (empty domains, impossible unary constraints, etc.)
- Redundancy detection: identifies constraints that are implied by others
- Implied constraint detection: finds constraints that could be added to
  strengthen propagation without changing the solution set
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint, BinaryConstraint, UnaryConstraint


@dataclass
class CheckResult:
    """Result of a model checking operation."""

    is_valid: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    redundant_constraints: list[Constraint] = field(default_factory=list)
    implied_constraints: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def summary(self) -> str:
        """Return a human-readable summary of the check results."""
        lines = []
        status = "VALID" if self.is_valid else "INVALID"
        lines.append(f"Model Check: {status}")
        if self.issues:
            lines.append(f"  Issues ({len(self.issues)}):")
            for issue in self.issues:
                lines.append(f"    - {issue}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"    - {warning}")
        if self.redundant_constraints:
            lines.append(f"  Redundant constraints: {len(self.redundant_constraints)}")
        if self.implied_constraints:
            lines.append(f"  Implied constraints: {len(self.implied_constraints)}")
        return "\n".join(lines)


class ModelChecker:
    """Verifies CSP model consistency and quality.

    Performs static analysis on the model to detect issues before solving.
    This can save significant time by catching modeling errors early.

    Args:
        variables: All variables in the CSP.
        constraints: All constraints in the CSP.
    """

    def __init__(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> None:
        self._variables = variables
        self._constraints = constraints
        self._var_constraints: dict[Variable, list[Constraint]] = {}
        self._build_var_map()

    def _build_var_map(self) -> None:
        """Build mapping from variables to their constraints."""
        for var in self._variables:
            self._var_constraints[var] = []
        for cstr in self._constraints:
            for var in cstr.variables:
                if var in self._var_constraints:
                    self._var_constraints[var].append(cstr)

    def check_all(self) -> CheckResult:
        """Run all model checks and return combined results."""
        result = CheckResult(is_valid=True)

        self._check_empty_domains(result)
        self._check_unary_consistency(result)
        self._check_binary_arc_consistency(result)
        self._check_duplicate_constraints(result)
        self._check_disconnected_variables(result)
        self._check_redundant_constraints(result)
        self._detect_implied_constraints(result)

        return result

    def check_consistency(self) -> CheckResult:
        """Check basic model consistency (fast checks only)."""
        result = CheckResult(is_valid=True)
        self._check_empty_domains(result)
        self._check_unary_consistency(result)
        self._check_binary_arc_consistency(result)
        return result

    def check_redundancy(self) -> CheckResult:
        """Check for redundant constraints."""
        result = CheckResult(is_valid=True)
        self._check_duplicate_constraints(result)
        self._check_redundant_constraints(result)
        return result

    def _check_empty_domains(self, result: CheckResult) -> None:
        """Check for variables with empty domains."""
        for var in self._variables:
            if var.domain.is_empty:
                result.is_valid = False
                result.issues.append(
                    f"Variable '{var.name}' has an empty domain"
                )

    def _check_unary_consistency(self, result: CheckResult) -> None:
        """Check that unary constraints don't eliminate all values."""
        for cstr in self._constraints:
            if not isinstance(cstr, UnaryConstraint):
                continue
            var = cstr.var
            supported = cstr.get_supported_values(var, {})
            current_domain = var.domain.values()
            feasible = supported & current_domain

            if not feasible:
                result.is_valid = False
                result.issues.append(
                    f"Unary constraint '{cstr.name}' eliminates all values "
                    f"from variable '{var.name}'"
                )
            elif len(feasible) < len(current_domain):
                removed_count = len(current_domain) - len(feasible)
                result.warnings.append(
                    f"Unary constraint '{cstr.name}' removes {removed_count} "
                    f"values from '{var.name}' (consider pre-filtering)"
                )

    def _check_binary_arc_consistency(self, result: CheckResult) -> None:
        """Check binary constraints for arc consistency violations."""
        for cstr in self._constraints:
            if not isinstance(cstr, BinaryConstraint):
                continue

            var1, var2 = cstr.var1, cstr.var2

            supported_1 = cstr.get_supported_values(var1, {})
            if not (supported_1 & var1.domain.values()):
                result.is_valid = False
                result.issues.append(
                    f"Binary constraint '{cstr.name}': no supported values "
                    f"for '{var1.name}'"
                )

            supported_2 = cstr.get_supported_values(var2, {})
            if not (supported_2 & var2.domain.values()):
                result.is_valid = False
                result.issues.append(
                    f"Binary constraint '{cstr.name}': no supported values "
                    f"for '{var2.name}'"
                )

    def _check_duplicate_constraints(self, result: CheckResult) -> None:
        """Detect duplicate constraints (same variables, same type)."""
        seen: dict[tuple[str, frozenset[int]], Constraint] = {}

        for cstr in self._constraints:
            var_ids = frozenset(v.index for v in cstr.variables)
            key = (type(cstr).__name__, var_ids)

            if key in seen:
                result.warnings.append(
                    f"Possible duplicate constraint: '{cstr.name}' and "
                    f"'{seen[key].name}' have same type and variables"
                )
                result.redundant_constraints.append(cstr)
            else:
                seen[key] = cstr

    def _check_disconnected_variables(self, result: CheckResult) -> None:
        """Detect variables not involved in any constraint."""
        for var in self._variables:
            constraints = self._var_constraints.get(var, [])
            if not constraints:
                result.warnings.append(
                    f"Variable '{var.name}' is not involved in any constraint "
                    f"(unconstrained variable)"
                )

    def _check_redundant_constraints(self, result: CheckResult) -> None:
        """Detect constraints that are implied by others.

        A constraint C is redundant if removing it doesn't change the
        solution set. We approximate this by checking if C's filtering
        is subsumed by other constraints on the same variables.
        """
        for i, cstr in enumerate(self._constraints):
            if not isinstance(cstr, BinaryConstraint):
                continue

            var1, var2 = cstr.var1, cstr.var2
            other_constraints = [
                c for j, c in enumerate(self._constraints)
                if j != i and isinstance(c, BinaryConstraint)
                and c.var1 is var1 and c.var2 is var2
            ]

            if not other_constraints:
                continue

            is_redundant = self._is_subsumed_by(cstr, other_constraints)
            if is_redundant:
                result.redundant_constraints.append(cstr)
                result.warnings.append(
                    f"Constraint '{cstr.name}' appears to be subsumed by "
                    f"other constraints on the same variables"
                )

    def _is_subsumed_by(
        self, cstr: BinaryConstraint, others: list[BinaryConstraint]
    ) -> bool:
        """Check if cstr is subsumed by the conjunction of others.

        Tests a sample of value pairs to see if cstr is always at least
        as permissive as the conjunction of others.
        """
        var1, var2 = cstr.var1, cstr.var2
        sample_vals1 = list(var1.domain)[:20]
        sample_vals2 = list(var2.domain)[:20]

        for v1 in sample_vals1:
            for v2 in sample_vals2:
                assignment = {var1: v1, var2: v2}
                others_satisfied = all(c.is_satisfied(assignment) for c in others)
                cstr_satisfied = cstr.is_satisfied(assignment)

                if others_satisfied and not cstr_satisfied:
                    return False

        return True

    def _detect_implied_constraints(self, result: CheckResult) -> None:
        """Detect potential implied constraints that could strengthen propagation.

        Looks for transitivity patterns: if x != y and y != z, then
        potentially x != z could be added (if not already present).
        """
        binary_pairs: dict[frozenset[int], BinaryConstraint] = {}
        for cstr in self._constraints:
            if isinstance(cstr, BinaryConstraint):
                key = frozenset([cstr.var1.index, cstr.var2.index])
                binary_pairs[key] = cstr

        var_by_index: dict[int, Variable] = {v.index: v for v in self._variables}

        for var in self._variables:
            neighbors = self._get_binary_neighbors(var)
            for n1, n2 in combinations(neighbors, 2):
                pair_key = frozenset([n1.index, n2.index])
                if pair_key not in binary_pairs:
                    if self._check_transitivity_implies(var, n1, n2):
                        result.implied_constraints.append(
                            f"Implied constraint between '{n1.name}' and "
                            f"'{n2.name}' (via transitivity through '{var.name}')"
                        )

    def _get_binary_neighbors(self, var: Variable) -> list[Variable]:
        """Get all variables connected to var by binary constraints."""
        neighbors = []
        for cstr in self._var_constraints.get(var, []):
            if isinstance(cstr, BinaryConstraint):
                if cstr.var1 is var:
                    neighbors.append(cstr.var2)
                elif cstr.var2 is var:
                    neighbors.append(cstr.var1)
        return neighbors

    def _check_transitivity_implies(
        self, pivot: Variable, n1: Variable, n2: Variable
    ) -> bool:
        """Check if constraints through pivot imply a constraint between n1 and n2.

        Samples value combinations to detect if the transitive closure
        restricts the (n1, n2) pair beyond their domains.
        """
        sample_n1 = list(n1.domain)[:10]
        sample_n2 = list(n2.domain)[:10]
        restricted_pairs = 0
        total_pairs = 0

        for v1 in sample_n1:
            for v2 in sample_n2:
                total_pairs += 1
                has_support = False
                for pv in pivot.domain:
                    assignment_1 = {pivot: pv, n1: v1}
                    assignment_2 = {pivot: pv, n2: v2}
                    cstr1_ok = all(
                        c.is_satisfied(assignment_1)
                        for c in self._var_constraints.get(pivot, [])
                        if isinstance(c, BinaryConstraint) and c.involves(n1)
                    )
                    cstr2_ok = all(
                        c.is_satisfied(assignment_2)
                        for c in self._var_constraints.get(pivot, [])
                        if isinstance(c, BinaryConstraint) and c.involves(n2)
                    )
                    if cstr1_ok and cstr2_ok:
                        has_support = True
                        break
                if not has_support:
                    restricted_pairs += 1

        if total_pairs == 0:
            return False
        return restricted_pairs > 0 and (restricted_pairs / total_pairs) > 0.1

    def verify_solution(self, assignment: dict[Variable, int]) -> CheckResult:
        """Verify that a complete assignment satisfies all constraints."""
        result = CheckResult(is_valid=True)

        for var in self._variables:
            if var not in assignment:
                result.is_valid = False
                result.issues.append(f"Variable '{var.name}' is not assigned")
            elif not var.domain.contains(assignment[var]):
                result.is_valid = False
                result.issues.append(
                    f"Variable '{var.name}' assigned value {assignment[var]} "
                    f"which is not in its domain"
                )

        for cstr in self._constraints:
            if not cstr.is_satisfied(assignment):
                result.is_valid = False
                result.issues.append(f"Constraint '{cstr.name}' is violated")

        return result
