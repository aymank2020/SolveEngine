"""Backbone detection for constraint satisfaction problems.

The backbone of a CSP is the set of variable-value pairs that appear in
every solution. Identifying the backbone is useful for:
- Understanding problem structure (which decisions are forced)
- Simplifying the problem by fixing backbone variables
- Measuring problem hardness (large backbones correlate with difficulty)

Detection uses iterative solving: for each candidate variable-value pair,
we check if a solution exists where that variable takes a different value.
If no such solution exists, the pair is part of the backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint, UnaryConstraint
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.heuristics.variable_ordering import DomWdegSelector
from solveengine.heuristics.value_ordering import LCVOrderer


@dataclass
class BackboneResult:
    """Result of backbone detection.

    Attributes:
        backbone: Mapping from variable to its backbone value.
        non_backbone: Variables confirmed to have multiple possible values.
        undetermined: Variables whose status could not be determined.
        solutions_checked: Number of solutions examined during detection.
        solver_calls: Number of solver invocations made.
    """

    backbone: dict[Variable, int] = field(default_factory=dict)
    non_backbone: set[Variable] = field(default_factory=set)
    undetermined: set[Variable] = field(default_factory=set)
    solutions_checked: int = 0
    solver_calls: int = 0

    @property
    def backbone_size(self) -> int:
        """Number of backbone variables found."""
        return len(self.backbone)

    @property
    def backbone_fraction(self) -> float:
        """Fraction of variables that are backbone."""
        total = len(self.backbone) + len(self.non_backbone) + len(self.undetermined)
        if total == 0:
            return 0.0
        return len(self.backbone) / total

    def is_backbone_variable(self, var: Variable) -> bool:
        """Check if a variable is in the backbone."""
        return var in self.backbone

    def get_backbone_value(self, var: Variable) -> int | None:
        """Get the backbone value for a variable, or None if not backbone."""
        return self.backbone.get(var)


class BackboneDetector:
    """Detects backbone variables in a CSP.

    Uses iterative solving with negation to identify variables that
    must take the same value in all solutions.

    The algorithm:
    1. Find an initial solution
    2. For each variable-value pair in the solution:
       a. Add a constraint excluding that value
       b. Try to solve
       c. If unsatisfiable, the pair is backbone
       d. If satisfiable, the variable is non-backbone

    Optimizations:
    - Use the new solution to eliminate multiple candidates at once
    - Skip variables already determined
    - Use node limits to avoid spending too long on hard subproblems

    Args:
        variables: List of CSP variables.
        constraints: List of constraints.
        node_limit: Maximum nodes per solver call (0 = unlimited).
        use_iterative_refinement: Use solutions to eliminate multiple candidates.
    """

    def __init__(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        node_limit: int = 100000,
        use_iterative_refinement: bool = True,
    ) -> None:
        self._variables = variables
        self._constraints = constraints
        self._node_limit = node_limit
        self._use_refinement = use_iterative_refinement

    def detect(self) -> BackboneResult:
        """Run backbone detection.

        Returns a BackboneResult with backbone variables, non-backbone
        variables, and any undetermined variables.
        """
        result = BackboneResult()

        # Step 1: Find initial solution
        initial_solution = self._find_solution()
        result.solver_calls += 1

        if initial_solution is None:
            # Problem is unsatisfiable - no backbone
            return result

        result.solutions_checked += 1

        # Initialize candidates: all variables with their solution values
        candidates: dict[Variable, int] = dict(initial_solution)

        # Step 2: Iteratively check each candidate
        for var in list(candidates.keys()):
            if var in result.backbone or var in result.non_backbone:
                continue

            value = candidates[var]
            is_backbone = self._check_backbone_pair(var, value, result)
            result.solver_calls += 1

            if is_backbone is True:
                result.backbone[var] = value
            elif is_backbone is False:
                result.non_backbone.add(var)
                # If refinement is enabled, use the counter-solution
                # to eliminate other candidates
                if self._use_refinement:
                    counter = self._last_counter_solution
                    if counter:
                        result.solutions_checked += 1
                        self._refine_candidates(
                            candidates, counter, result
                        )
            else:
                # Undetermined (solver hit node limit)
                result.undetermined.add(var)

        return result

    def detect_for_subset(
        self, target_vars: Sequence[Variable]
    ) -> BackboneResult:
        """Detect backbone status for a specific subset of variables.

        More efficient when only a few variables need to be checked.

        Args:
            target_vars: Variables to check for backbone membership.

        Returns:
            BackboneResult for the target variables only.
        """
        result = BackboneResult()

        initial_solution = self._find_solution()
        result.solver_calls += 1

        if initial_solution is None:
            return result

        result.solutions_checked += 1

        for var in target_vars:
            if var not in initial_solution:
                result.undetermined.add(var)
                continue

            value = initial_solution[var]
            is_backbone = self._check_backbone_pair(var, value, result)
            result.solver_calls += 1

            if is_backbone is True:
                result.backbone[var] = value
            elif is_backbone is False:
                result.non_backbone.add(var)
            else:
                result.undetermined.add(var)

        return result

    def _find_solution(self) -> dict[Variable, int] | None:
        """Find a solution to the CSP."""
        self._restore_domains()
        solver = BacktrackSolver(
            self._variables,
            self._constraints,
            var_selector=DomWdegSelector(),
            val_orderer=LCVOrderer(),
            use_forward_check=True,
        )
        if self._node_limit > 0:
            solver.set_node_limit(self._node_limit)
        return solver.solve()

    def _check_backbone_pair(
        self, var: Variable, value: int, result: BackboneResult
    ) -> bool | None:
        """Check if (var, value) is a backbone pair.

        Returns True if backbone, False if not, None if undetermined.
        """
        self._last_counter_solution: dict[Variable, int] | None = None
        self._restore_domains()

        # Add negation constraint: var != value
        negation = UnaryConstraint(
            var, lambda v, excluded=value: v != excluded,
            f"negate_{var.name}!={value}",
        )
        augmented_constraints = self._constraints + [negation]

        solver = BacktrackSolver(
            self._variables,
            augmented_constraints,
            var_selector=DomWdegSelector(),
            val_orderer=LCVOrderer(),
            use_forward_check=True,
        )
        if self._node_limit > 0:
            solver.set_node_limit(self._node_limit)

        counter_solution = solver.solve()

        if counter_solution is None:
            # Check if we hit the node limit
            if self._node_limit > 0 and solver.stats.nodes_explored >= self._node_limit:
                return None  # Undetermined
            return True  # Backbone: no solution without this value

        self._last_counter_solution = counter_solution
        return False  # Not backbone: found a solution with different value

    def _refine_candidates(
        self,
        candidates: dict[Variable, int],
        counter_solution: dict[Variable, int],
        result: BackboneResult,
    ) -> None:
        """Use a counter-solution to eliminate backbone candidates.

        If a variable has a different value in the counter-solution,
        it cannot be backbone.
        """
        for var, candidate_value in list(candidates.items()):
            if var in result.backbone or var in result.non_backbone:
                continue
            if var in counter_solution:
                if counter_solution[var] != candidate_value:
                    result.non_backbone.add(var)

    def _restore_domains(self) -> None:
        """Restore all variable domains to initial state."""
        for var in self._variables:
            var.restore_to(0)
            var.unassign()


def compute_backbone_fraction(
    variables: list[Variable], constraints: list[Constraint]
) -> float:
    """Convenience function to compute the backbone fraction.

    Returns the fraction of variables that are backbone (0.0 to 1.0).
    Returns 0.0 if the problem is unsatisfiable.
    """
    detector = BackboneDetector(variables, constraints)
    result = detector.detect()
    return result.backbone_fraction


def find_forced_assignments(
    variables: list[Variable], constraints: list[Constraint]
) -> dict[Variable, int]:
    """Find all forced variable assignments (backbone).

    Returns a mapping from backbone variables to their forced values.
    Useful for preprocessing before search.
    """
    detector = BackboneDetector(variables, constraints)
    result = detector.detect()
    return dict(result.backbone)
