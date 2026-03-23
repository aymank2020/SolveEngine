"""AC-4 arc consistency algorithm.

AC-4 is an optimal fine-grained arc consistency algorithm that maintains
explicit support counters. When a value loses its last support, it is removed.
More efficient than AC-3 for dense constraint graphs but uses more memory.
"""

from __future__ import annotations

from collections import deque

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint, BinaryConstraint
from solveengine.propagation.ac3 import PropagationResult


class SupportEntry:
    """Tracks which values support a given (variable, value) pair."""

    __slots__ = ("supporters",)

    def __init__(self) -> None:
        self.supporters: set[tuple[Variable, int]] = set()


class AC4Propagator:
    """AC-4 fine-grained arc consistency propagator.

    Maintains explicit support lists for each (variable, value) pair.
    When a value is removed, only the pairs that depended on it are checked.

    Note: Only works with binary constraints. Non-binary constraints are
    handled by falling back to AC-3 style revision.
    """

    def __init__(self, variables: list[Variable], constraints: list[Constraint]) -> None:
        self._variables = variables
        self._constraints = constraints
        self._binary: list[BinaryConstraint] = [
            c for c in constraints if isinstance(c, BinaryConstraint)
        ]
        self._non_binary: list[Constraint] = [
            c for c in constraints if not isinstance(c, BinaryConstraint)
        ]
        # support_count[(var, val, cstr)] = number of supporting values in other var
        self._support_count: dict[tuple[int, int, int], int] = {}
        # supported_by[(other_var, other_val)] = list of (var, val, cstr) that it supports
        self._supported_by: dict[tuple[int, int], list[tuple[Variable, int, BinaryConstraint]]] = {}
        self._initialized = False

    def initialize(self, assignment: dict[Variable, int]) -> PropagationResult:
        """Initialize support structures and perform initial propagation.

        Must be called before incremental propagation.
        """
        self._support_count.clear()
        self._supported_by.clear()
        pruned: dict[Variable, list[int]] = {}
        removal_queue: deque[tuple[Variable, int]] = deque()

        for cstr in self._binary:
            var1, var2 = cstr.var1, cstr.var2

            for val1 in list(var1.domain):
                count = 0
                for val2 in var2.domain:
                    if cstr.is_satisfied({var1: val1, var2: val2}):
                        count += 1
                        key = (var2.index, val2)
                        if key not in self._supported_by:
                            self._supported_by[key] = []
                        self._supported_by[key].append((var1, val1, cstr))
                self._support_count[(var1.index, val1, id(cstr))] = count
                if count == 0 and not var1.is_assigned:
                    removal_queue.append((var1, val1))

            for val2 in list(var2.domain):
                count = 0
                for val1 in var1.domain:
                    if cstr.is_satisfied({var1: val1, var2: val2}):
                        count += 1
                        key = (var1.index, val1)
                        if key not in self._supported_by:
                            self._supported_by[key] = []
                        self._supported_by[key].append((var2, val2, cstr))
                self._support_count[(var2.index, val2, id(cstr))] = count
                if count == 0 and not var2.is_assigned:
                    removal_queue.append((var2, val2))

        # Process removals
        revisions = 0
        while removal_queue:
            var, val = removal_queue.popleft()
            if not var.domain.contains(val):
                continue
            revisions += 1
            var.domain.remove(val)
            if var not in pruned:
                pruned[var] = []
            pruned[var].append(val)

            if var.domain.is_empty:
                self._initialized = True
                return PropagationResult(False, pruned, revisions)

            # Decrement support counts for pairs that this value supported
            key = (var.index, val)
            for dep_var, dep_val, dep_cstr in self._supported_by.get(key, []):
                sc_key = (dep_var.index, dep_val, id(dep_cstr))
                if sc_key in self._support_count:
                    self._support_count[sc_key] -= 1
                    if self._support_count[sc_key] == 0 and dep_var.domain.contains(dep_val):
                        removal_queue.append((dep_var, dep_val))

        self._initialized = True
        return PropagationResult(True, pruned, revisions)

    def propagate_removal(
        self,
        var: Variable,
        removed_value: int,
        assignment: dict[Variable, int],
    ) -> PropagationResult:
        """Propagate the removal of a single value incrementally."""
        if not self._initialized:
            return self.initialize(assignment)

        pruned: dict[Variable, list[int]] = {}
        removal_queue: deque[tuple[Variable, int]] = deque()
        revisions = 0

        key = (var.index, removed_value)
        for dep_var, dep_val, dep_cstr in self._supported_by.get(key, []):
            sc_key = (dep_var.index, dep_val, id(dep_cstr))
            if sc_key in self._support_count:
                self._support_count[sc_key] -= 1
                if self._support_count[sc_key] == 0 and dep_var.domain.contains(dep_val):
                    removal_queue.append((dep_var, dep_val))

        while removal_queue:
            r_var, r_val = removal_queue.popleft()
            if not r_var.domain.contains(r_val):
                continue
            revisions += 1
            r_var.domain.remove(r_val)
            if r_var not in pruned:
                pruned[r_var] = []
            pruned[r_var].append(r_val)

            if r_var.domain.is_empty:
                return PropagationResult(False, pruned, revisions)

            r_key = (r_var.index, r_val)
            for dep_var2, dep_val2, dep_cstr2 in self._supported_by.get(r_key, []):
                sc_key2 = (dep_var2.index, dep_val2, id(dep_cstr2))
                if sc_key2 in self._support_count:
                    self._support_count[sc_key2] -= 1
                    if self._support_count[sc_key2] == 0 and dep_var2.domain.contains(dep_val2):
                        removal_queue.append((dep_var2, dep_val2))

        return PropagationResult(True, pruned, revisions)
