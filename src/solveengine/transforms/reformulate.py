"""Constraint reformulation transforms.

Transforms constraints into equivalent but more propagation-friendly forms.
For example, decomposing a global constraint into binary constraints,
or merging overlapping binary constraints into a table constraint.
"""

from __future__ import annotations

from solveengine.core.variable import Variable
from solveengine.core.constraint import (
    Constraint,
    BinaryConstraint,
    TableConstraint,
)
from solveengine.global_cstr.alldiff import AllDifferent


class ConstraintReformulator:
    """Transforms constraints into equivalent alternative forms."""

    def decompose_alldiff(self, alldiff: AllDifferent) -> list[BinaryConstraint]:
        """Decompose AllDifferent into pairwise not-equal constraints.

        AllDiff(x1, ..., xn) → xi != xj for all i < j.

        Note: The decomposition loses the global propagation power of
        AllDifferent but may be useful for solvers that only handle binary.
        """
        variables = alldiff.variables
        constraints = []
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                cstr = BinaryConstraint(
                    variables[i],
                    variables[j],
                    lambda a, b: a != b,
                    name=f"{variables[i].name}!={variables[j].name}",
                )
                constraints.append(cstr)
        return constraints

    def merge_to_table(
        self,
        constraints: list[BinaryConstraint],
        var1: Variable,
        var2: Variable,
    ) -> TableConstraint | None:
        """Merge multiple binary constraints between the same pair into a table.

        Computes the intersection of allowed tuples from all constraints.
        Returns None if no constraints exist between the pair.
        """
        relevant = [
            c for c in constraints
            if (c.var1 is var1 and c.var2 is var2) or (c.var1 is var2 and c.var2 is var1)
        ]

        if not relevant:
            return None

        # Compute allowed tuples
        allowed: set[tuple[int, ...]] = set()
        for val1 in var1.domain:
            for val2 in var2.domain:
                assignment = {var1: val1, var2: val2}
                if all(c.is_satisfied(assignment) for c in relevant):
                    allowed.add((val1, val2))

        return TableConstraint([var1, var2], allowed, name=f"Table({var1.name},{var2.name})")

    def tighten_domains(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> dict[Variable, list[int]]:
        """Pre-process: remove values that cannot participate in any solution.

        For each variable, remove values that have no support in at least
        one constraint. This is equivalent to enforcing node + arc consistency.
        """
        pruned: dict[Variable, list[int]] = {}

        for var in variables:
            removed = []
            for val in list(var.domain):
                # Check if val has support in all constraints involving var
                has_support = True
                for cstr in constraints:
                    if not cstr.involves(var):
                        continue
                    supported = cstr.get_supported_values(var, {})
                    if val not in supported:
                        has_support = False
                        break

                if not has_support:
                    if var.domain.remove(val):
                        removed.append(val)

            if removed:
                pruned[var] = removed

        return pruned

    def implied_constraints(
        self,
        constraints: list[Constraint],
        variables: list[Variable],
    ) -> list[BinaryConstraint]:
        """Detect implied constraints that can strengthen propagation.

        For example, if x < y and y < z, then x < z is implied.
        Adding it explicitly can improve propagation.
        """
        implied: list[BinaryConstraint] = []

        # Find transitivity chains in less-than constraints
        less_than: dict[int, set[int]] = {}  # var_idx -> set of vars it's less than
        for cstr in constraints:
            if not isinstance(cstr, BinaryConstraint):
                continue
            # Check if this is a less-than constraint by testing boundary values
            v1, v2 = cstr.var1, cstr.var2
            if v1.domain_size > 0 and v2.domain_size > 0:
                min1 = v1.domain.min_value
                max2 = v2.domain.max_value
                # Heuristic: if (max, min) fails and (min, max) passes, likely <
                if (not cstr.is_satisfied({v1: max2, v2: min1}) and
                    cstr.is_satisfied({v1: min1, v2: max2})):
                    if v1.index not in less_than:
                        less_than[v1.index] = set()
                    less_than[v1.index].add(v2.index)

        # Find transitive pairs
        var_by_idx = {v.index: v for v in variables}
        for a_idx, b_set in less_than.items():
            for b_idx in b_set:
                if b_idx in less_than:
                    for c_idx in less_than[b_idx]:
                        if c_idx != a_idx and c_idx not in less_than.get(a_idx, set()):
                            a_var = var_by_idx[a_idx]
                            c_var = var_by_idx[c_idx]
                            implied.append(BinaryConstraint(
                                a_var, c_var,
                                lambda x, y: x < y,
                                name=f"implied_{a_var.name}<{c_var.name}",
                            ))

        return implied
