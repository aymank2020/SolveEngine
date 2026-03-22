"""Node consistency enforcement.

Node consistency ensures that every value in a variable's domain satisfies
all unary constraints on that variable. This is a preprocessing step that
should be applied before arc consistency.
"""

from __future__ import annotations

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint, UnaryConstraint


def enforce_node_consistency(
    variables: list[Variable],
    constraints: list[Constraint],
) -> dict[Variable, list[int]]:
    """Enforce node consistency by removing values that violate unary constraints.

    Args:
        variables: List of all variables.
        constraints: List of all constraints (only unary ones are processed).

    Returns:
        Dictionary mapping variables to lists of removed values.
    """
    unary_constraints: list[UnaryConstraint] = [
        c for c in constraints if isinstance(c, UnaryConstraint)
    ]

    pruned: dict[Variable, list[int]] = {}

    for cstr in unary_constraints:
        var = cstr.var
        removed = []
        for val in list(var.domain):
            if not cstr.is_satisfied({var: val}):
                if var.domain.remove(val):
                    removed.append(val)
        if removed:
            if var not in pruned:
                pruned[var] = []
            pruned[var].extend(removed)

    return pruned
