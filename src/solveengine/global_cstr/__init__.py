"""Global constraints: AllDifferent, Sum, Element, Cardinality."""

from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.global_cstr.linear import SumConstraint, ScalarProduct
from solveengine.global_cstr.element import ElementConstraint
from solveengine.global_cstr.cardinality import CardinalityConstraint

__all__ = [
    "AllDifferent",
    "SumConstraint",
    "ScalarProduct",
    "ElementConstraint",
    "CardinalityConstraint",
]
