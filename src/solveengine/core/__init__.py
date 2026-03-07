"""Core primitives: Variable, Domain, Constraint."""

from solveengine.core.variable import Variable
from solveengine.core.domain import Domain
from solveengine.core.constraint import Constraint, BinaryConstraint, UnaryConstraint

__all__ = ["Variable", "Domain", "Constraint", "BinaryConstraint", "UnaryConstraint"]
