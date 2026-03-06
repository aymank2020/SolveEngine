"""SolveEngine: Constraint satisfaction solver with propagation and learning."""

from solveengine.core.variable import Variable
from solveengine.core.domain import Domain
from solveengine.core.constraint import Constraint, BinaryConstraint, UnaryConstraint
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.global_cstr.linear import SumConstraint

__version__ = "0.4.0"

__all__ = [
    "Variable",
    "Domain",
    "Constraint",
    "BinaryConstraint",
    "UnaryConstraint",
    "BacktrackSolver",
    "AllDifferent",
    "SumConstraint",
]
