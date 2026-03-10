"""Core type definitions and protocols for SolveEngine.

Defines the fundamental types and protocols used throughout the solver.
These provide a stable interface that modules can depend on without
creating circular imports.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable, Any
from dataclasses import dataclass


class SolverStatus(Enum):
    """Status of the solver."""

    IDLE = "idle"
    RUNNING = "running"
    SOLVED = "solved"
    UNSATISFIABLE = "unsatisfiable"
    TIMEOUT = "timeout"
    ABORTED = "aborted"
    ERROR = "error"


class PropagationLevel(Enum):
    """Level of consistency enforcement."""

    NONE = "none"
    NODE = "node"
    ARC = "arc"
    BOUNDS = "bounds"
    SINGLETON = "singleton"
    FULL = "full"


class SearchPhase(Enum):
    """Current phase of the search process."""

    PREPROCESSING = "preprocessing"
    INITIAL_PROPAGATION = "initial_propagation"
    SEARCH = "search"
    BACKTRACKING = "backtracking"
    RESTART = "restart"
    POSTPROCESSING = "postprocessing"


@runtime_checkable
class Propagator(Protocol):
    """Protocol for constraint propagators."""

    def propagate(self, assignment: dict[Any, int], trigger_var: Any | None = None) -> Any:
        """Run propagation and return result."""
        ...


@runtime_checkable
class Selector(Protocol):
    """Protocol for variable selectors."""

    def select(self, unassigned: list[Any], constraints: list[Any], assignment: dict[Any, int]) -> Any:
        """Select the next variable to assign."""
        ...


@runtime_checkable
class Orderer(Protocol):
    """Protocol for value orderers."""

    def order(self, var: Any, constraints: list[Any], assignment: dict[Any, int]) -> list[int]:
        """Order the values of a variable for trial."""
        ...


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for solver behavior."""

    propagation_level: PropagationLevel = PropagationLevel.ARC
    use_restarts: bool = False
    use_learning: bool = False
    use_backjumping: bool = False
    node_limit: int = 0
    time_limit_ms: int = 0
    solution_limit: int = 1
    random_seed: int | None = None
    verbose: bool = False

    @property
    def has_node_limit(self) -> bool:
        return self.node_limit > 0

    @property
    def has_time_limit(self) -> bool:
        return self.time_limit_ms > 0

    @property
    def find_all(self) -> bool:
        return self.solution_limit == 0


@dataclass
class SolverResult:
    """Complete result from a solver run."""

    status: SolverStatus
    solutions: list[dict[str, int]]
    nodes_explored: int = 0
    backtracks: int = 0
    propagations: int = 0
    restarts: int = 0
    time_ms: float = 0.0

    @property
    def is_solved(self) -> bool:
        return self.status == SolverStatus.SOLVED

    @property
    def is_unsat(self) -> bool:
        return self.status == SolverStatus.UNSATISFIABLE

    @property
    def first_solution(self) -> dict[str, int] | None:
        return self.solutions[0] if self.solutions else None

    @property
    def num_solutions(self) -> int:
        return len(self.solutions)
