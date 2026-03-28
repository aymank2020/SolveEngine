"""Queue policies for propagation ordering.

Different queue orderings can significantly affect propagation efficiency.
This module provides pluggable policies for the AC-3 propagator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class QueuePolicy(ABC):
    """Abstract base for propagation queue ordering policies."""

    @abstractmethod
    def create_queue(
        self, arcs: Sequence[tuple[Variable, Constraint]]
    ) -> deque[tuple[Variable, Constraint]]:
        """Create an ordered queue from the given arcs."""
        ...

    @abstractmethod
    def priority(self, var: Variable, cstr: Constraint) -> float:
        """Return priority for an arc (lower = higher priority)."""
        ...


class FIFOPolicy(QueuePolicy):
    """First-in-first-out ordering (standard AC-3)."""

    def create_queue(
        self, arcs: Sequence[tuple[Variable, Constraint]]
    ) -> deque[tuple[Variable, Constraint]]:
        return deque(arcs)

    def priority(self, var: Variable, cstr: Constraint) -> float:
        return 0.0


class SmallDomainFirstPolicy(QueuePolicy):
    """Process arcs with smaller variable domains first.

    Intuition: smaller domains are more likely to cause wipeouts,
    so detecting failures early saves work.
    """

    def create_queue(
        self, arcs: Sequence[tuple[Variable, Constraint]]
    ) -> deque[tuple[Variable, Constraint]]:
        sorted_arcs = sorted(arcs, key=lambda a: a[0].domain_size)
        return deque(sorted_arcs)

    def priority(self, var: Variable, cstr: Constraint) -> float:
        return float(var.domain_size)


class HighWeightFirstPolicy(QueuePolicy):
    """Process arcs involving high-weight variables first.

    Uses the dom/wdeg weight to prioritize variables that have
    historically caused more failures.
    """

    def create_queue(
        self, arcs: Sequence[tuple[Variable, Constraint]]
    ) -> deque[tuple[Variable, Constraint]]:
        sorted_arcs = sorted(arcs, key=lambda a: -a[0].weight)
        return deque(sorted_arcs)

    def priority(self, var: Variable, cstr: Constraint) -> float:
        return -var.weight


class ConstraintArityPolicy(QueuePolicy):
    """Process arcs from higher-arity constraints first.

    Higher-arity constraints are more likely to prune values,
    so processing them first can trigger more propagation.
    """

    def create_queue(
        self, arcs: Sequence[tuple[Variable, Constraint]]
    ) -> deque[tuple[Variable, Constraint]]:
        sorted_arcs = sorted(arcs, key=lambda a: -a[1].arity)
        return deque(sorted_arcs)

    def priority(self, var: Variable, cstr: Constraint) -> float:
        return -float(cstr.arity)
