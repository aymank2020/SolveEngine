"""Search tree tracer for debugging solver behavior.

Records decisions, propagations, and backtracks during solving
to help understand solver behavior and identify performance issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from solveengine.core.variable import Variable


class EventType(Enum):
    DECISION = "decision"
    PROPAGATION = "propagation"
    BACKTRACK = "backtrack"
    WIPEOUT = "wipeout"
    SOLUTION = "solution"
    PRUNE = "prune"


@dataclass
class TraceEvent:
    """A single event in the search trace."""

    event_type: EventType
    depth: int
    variable: str | None = None
    value: int | None = None
    pruned_count: int = 0
    message: str = ""


class SearchTracer:
    """Records search events for debugging and analysis.

    Can be attached to a solver to record all decisions, propagations,
    and backtracks. Useful for understanding why a solver is slow or
    why it fails to find a solution.
    """

    def __init__(self, max_events: int = 100000) -> None:
        self._events: list[TraceEvent] = []
        self._max_events = max_events
        self._enabled = True
        self._depth = 0

    @property
    def events(self) -> list[TraceEvent]:
        return self._events

    @property
    def num_events(self) -> int:
        return len(self._events)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def clear(self) -> None:
        self._events.clear()
        self._depth = 0

    def record_decision(self, var: Variable, value: int, depth: int) -> None:
        """Record a variable assignment decision."""
        if not self._enabled:
            return
        self._depth = depth
        self._add_event(TraceEvent(
            event_type=EventType.DECISION,
            depth=depth,
            variable=var.name,
            value=value,
        ))

    def record_propagation(self, var: Variable, pruned_count: int, depth: int) -> None:
        """Record a propagation event."""
        if not self._enabled:
            return
        self._add_event(TraceEvent(
            event_type=EventType.PROPAGATION,
            depth=depth,
            variable=var.name,
            pruned_count=pruned_count,
        ))

    def record_backtrack(self, var: Variable, depth: int) -> None:
        """Record a backtrack event."""
        if not self._enabled:
            return
        self._add_event(TraceEvent(
            event_type=EventType.BACKTRACK,
            depth=depth,
            variable=var.name,
        ))

    def record_wipeout(self, var: Variable, depth: int) -> None:
        """Record a domain wipeout."""
        if not self._enabled:
            return
        self._add_event(TraceEvent(
            event_type=EventType.WIPEOUT,
            depth=depth,
            variable=var.name,
            message=f"Domain of {var.name} wiped out",
        ))

    def record_solution(self, depth: int) -> None:
        """Record that a solution was found."""
        if not self._enabled:
            return
        self._add_event(TraceEvent(
            event_type=EventType.SOLUTION,
            depth=depth,
        ))

    def _add_event(self, event: TraceEvent) -> None:
        if len(self._events) < self._max_events:
            self._events.append(event)

    def summary(self) -> dict[str, int]:
        """Get a summary of event counts by type."""
        counts: dict[str, int] = {}
        for event in self._events:
            key = event.event_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def decisions_at_depth(self, depth: int) -> list[TraceEvent]:
        """Get all decision events at a specific depth."""
        return [
            e for e in self._events
            if e.event_type == EventType.DECISION and e.depth == depth
        ]

    def max_depth_reached(self) -> int:
        """Maximum search depth reached during solving."""
        if not self._events:
            return 0
        return max(e.depth for e in self._events)

    def backtrack_ratio(self) -> float:
        """Ratio of backtracks to decisions (higher = more thrashing)."""
        decisions = sum(1 for e in self._events if e.event_type == EventType.DECISION)
        backtracks = sum(1 for e in self._events if e.event_type == EventType.BACKTRACK)
        if decisions == 0:
            return 0.0
        return backtracks / decisions
