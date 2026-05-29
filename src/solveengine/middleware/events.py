"""Event system for solver observability.

Provides a publish-subscribe event bus that solver components can use
to communicate without tight coupling. Events carry structured data
about solver state changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventType(Enum):
    DECISION_MADE = "decision_made"
    BACKTRACK = "backtrack"
    PROPAGATION_START = "propagation_start"
    PROPAGATION_END = "propagation_end"
    DOMAIN_REDUCED = "domain_reduced"
    DOMAIN_WIPEOUT = "domain_wipeout"
    SOLUTION_FOUND = "solution_found"
    RESTART = "restart"
    NOGOOD_LEARNED = "nogood_learned"
    CONSTRAINT_ADDED = "constraint_added"
    SEARCH_START = "search_start"
    SEARCH_END = "search_end"


@dataclass
class SolverEvent:
    """A solver event with type and payload."""

    event_type: EventType
    depth: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def variable_name(self) -> str | None:
        return self.data.get("variable")

    @property
    def value(self) -> int | None:
        return self.data.get("value")

    def __repr__(self) -> str:
        return f"SolverEvent({self.event_type.value}, depth={self.depth})"


# Type alias for event handlers
EventHandler = Callable[[SolverEvent], None]


class EventBus:
    """Publish-subscribe event bus for solver events.

    Components can subscribe to specific event types and receive
    notifications when those events occur. Supports filtering by
    event type and wildcard subscriptions.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._event_log: list[SolverEvent] = []
        self._log_enabled: bool = False
        self._max_log_size: int = 10000

    @property
    def log(self) -> list[SolverEvent]:
        return self._event_log

    @property
    def log_enabled(self) -> bool:
        return self._log_enabled

    def enable_logging(self, max_size: int = 10000) -> None:
        """Enable event logging."""
        self._log_enabled = True
        self._max_log_size = max_size

    def disable_logging(self) -> None:
        """Disable event logging."""
        self._log_enabled = False

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe to a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all event types."""
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe from a specific event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h is not handler
            ]

    def publish(self, event: SolverEvent) -> None:
        """Publish an event to all subscribers."""
        if self._log_enabled:
            if len(self._event_log) < self._max_log_size:
                self._event_log.append(event)

        # Notify type-specific handlers
        for handler in self._handlers.get(event.event_type, []):
            handler(event)

        # Notify global handlers
        for handler in self._global_handlers:
            handler(event)

    def clear_log(self) -> None:
        """Clear the event log."""
        self._event_log.clear()

    def clear_handlers(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
        self._global_handlers.clear()

    def event_count(self, event_type: EventType | None = None) -> int:
        """Count events in the log, optionally filtered by type."""
        if event_type is None:
            return len(self._event_log)
        return sum(1 for e in self._event_log if e.event_type == event_type)
