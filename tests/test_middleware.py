"""Tests for middleware: hooks and events."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.middleware.hooks import SolverHook, HookManager
from solveengine.middleware.events import EventBus, SolverEvent, EventType


class CountingHook(SolverHook):
    """Test hook that counts events."""

    def __init__(self):
        self.decisions = 0
        self.backtracks = 0
        self.solutions = 0

    def on_decision(self, var, value, depth):
        self.decisions += 1

    def on_backtrack(self, var, depth):
        self.backtracks += 1

    def on_solution(self, assignment):
        self.solutions += 1


class TestHookManager:
    def test_register_and_fire(self):
        """Registered hooks receive events."""
        manager = HookManager()
        hook = CountingHook()
        manager.register(hook)

        var = Variable("x", [1, 2])
        manager.fire_decision(var, 1, 0)
        assert hook.decisions == 1

    def test_multiple_hooks(self):
        """Multiple hooks all receive events."""
        manager = HookManager()
        h1 = CountingHook()
        h2 = CountingHook()
        manager.register(h1)
        manager.register(h2)

        var = Variable("x", [1, 2])
        manager.fire_decision(var, 1, 0)
        assert h1.decisions == 1
        assert h2.decisions == 1

    def test_disable_suppresses_events(self):
        """Disabled manager doesn't fire events."""
        manager = HookManager()
        hook = CountingHook()
        manager.register(hook)
        manager.disable()

        var = Variable("x", [1, 2])
        manager.fire_decision(var, 1, 0)
        assert hook.decisions == 0

    def test_unregister(self):
        """Unregistered hooks stop receiving events."""
        manager = HookManager()
        hook = CountingHook()
        manager.register(hook)
        manager.unregister(hook)

        var = Variable("x", [1, 2])
        manager.fire_decision(var, 1, 0)
        assert hook.decisions == 0


class TestEventBus:
    def test_subscribe_and_publish(self):
        """Subscribers receive published events."""
        bus = EventBus()
        received = []
        bus.subscribe(EventType.DECISION_MADE, lambda e: received.append(e))

        event = SolverEvent(EventType.DECISION_MADE, depth=1)
        bus.publish(event)
        assert len(received) == 1

    def test_type_filtering(self):
        """Subscribers only receive their subscribed event type."""
        bus = EventBus()
        decisions = []
        backtracks = []
        bus.subscribe(EventType.DECISION_MADE, lambda e: decisions.append(e))
        bus.subscribe(EventType.BACKTRACK, lambda e: backtracks.append(e))

        bus.publish(SolverEvent(EventType.DECISION_MADE))
        bus.publish(SolverEvent(EventType.BACKTRACK))

        assert len(decisions) == 1
        assert len(backtracks) == 1

    def test_global_subscriber(self):
        """Global subscribers receive all events."""
        bus = EventBus()
        all_events = []
        bus.subscribe_all(lambda e: all_events.append(e))

        bus.publish(SolverEvent(EventType.DECISION_MADE))
        bus.publish(SolverEvent(EventType.BACKTRACK))
        bus.publish(SolverEvent(EventType.SOLUTION_FOUND))

        assert len(all_events) == 3

    def test_logging(self):
        """Event logging records events."""
        bus = EventBus()
        bus.enable_logging()
        bus.publish(SolverEvent(EventType.DECISION_MADE))
        bus.publish(SolverEvent(EventType.BACKTRACK))
        assert len(bus.log) == 2

    def test_event_count(self):
        """Event count works with type filter."""
        bus = EventBus()
        bus.enable_logging()
        bus.publish(SolverEvent(EventType.DECISION_MADE))
        bus.publish(SolverEvent(EventType.DECISION_MADE))
        bus.publish(SolverEvent(EventType.BACKTRACK))
        assert bus.event_count(EventType.DECISION_MADE) == 2
        assert bus.event_count(EventType.BACKTRACK) == 1
