"""Middleware: hooks, plugins, and event system for solver extensibility."""

from solveengine.middleware.hooks import SolverHook, HookManager
from solveengine.middleware.events import SolverEvent, EventBus

__all__ = ["SolverHook", "HookManager", "SolverEvent", "EventBus"]
