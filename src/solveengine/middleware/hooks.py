"""Hook system for extending solver behavior.

Hooks allow external code to observe and modify solver behavior at
key points: before/after decisions, propagation, backtracks, and solutions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class SolverHook(ABC):
    """Abstract base for solver hooks.

    Implement any subset of the hook methods to observe or modify
    solver behavior at specific points.
    """

    def on_decision(self, var: Variable, value: int, depth: int) -> None:
        """Called after a variable is assigned."""
        pass

    def on_backtrack(self, var: Variable, depth: int) -> None:
        """Called when backtracking from a variable."""
        pass

    def on_propagation(self, pruned_count: int, depth: int) -> None:
        """Called after propagation completes."""
        pass

    def on_wipeout(self, var: Variable, constraint: Constraint, depth: int) -> None:
        """Called when a domain wipeout occurs."""
        pass

    def on_solution(self, assignment: dict[Variable, int]) -> None:
        """Called when a solution is found."""
        pass

    def on_restart(self, restart_number: int, nodes_explored: int) -> None:
        """Called when the solver restarts."""
        pass

    def on_start(self, num_variables: int, num_constraints: int) -> None:
        """Called when solving begins."""
        pass

    def on_finish(self, solved: bool, nodes: int) -> None:
        """Called when solving ends."""
        pass


class HookManager:
    """Manages a collection of solver hooks.

    Dispatches events to all registered hooks in order.
    Hooks are called synchronously and should be lightweight.
    """

    def __init__(self) -> None:
        self._hooks: list[SolverHook] = []
        self._enabled: bool = True

    @property
    def num_hooks(self) -> int:
        return len(self._hooks)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def register(self, hook: SolverHook) -> None:
        """Register a hook to receive solver events."""
        self._hooks.append(hook)

    def unregister(self, hook: SolverHook) -> None:
        """Remove a previously registered hook."""
        self._hooks = [h for h in self._hooks if h is not hook]

    def clear(self) -> None:
        """Remove all hooks."""
        self._hooks.clear()

    def fire_decision(self, var: Variable, value: int, depth: int) -> None:
        if not self._enabled:
            return
        for hook in self._hooks:
            hook.on_decision(var, value, depth)

    def fire_backtrack(self, var: Variable, depth: int) -> None:
        if not self._enabled:
            return
        for hook in self._hooks:
            hook.on_backtrack(var, depth)

    def fire_propagation(self, pruned_count: int, depth: int) -> None:
        if not self._enabled:
            return
        for hook in self._hooks:
            hook.on_propagation(pruned_count, depth)

    def fire_wipeout(self, var: Variable, constraint: Constraint, depth: int) -> None:
        if not self._enabled:
            return
        for hook in self._hooks:
            hook.on_wipeout(var, constraint, depth)

    def fire_solution(self, assignment: dict[Variable, int]) -> None:
        if not self._enabled:
            return
        for hook in self._hooks:
            hook.on_solution(assignment)

    def fire_restart(self, restart_number: int, nodes_explored: int) -> None:
        if not self._enabled:
            return
        for hook in self._hooks:
            hook.on_restart(restart_number, nodes_explored)

    def fire_start(self, num_variables: int, num_constraints: int) -> None:
        if not self._enabled:
            return
        for hook in self._hooks:
            hook.on_start(num_variables, num_constraints)

    def fire_finish(self, solved: bool, nodes: int) -> None:
        if not self._enabled:
            return
        for hook in self._hooks:
            hook.on_finish(solved, nodes)
