"""Assignment management for constraint satisfaction problems.

Provides a rich assignment container that goes beyond a simple dictionary.
Tracks partial assignments with full undo support, computes assignment
hashes for caching/memoization, and maintains a history of all assignments
made during search for analysis and debugging.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from solveengine.core.variable import Variable


@dataclass
class AssignmentEvent:
    """Records a single assignment or unassignment event."""

    variable: Variable
    value: int | None
    is_assign: bool
    timestamp: int
    depth: int

    @property
    def is_unassign(self) -> bool:
        return not self.is_assign


class Assignment:
    """Manages partial assignments for CSP solving.

    Provides:
    - O(1) assignment/unassignment with undo stack
    - Hash computation for caching partial assignments
    - Full history tracking for analysis
    - Snapshot/restore for branching

    Args:
        variables: The complete set of CSP variables.
        track_history: Whether to record all assignment events.
    """

    def __init__(
        self, variables: Sequence[Variable], track_history: bool = True
    ) -> None:
        self._variables = list(variables)
        self._var_set = set(variables)
        self._mapping: dict[Variable, int] = {}
        self._undo_stack: list[tuple[Variable, int | None]] = []
        self._history: list[AssignmentEvent] = [] if track_history else []
        self._track_history = track_history
        self._timestamp: int = 0
        self._depth: int = 0
        self._hash_cache: int | None = None
        self._hash_valid: bool = False

    @property
    def size(self) -> int:
        """Number of currently assigned variables."""
        return len(self._mapping)

    @property
    def is_complete(self) -> bool:
        """Whether all variables are assigned."""
        return len(self._mapping) == len(self._variables)

    @property
    def is_empty(self) -> bool:
        """Whether no variables are assigned."""
        return len(self._mapping) == 0

    @property
    def depth(self) -> int:
        """Current search depth (number of assignments on the undo stack)."""
        return self._depth

    @property
    def num_variables(self) -> int:
        """Total number of variables in the problem."""
        return len(self._variables)

    @property
    def history_length(self) -> int:
        """Number of events in the history."""
        return len(self._history)

    def assign(self, var: Variable, value: int) -> None:
        """Assign a value to a variable.

        Pushes the previous state onto the undo stack.
        Raises ValueError if the variable is already assigned.
        """
        if var in self._mapping:
            raise ValueError(
                f"Variable '{var.name}' is already assigned to {self._mapping[var]}"
            )
        if var not in self._var_set:
            raise ValueError(f"Variable '{var.name}' is not part of this assignment")

        previous = self._mapping.get(var)
        self._undo_stack.append((var, previous))
        self._mapping[var] = value
        self._depth += 1
        self._hash_valid = False
        self._timestamp += 1

        if self._track_history:
            self._history.append(AssignmentEvent(
                variable=var,
                value=value,
                is_assign=True,
                timestamp=self._timestamp,
                depth=self._depth,
            ))

    def unassign(self, var: Variable) -> int | None:
        """Remove the assignment for a variable.

        Returns the previously assigned value, or None if not assigned.
        Does NOT use the undo stack (for manual unassignment).
        """
        value = self._mapping.pop(var, None)
        self._hash_valid = False
        self._timestamp += 1

        if self._track_history and value is not None:
            self._history.append(AssignmentEvent(
                variable=var,
                value=value,
                is_assign=False,
                timestamp=self._timestamp,
                depth=self._depth,
            ))

        return value

    def undo(self) -> tuple[Variable, int] | None:
        """Undo the most recent assignment.

        Returns the (variable, value) that was undone, or None if stack is empty.
        """
        if not self._undo_stack:
            return None

        var, previous = self._undo_stack.pop()
        current_value = self._mapping.get(var)

        if previous is None:
            self._mapping.pop(var, None)
        else:
            self._mapping[var] = previous

        self._depth -= 1
        self._hash_valid = False
        self._timestamp += 1

        if self._track_history and current_value is not None:
            self._history.append(AssignmentEvent(
                variable=var,
                value=current_value,
                is_assign=False,
                timestamp=self._timestamp,
                depth=self._depth,
            ))

        if current_value is not None:
            return (var, current_value)
        return None

    def undo_to_depth(self, target_depth: int) -> list[tuple[Variable, int]]:
        """Undo assignments until reaching the target depth.

        Returns list of (variable, value) pairs that were undone.
        """
        undone: list[tuple[Variable, int]] = []
        while self._depth > target_depth and self._undo_stack:
            result = self.undo()
            if result is not None:
                undone.append(result)
        return undone

    def get(self, var: Variable) -> int | None:
        """Get the assigned value for a variable, or None if unassigned."""
        return self._mapping.get(var)

    def contains(self, var: Variable) -> bool:
        """Check if a variable is currently assigned."""
        return var in self._mapping

    def assigned_variables(self) -> list[Variable]:
        """Return list of currently assigned variables."""
        return [v for v in self._variables if v in self._mapping]

    def unassigned_variables(self) -> list[Variable]:
        """Return list of currently unassigned variables."""
        return [v for v in self._variables if v not in self._mapping]

    def to_dict(self) -> dict[Variable, int]:
        """Return a copy of the current assignment as a dictionary."""
        return dict(self._mapping)

    def compute_hash(self) -> int:
        """Compute a hash of the current partial assignment.

        The hash is deterministic and depends only on which variables
        are assigned and their values (not on assignment order).
        Uses a cached value when the assignment hasn't changed.
        """
        if self._hash_valid and self._hash_cache is not None:
            return self._hash_cache

        items = sorted(
            ((var.index, val) for var, val in self._mapping.items()),
            key=lambda x: x[0],
        )
        hash_input = "|".join(f"{idx}:{val}" for idx, val in items)
        self._hash_cache = int(
            hashlib.md5(hash_input.encode()).hexdigest()[:16], 16
        )
        self._hash_valid = True
        return self._hash_cache

    def fingerprint(self) -> str:
        """Return a compact string fingerprint of the assignment.

        Useful for logging and debugging.
        """
        items = sorted(
            ((var.index, var.name, val) for var, val in self._mapping.items()),
            key=lambda x: x[0],
        )
        parts = [f"{name}={val}" for _, name, val in items]
        return "{" + ", ".join(parts) + "}"

    def snapshot(self) -> AssignmentSnapshot:
        """Create a snapshot of the current state for later restoration."""
        return AssignmentSnapshot(
            mapping=dict(self._mapping),
            depth=self._depth,
            undo_stack_size=len(self._undo_stack),
            timestamp=self._timestamp,
        )

    def restore(self, snapshot: AssignmentSnapshot) -> None:
        """Restore state from a snapshot.

        Undoes assignments back to the snapshot state.
        """
        self.undo_to_depth(snapshot.depth)
        self._mapping = dict(snapshot.mapping)
        self._undo_stack = self._undo_stack[:snapshot.undo_stack_size]
        self._depth = snapshot.depth
        self._hash_valid = False

    def get_history_for_variable(self, var: Variable) -> list[AssignmentEvent]:
        """Get all history events for a specific variable."""
        return [e for e in self._history if e.variable is var]

    def get_assignment_count(self, var: Variable) -> int:
        """Count how many times a variable has been assigned in history."""
        return sum(
            1 for e in self._history
            if e.variable is var and e.is_assign
        )

    def get_value_frequency(self, var: Variable) -> dict[int, int]:
        """Get frequency of each value assigned to a variable."""
        freq: dict[int, int] = {}
        for event in self._history:
            if event.variable is var and event.is_assign and event.value is not None:
                freq[event.value] = freq.get(event.value, 0) + 1
        return freq

    def clear(self) -> None:
        """Clear all assignments and undo stack."""
        self._mapping.clear()
        self._undo_stack.clear()
        self._depth = 0
        self._hash_valid = False
        self._timestamp += 1

    def clear_history(self) -> None:
        """Clear the history log (keeps current assignment)."""
        self._history.clear()

    def __contains__(self, var: Variable) -> bool:
        return var in self._mapping

    def __getitem__(self, var: Variable) -> int:
        if var not in self._mapping:
            raise KeyError(f"Variable '{var.name}' is not assigned")
        return self._mapping[var]

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self) -> Iterator[Variable]:
        return iter(self._mapping)

    def __repr__(self) -> str:
        return f"Assignment({self.size}/{self.num_variables}, depth={self._depth})"


@dataclass
class AssignmentSnapshot:
    """Immutable snapshot of an assignment state."""

    mapping: dict[Variable, int]
    depth: int
    undo_stack_size: int
    timestamp: int

    @property
    def size(self) -> int:
        return len(self.mapping)

    def __repr__(self) -> str:
        return f"AssignmentSnapshot(size={self.size}, depth={self.depth})"
