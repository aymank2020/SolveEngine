"""Checkpoint management for solver state.

Allows saving and restoring solver state at arbitrary points during search.
Useful for implementing restarts, portfolio solving, and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from solveengine.core.variable import Variable


@dataclass
class Checkpoint:
    """A snapshot of solver state at a point in time."""

    checkpoint_id: int
    depth: int
    assignment: dict[int, int]  # var_index -> value
    domain_states: dict[int, frozenset[int]]  # var_index -> domain values
    generations: dict[int, int]  # var_index -> generation
    stats_snapshot: dict[str, int] = field(default_factory=dict)

    @property
    def num_assigned(self) -> int:
        return len(self.assignment)


class CheckpointManager:
    """Manages checkpoints for solver state save/restore.

    Supports multiple named checkpoints with automatic cleanup
    of stale checkpoints.
    """

    def __init__(self, variables: list[Variable], max_checkpoints: int = 100) -> None:
        self._variables = variables
        self._max_checkpoints = max_checkpoints
        self._checkpoints: dict[int, Checkpoint] = {}
        self._next_id: int = 0

    @property
    def num_checkpoints(self) -> int:
        return len(self._checkpoints)

    def save(self, assignment: dict[Variable, int], depth: int) -> int:
        """Save current state as a checkpoint. Returns checkpoint ID."""
        if len(self._checkpoints) >= self._max_checkpoints:
            # Remove oldest checkpoint
            oldest_id = min(self._checkpoints.keys())
            del self._checkpoints[oldest_id]

        cp_id = self._next_id
        self._next_id += 1

        domain_states = {
            var.index: var.domain.values() for var in self._variables
        }
        generations = {
            var.index: var.domain.generation for var in self._variables
        }
        assignment_snapshot = {
            var.index: val for var, val in assignment.items()
        }

        checkpoint = Checkpoint(
            checkpoint_id=cp_id,
            depth=depth,
            assignment=assignment_snapshot,
            domain_states=domain_states,
            generations=generations,
        )
        self._checkpoints[cp_id] = checkpoint
        return cp_id

    def restore(self, checkpoint_id: int) -> dict[int, int]:
        """Restore solver state from a checkpoint.

        Restores domain states for all variables.
        Returns the assignment at checkpoint time (var_index -> value).
        """
        if checkpoint_id not in self._checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        cp = self._checkpoints[checkpoint_id]

        # Restore domains
        for var in self._variables:
            target_gen = cp.generations.get(var.index, 0)
            var.restore_to(target_gen)
            var.unassign()

            # Re-assign if was assigned at checkpoint
            if var.index in cp.assignment:
                val = cp.assignment[var.index]
                if var.domain.contains(val):
                    var.assign(val)

        return dict(cp.assignment)

    def get(self, checkpoint_id: int) -> Checkpoint | None:
        """Get a checkpoint by ID without restoring."""
        return self._checkpoints.get(checkpoint_id)

    def discard(self, checkpoint_id: int) -> bool:
        """Remove a checkpoint. Returns True if it existed."""
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]
            return True
        return False

    def discard_after(self, checkpoint_id: int) -> int:
        """Remove all checkpoints created after the given ID.

        Returns the number of checkpoints removed.
        """
        to_remove = [cid for cid in self._checkpoints if cid > checkpoint_id]
        for cid in to_remove:
            del self._checkpoints[cid]
        return len(to_remove)

    def clear(self) -> None:
        """Remove all checkpoints."""
        self._checkpoints.clear()
