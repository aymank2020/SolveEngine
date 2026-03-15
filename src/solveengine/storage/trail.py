"""Trail-based undo mechanism for domain modifications.

The trail records all domain modifications so they can be undone
efficiently during backtracking. This is more memory-efficient than
copying entire domains at each decision point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrailEntry:
    """A single modification recorded on the trail."""

    variable_index: int
    removed_value: int
    decision_level: int


class Trail:
    """Trail for recording and undoing domain modifications.

    The trail is a stack of modifications. When backtracking to a
    decision level, all modifications at or above that level are undone.
    """

    def __init__(self) -> None:
        self._entries: list[TrailEntry] = []
        self._level_markers: list[int] = []  # Index into _entries at each level
        self._current_level: int = 0

    @property
    def size(self) -> int:
        """Total number of entries on the trail."""
        return len(self._entries)

    @property
    def current_level(self) -> int:
        return self._current_level

    def new_level(self) -> int:
        """Start a new decision level. Returns the new level number."""
        self._level_markers.append(len(self._entries))
        self._current_level += 1
        return self._current_level

    def record(self, variable_index: int, removed_value: int) -> None:
        """Record a domain modification at the current level."""
        entry = TrailEntry(
            variable_index=variable_index,
            removed_value=removed_value,
            decision_level=self._current_level,
        )
        self._entries.append(entry)

    def undo_level(self) -> list[TrailEntry]:
        """Undo all modifications at the current level.

        Returns the list of undone entries (for domain restoration).
        """
        if not self._level_markers:
            return []

        marker = self._level_markers.pop()
        undone = self._entries[marker:]
        self._entries = self._entries[:marker]
        self._current_level -= 1
        return undone

    def undo_to_level(self, target_level: int) -> list[TrailEntry]:
        """Undo all modifications down to (but not including) target_level.

        Returns all undone entries.
        """
        all_undone: list[TrailEntry] = []
        while self._current_level > target_level:
            undone = self.undo_level()
            all_undone.extend(undone)
        return all_undone

    def entries_at_level(self, level: int) -> list[TrailEntry]:
        """Get all entries recorded at a specific level."""
        return [e for e in self._entries if e.decision_level == level]

    def clear(self) -> None:
        """Clear the entire trail."""
        self._entries.clear()
        self._level_markers.clear()
        self._current_level = 0

    def __len__(self) -> int:
        return len(self._entries)
