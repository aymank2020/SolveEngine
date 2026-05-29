"""Structured logging hook for solver events.

Provides configurable logging of solver events with support for
multiple output targets (file, stream, structured JSON). Useful for
debugging solver behavior, analyzing search patterns, and generating
solve traces for visualization.

Verbosity levels:
- QUIET: Only log solutions and final statistics
- NORMAL: Log decisions, backtracks, and restarts
- VERBOSE: Log propagation details and domain changes
- DEBUG: Log everything including constraint checks
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import IO, TextIO

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.middleware.hooks import SolverHook


class Verbosity(IntEnum):
    """Logging verbosity levels."""

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


@dataclass
class LogEntry:
    """A single structured log entry."""

    timestamp: float
    event_type: str
    depth: int
    message: str
    data: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "event": self.event_type,
            "depth": self.depth,
            "message": self.message,
            "data": self.data,
        }

    def format_text(self) -> str:
        """Format as human-readable text."""
        indent = "  " * self.depth
        elapsed = f"[{self.timestamp:.4f}s]"
        return f"{elapsed} {indent}{self.event_type}: {self.message}"


class LoggingHook(SolverHook):
    """Structured logging hook for solver events.

    Captures solver events and outputs them in configurable format
    to configurable destinations.

    Args:
        verbosity: Minimum verbosity level for logging.
        output: Output stream (defaults to stdout-like StringIO).
        format: Output format ("text" or "json").
        max_entries: Maximum log entries to keep in memory.
        log_file: Optional file path to write logs to.
    """

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        output: TextIO | None = None,
        format: str = "text",
        max_entries: int = 50000,
        log_file: str | None = None,
    ) -> None:
        self._verbosity = verbosity
        self._output = output
        self._format = format
        self._max_entries = max_entries
        self._entries: list[LogEntry] = []
        self._start_time: float = 0.0
        self._file_handle: IO[str] | None = None
        self._decision_count: int = 0
        self._backtrack_count: int = 0
        self._solution_count: int = 0
        self._propagation_count: int = 0
        self._restart_count: int = 0
        self._wipeout_count: int = 0

        if log_file is not None:
            self._file_handle = open(log_file, "w", encoding="utf-8")

    @property
    def verbosity(self) -> Verbosity:
        return self._verbosity

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def decision_count(self) -> int:
        return self._decision_count

    @property
    def backtrack_count(self) -> int:
        return self._backtrack_count

    @property
    def solution_count(self) -> int:
        return self._solution_count

    def set_verbosity(self, level: Verbosity) -> None:
        """Change verbosity level."""
        self._verbosity = level

    def on_start(self, num_variables: int, num_constraints: int) -> None:
        """Log solver start."""
        self._start_time = time.perf_counter()
        self._log(
            Verbosity.QUIET,
            "start",
            0,
            f"Solving: {num_variables} variables, {num_constraints} constraints",
            {"num_variables": num_variables, "num_constraints": num_constraints},
        )

    def on_finish(self, solved: bool, nodes: int) -> None:
        """Log solver finish with summary statistics."""
        elapsed = time.perf_counter() - self._start_time
        status = "SOLVED" if solved else "UNSATISFIABLE"
        self._log(
            Verbosity.QUIET,
            "finish",
            0,
            f"{status} in {elapsed:.4f}s ({nodes} nodes)",
            {
                "solved": solved,
                "nodes": nodes,
                "elapsed": elapsed,
                "decisions": self._decision_count,
                "backtracks": self._backtrack_count,
                "solutions": self._solution_count,
                "restarts": self._restart_count,
                "wipeouts": self._wipeout_count,
            },
        )

    def on_decision(self, var: Variable, value: int, depth: int) -> None:
        """Log a variable assignment decision."""
        self._decision_count += 1
        self._log(
            Verbosity.NORMAL,
            "decision",
            depth,
            f"{var.name} = {value} (dom_size={var.domain_size})",
            {"variable": var.name, "value": value, "domain_size": var.domain_size},
        )

    def on_backtrack(self, var: Variable, depth: int) -> None:
        """Log a backtrack event."""
        self._backtrack_count += 1
        self._log(
            Verbosity.NORMAL,
            "backtrack",
            depth,
            f"Backtrack from {var.name} at depth {depth}",
            {"variable": var.name, "depth": depth},
        )

    def on_propagation(self, pruned_count: int, depth: int) -> None:
        """Log propagation results."""
        self._propagation_count += 1
        self._log(
            Verbosity.VERBOSE,
            "propagation",
            depth,
            f"Pruned {pruned_count} values",
            {"pruned_count": pruned_count},
        )

    def on_wipeout(self, var: Variable, constraint: Constraint, depth: int) -> None:
        """Log a domain wipeout."""
        self._wipeout_count += 1
        self._log(
            Verbosity.NORMAL,
            "wipeout",
            depth,
            f"Domain wipeout: {var.name} by {constraint.name}",
            {"variable": var.name, "constraint": constraint.name},
        )

    def on_solution(self, assignment: dict[Variable, int]) -> None:
        """Log a solution found."""
        self._solution_count += 1
        solution_str = ", ".join(
            f"{v.name}={val}" for v, val in sorted(assignment.items(), key=lambda x: x[0].index)
        )
        self._log(
            Verbosity.QUIET,
            "solution",
            0,
            f"Solution #{self._solution_count}: {solution_str}",
            {"solution_number": self._solution_count, "assignment_size": len(assignment)},
        )

    def on_restart(self, restart_number: int, nodes_explored: int) -> None:
        """Log a restart event."""
        self._restart_count += 1
        self._log(
            Verbosity.NORMAL,
            "restart",
            0,
            f"Restart #{restart_number} after {nodes_explored} nodes",
            {"restart_number": restart_number, "nodes_explored": nodes_explored},
        )

    def _log(
        self,
        min_verbosity: Verbosity,
        event_type: str,
        depth: int,
        message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        """Create and store a log entry if verbosity allows."""
        if self._verbosity < min_verbosity:
            return

        elapsed = time.perf_counter() - self._start_time if self._start_time > 0 else 0.0
        entry = LogEntry(
            timestamp=elapsed,
            event_type=event_type,
            depth=depth,
            message=message,
            data=data or {},
        )

        if len(self._entries) < self._max_entries:
            self._entries.append(entry)

        self._emit(entry)

    def _emit(self, entry: LogEntry) -> None:
        """Write a log entry to the configured output."""
        if self._format == "json":
            line = json.dumps(entry.to_dict())
        else:
            line = entry.format_text()

        if self._output is not None:
            self._output.write(line + "\n")
            self._output.flush()

        if self._file_handle is not None:
            self._file_handle.write(line + "\n")
            self._file_handle.flush()

    def get_entries(self, event_type: str | None = None) -> list[LogEntry]:
        """Get log entries, optionally filtered by event type."""
        if event_type is None:
            return list(self._entries)
        return [e for e in self._entries if e.event_type == event_type]

    def get_summary(self) -> dict[str, int]:
        """Get summary counts of all event types."""
        return {
            "decisions": self._decision_count,
            "backtracks": self._backtrack_count,
            "solutions": self._solution_count,
            "propagations": self._propagation_count,
            "restarts": self._restart_count,
            "wipeouts": self._wipeout_count,
            "total_entries": len(self._entries),
        }

    def get_decision_trace(self) -> list[tuple[str, int, int]]:
        """Get the sequence of decisions as (variable, value, depth) tuples."""
        trace = []
        for entry in self._entries:
            if entry.event_type == "decision":
                var_name = entry.data.get("variable", "")
                value = entry.data.get("value", 0)
                trace.append((str(var_name), int(value), entry.depth))  # type: ignore[arg-type]
        return trace

    def clear(self) -> None:
        """Clear all log entries and reset counters."""
        self._entries.clear()
        self._decision_count = 0
        self._backtrack_count = 0
        self._solution_count = 0
        self._propagation_count = 0
        self._restart_count = 0
        self._wipeout_count = 0

    def close(self) -> None:
        """Close file handle if open."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self) -> None:
        self.close()
