"""Phase saving for CSP solvers.

Phase saving remembers the last assigned value for each variable and uses
it as the first choice when that variable is selected again after a restart.
This technique, borrowed from SAT solving (CDCL solvers), helps the solver
quickly reconstruct promising partial assignments after restarts.

The phase history also provides useful information for adaptive heuristics
by tracking how often each variable takes each value across restarts.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.heuristics.value_ordering import ValueOrderer


class PhaseRecord:
    """Records phase information for a single variable.

    Tracks the last assigned value, assignment frequency per value,
    and the number of times each value led to a conflict.
    """

    __slots__ = ("_variable", "_last_value", "_value_counts", "_conflict_counts", "_total_assigns")

    def __init__(self, variable: Variable) -> None:
        self._variable = variable
        self._last_value: int | None = None
        self._value_counts: dict[int, int] = defaultdict(int)
        self._conflict_counts: dict[int, int] = defaultdict(int)
        self._total_assigns: int = 0

    @property
    def variable(self) -> Variable:
        return self._variable

    @property
    def last_value(self) -> int | None:
        """The most recently assigned value for this variable."""
        return self._last_value

    @property
    def total_assignments(self) -> int:
        """Total number of times this variable has been assigned."""
        return self._total_assigns

    def record_assignment(self, value: int) -> None:
        """Record that the variable was assigned this value."""
        self._last_value = value
        self._value_counts[value] += 1
        self._total_assigns += 1

    def record_conflict(self, value: int) -> None:
        """Record that assigning this value led to a conflict."""
        self._conflict_counts[value] += 1

    def assignment_frequency(self, value: int) -> float:
        """Fraction of assignments that used this value."""
        if self._total_assigns == 0:
            return 0.0
        return self._value_counts[value] / self._total_assigns

    def conflict_rate(self, value: int) -> float:
        """Fraction of assignments of this value that led to conflict."""
        assigns = self._value_counts[value]
        if assigns == 0:
            return 0.0
        return self._conflict_counts[value] / assigns

    def best_value(self) -> int | None:
        """Return the value with the best success rate (lowest conflict rate).

        Among values that have been tried, returns the one with the lowest
        conflict rate. Returns None if no values have been tried.
        """
        if not self._value_counts:
            return self._last_value

        candidates = [
            (val, self.conflict_rate(val), -count)
            for val, count in self._value_counts.items()
        ]
        if not candidates:
            return self._last_value

        candidates.sort(key=lambda x: (x[1], x[2]))
        return candidates[0][0]

    def reset_counts(self) -> None:
        """Reset frequency counts but preserve last value."""
        self._value_counts.clear()
        self._conflict_counts.clear()
        self._total_assigns = 0


class PhaseSaving:
    """Phase saving manager for CSP solvers.

    Maintains phase records for all variables and provides value ordering
    that prioritizes the saved phase (last assigned value) after restarts.

    Usage:
        phase_saver = PhaseSaving(variables)
        # During search, record assignments:
        phase_saver.record_assignment(var, value)
        # On conflict:
        phase_saver.record_conflict(var, value)
        # After restart, get phase-guided value ordering:
        orderer = phase_saver.create_value_orderer()
    """

    def __init__(self, variables: Sequence[Variable]) -> None:
        self._records: dict[Variable, PhaseRecord] = {}
        for var in variables:
            self._records[var] = PhaseRecord(var)
        self._restart_count: int = 0
        self._total_conflicts: int = 0
        self._phase_hits: int = 0
        self._phase_misses: int = 0

    @property
    def restart_count(self) -> int:
        """Number of restarts recorded."""
        return self._restart_count

    @property
    def total_conflicts(self) -> int:
        """Total conflicts recorded across all variables."""
        return self._total_conflicts

    @property
    def phase_hit_rate(self) -> float:
        """Fraction of assignments that used the saved phase value."""
        total = self._phase_hits + self._phase_misses
        if total == 0:
            return 0.0
        return self._phase_hits / total

    def record_assignment(self, var: Variable, value: int) -> None:
        """Record that var was assigned value during search."""
        record = self._records.get(var)
        if record is None:
            record = PhaseRecord(var)
            self._records[var] = record

        if record.last_value == value:
            self._phase_hits += 1
        elif record.last_value is not None:
            self._phase_misses += 1

        record.record_assignment(value)

    def record_conflict(self, var: Variable, value: int) -> None:
        """Record that assigning value to var led to a conflict."""
        record = self._records.get(var)
        if record is not None:
            record.record_conflict(value)
        self._total_conflicts += 1

    def record_restart(self) -> None:
        """Record that a restart occurred."""
        self._restart_count += 1

    def get_phase_value(self, var: Variable) -> int | None:
        """Get the saved phase value for a variable.

        Returns the last assigned value, or None if never assigned.
        """
        record = self._records.get(var)
        if record is None:
            return None
        return record.last_value

    def get_best_value(self, var: Variable) -> int | None:
        """Get the best historical value for a variable.

        Returns the value with the lowest conflict rate among those tried.
        """
        record = self._records.get(var)
        if record is None:
            return None
        return record.best_value()

    def get_value_ordering(self, var: Variable) -> list[int]:
        """Get phase-guided value ordering for a variable.

        Orders values with the saved phase value first, then by
        historical success rate, then by ascending value.
        """
        record = self._records.get(var)
        domain_values = sorted(var.domain)

        if record is None or record.last_value is None:
            return domain_values

        phase_val = record.last_value

        def sort_key(value: int) -> tuple[int, float, int]:
            if value == phase_val:
                priority = 0
            else:
                priority = 1
            conflict_rate = record.conflict_rate(value)
            return (priority, conflict_rate, value)

        domain_values.sort(key=sort_key)
        return domain_values

    def create_value_orderer(self) -> PhaseValueOrderer:
        """Create a ValueOrderer that uses phase saving for ordering."""
        return PhaseValueOrderer(self)

    def reset_all(self) -> None:
        """Reset all phase records (full reset)."""
        for record in self._records.values():
            record.reset_counts()
        self._restart_count = 0
        self._total_conflicts = 0
        self._phase_hits = 0
        self._phase_misses = 0

    def get_statistics(self) -> dict[str, float]:
        """Return summary statistics about phase saving effectiveness."""
        total_assigns = sum(r.total_assignments for r in self._records.values())
        vars_with_phase = sum(1 for r in self._records.values() if r.last_value is not None)
        avg_conflict_rate = 0.0
        if self._records:
            rates = []
            for record in self._records.values():
                if record.last_value is not None:
                    rates.append(record.conflict_rate(record.last_value))
            if rates:
                avg_conflict_rate = sum(rates) / len(rates)

        return {
            "total_assignments": float(total_assigns),
            "variables_with_phase": float(vars_with_phase),
            "phase_hit_rate": self.phase_hit_rate,
            "avg_phase_conflict_rate": avg_conflict_rate,
            "restarts": float(self._restart_count),
        }


class PhaseValueOrderer(ValueOrderer):
    """Value orderer that uses phase saving to guide value selection.

    Places the saved phase value first, then orders remaining values
    by their historical conflict rate (lower conflict rate first).
    """

    def __init__(self, phase_saver: PhaseSaving) -> None:
        self._phase_saver = phase_saver

    def order(
        self,
        var: Variable,
        constraints: list[Constraint],
        assignment: dict[Variable, int],
    ) -> list[int]:
        """Return values ordered by phase saving heuristic."""
        return self._phase_saver.get_value_ordering(var)
