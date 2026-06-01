"""Solver profiler for performance analysis.

Tracks time spent in each solver component (propagation, search, heuristics)
and identifies bottlenecks. Provides detailed timing breakdowns and
performance ratios to guide optimization efforts.

Usage:
    profiler = SolverProfiler()
    with profiler.measure("propagation"):
        propagator.propagate(...)
    with profiler.measure("search"):
        solver.search(...)
    print(profiler.report())
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.middleware.hooks import SolverHook


@dataclass
class TimingEntry:
    """Accumulated timing data for a single component."""

    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_time: float = 0.0

    @property
    def avg_time(self) -> float:
        """Average time per call."""
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count

    def record(self, elapsed: float) -> None:
        """Record a timing measurement."""
        self.total_time += elapsed
        self.call_count += 1
        self.last_time = elapsed
        if elapsed < self.min_time:
            self.min_time = elapsed
        if elapsed > self.max_time:
            self.max_time = elapsed


@dataclass
class ProfileSnapshot:
    """A snapshot of profiler state at a point in time."""

    timestamp: float
    component: str
    elapsed: float
    cumulative: float
    depth: int


class SolverProfiler:
    """Profiles solver performance by component.

    Tracks time spent in propagation, search, heuristic evaluation,
    and other solver components. Identifies bottlenecks and provides
    actionable performance insights.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._timings: dict[str, TimingEntry] = {}
        self._active_timers: list[tuple[str, float]] = []
        self._snapshots: list[ProfileSnapshot] = []
        self._start_time: float = 0.0
        self._total_time: float = 0.0
        self._max_snapshots: int = 10000

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def total_time(self) -> float:
        """Total profiled time in seconds."""
        return self._total_time

    @property
    def components(self) -> list[str]:
        """List of profiled component names."""
        return list(self._timings.keys())

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @contextmanager
    def measure(self, component: str) -> Generator[None, None, None]:
        """Context manager to measure time spent in a component.

        Usage:
            with profiler.measure("propagation"):
                do_propagation()
        """
        if not self._enabled:
            yield
            return

        start = time.perf_counter()
        self._active_timers.append((component, start))
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._active_timers.pop()
            self._record_timing(component, elapsed)

    def start_timer(self, component: str) -> float:
        """Manually start a timer. Returns the start timestamp."""
        if not self._enabled:
            return 0.0
        start = time.perf_counter()
        self._active_timers.append((component, start))
        return start

    def stop_timer(self, component: str) -> float:
        """Manually stop a timer. Returns elapsed time."""
        if not self._enabled:
            return 0.0

        now = time.perf_counter()
        for i in range(len(self._active_timers) - 1, -1, -1):
            if self._active_timers[i][0] == component:
                _, start = self._active_timers.pop(i)
                elapsed = now - start
                self._record_timing(component, elapsed)
                return elapsed
        return 0.0

    def _record_timing(self, component: str, elapsed: float) -> None:
        """Record a timing measurement for a component."""
        if component not in self._timings:
            self._timings[component] = TimingEntry(name=component)

        entry = self._timings[component]
        entry.record(elapsed)
        self._total_time += elapsed

        if len(self._snapshots) < self._max_snapshots:
            self._snapshots.append(ProfileSnapshot(
                timestamp=time.perf_counter(),
                component=component,
                elapsed=elapsed,
                cumulative=entry.total_time,
                depth=len(self._active_timers),
            ))

    def get_timing(self, component: str) -> TimingEntry | None:
        """Get timing data for a specific component."""
        return self._timings.get(component)

    def get_time_fraction(self, component: str) -> float:
        """Get fraction of total time spent in a component."""
        if self._total_time == 0:
            return 0.0
        entry = self._timings.get(component)
        if entry is None:
            return 0.0
        return entry.total_time / self._total_time

    def get_propagation_search_ratio(self) -> float:
        """Get the ratio of propagation time to search time.

        A high ratio (>5) suggests propagation is the bottleneck.
        A low ratio (<1) suggests the search heuristic may need improvement.
        """
        prop_time = self._timings.get("propagation")
        search_time = self._timings.get("search")

        prop_total = prop_time.total_time if prop_time else 0.0
        search_total = search_time.total_time if search_time else 0.0

        if search_total == 0:
            return float('inf') if prop_total > 0 else 0.0
        return prop_total / search_total

    def get_bottleneck(self) -> str | None:
        """Identify the component consuming the most time."""
        if not self._timings:
            return None
        return max(self._timings.values(), key=lambda e: e.total_time).name

    def get_top_components(self, k: int = 5) -> list[tuple[str, float, float]]:
        """Get top k components by time spent.

        Returns list of (name, total_time, fraction) tuples.
        """
        entries = sorted(
            self._timings.values(),
            key=lambda e: e.total_time,
            reverse=True,
        )[:k]

        result = []
        for entry in entries:
            fraction = entry.total_time / self._total_time if self._total_time > 0 else 0.0
            result.append((entry.name, entry.total_time, fraction))
        return result

    def report(self) -> str:
        """Generate a human-readable performance report."""
        lines = ["=" * 60, "Solver Performance Profile", "=" * 60]
        lines.append(f"Total profiled time: {self._total_time:.4f}s")
        lines.append("")

        if not self._timings:
            lines.append("No timing data collected.")
            return "\n".join(lines)

        lines.append(f"{'Component':<20} {'Total':>10} {'Calls':>8} {'Avg':>10} {'%':>6}")
        lines.append("-" * 60)

        sorted_entries = sorted(
            self._timings.values(),
            key=lambda e: e.total_time,
            reverse=True,
        )

        for entry in sorted_entries:
            fraction = (entry.total_time / self._total_time * 100) if self._total_time > 0 else 0
            lines.append(
                f"{entry.name:<20} {entry.total_time:>10.4f}s "
                f"{entry.call_count:>7}  {entry.avg_time:>9.6f}s {fraction:>5.1f}%"
            )

        lines.append("")
        lines.append("Key Ratios:")
        prop_search = self.get_propagation_search_ratio()
        lines.append(f"  Propagation/Search ratio: {prop_search:.2f}")

        bottleneck = self.get_bottleneck()
        if bottleneck:
            lines.append(f"  Bottleneck: {bottleneck}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all profiling data."""
        self._timings.clear()
        self._active_timers.clear()
        self._snapshots.clear()
        self._total_time = 0.0

    def create_hook(self) -> ProfilerHook:
        """Create a SolverHook that automatically profiles solver events."""
        return ProfilerHook(self)


class ProfilerHook(SolverHook):
    """Solver hook that automatically records timing for solver events.

    Integrates with the hook system to profile propagation, decisions,
    and backtracks without modifying solver code.
    """

    def __init__(self, profiler: SolverProfiler) -> None:
        self._profiler = profiler
        self._decision_start: float = 0.0
        self._propagation_start: float = 0.0
        self._solve_start: float = 0.0
        self._decision_count: int = 0
        self._backtrack_count: int = 0

    @property
    def decision_count(self) -> int:
        return self._decision_count

    @property
    def backtrack_count(self) -> int:
        return self._backtrack_count

    def on_start(self, num_variables: int, num_constraints: int) -> None:
        self._solve_start = time.perf_counter()
        self._profiler.start_timer("total_solve")

    def on_finish(self, solved: bool, nodes: int) -> None:
        self._profiler.stop_timer("total_solve")

    def on_decision(self, var: Variable, value: int, depth: int) -> None:
        self._decision_count += 1
        self._decision_start = time.perf_counter()

    def on_backtrack(self, var: Variable, depth: int) -> None:
        self._backtrack_count += 1
        if self._decision_start > 0:
            elapsed = time.perf_counter() - self._decision_start
            self._profiler._record_timing("decision_to_backtrack", elapsed)

    def on_propagation(self, pruned_count: int, depth: int) -> None:
        self._profiler._record_timing("propagation_event", 0.0)

    def on_wipeout(self, var: Variable, constraint: Constraint, depth: int) -> None:
        self._profiler._record_timing("wipeout_event", 0.0)

    def on_restart(self, restart_number: int, nodes_explored: int) -> None:
        self._profiler._record_timing("restart_event", 0.0)
