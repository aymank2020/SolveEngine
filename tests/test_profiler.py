"""Tests for the solver profiler.

Tests cover:
- Timing accuracy and measurement
- Component tracking
- Bottleneck detection
- Report generation
- ProfilerHook integration
- Enable/disable behavior
"""

from __future__ import annotations

import time

import pytest

from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint, Constraint
from solveengine.debug.profiler import (
    SolverProfiler,
    ProfilerHook,
    TimingEntry,
    ProfileSnapshot,
)


class TestTimingEntry:
    """Tests for TimingEntry data class."""

    def test_initial_state(self) -> None:
        """New TimingEntry should have zero values."""
        entry = TimingEntry(name="test")
        assert entry.total_time == 0.0
        assert entry.call_count == 0
        assert entry.avg_time == 0.0

    def test_record_updates_stats(self) -> None:
        """Recording a measurement should update all stats."""
        entry = TimingEntry(name="test")
        entry.record(0.5)
        assert entry.total_time == 0.5
        assert entry.call_count == 1
        assert entry.min_time == 0.5
        assert entry.max_time == 0.5
        assert entry.last_time == 0.5

    def test_multiple_records(self) -> None:
        """Multiple recordings should accumulate correctly."""
        entry = TimingEntry(name="test")
        entry.record(0.1)
        entry.record(0.3)
        entry.record(0.2)
        assert entry.total_time == pytest.approx(0.6, abs=1e-9)
        assert entry.call_count == 3
        assert entry.min_time == pytest.approx(0.1, abs=1e-9)
        assert entry.max_time == pytest.approx(0.3, abs=1e-9)
        assert entry.avg_time == pytest.approx(0.2, abs=1e-9)

    def test_avg_time_zero_calls(self) -> None:
        """Average time with zero calls should be 0."""
        entry = TimingEntry(name="test")
        assert entry.avg_time == 0.0


class TestSolverProfiler:
    """Tests for the SolverProfiler class."""

    def test_measure_context_manager(self) -> None:
        """Context manager should record timing for a component."""
        profiler = SolverProfiler()
        with profiler.measure("test_component"):
            time.sleep(0.01)

        timing = profiler.get_timing("test_component")
        assert timing is not None
        assert timing.call_count == 1
        assert timing.total_time >= 0.005  # At least some time passed

    def test_multiple_measurements(self) -> None:
        """Multiple measurements of the same component accumulate."""
        profiler = SolverProfiler()
        for _ in range(3):
            with profiler.measure("repeated"):
                time.sleep(0.005)

        timing = profiler.get_timing("repeated")
        assert timing is not None
        assert timing.call_count == 3
        assert timing.total_time >= 0.01

    def test_different_components(self) -> None:
        """Different components should be tracked separately."""
        profiler = SolverProfiler()
        with profiler.measure("propagation"):
            time.sleep(0.01)
        with profiler.measure("search"):
            time.sleep(0.005)

        assert "propagation" in profiler.components
        assert "search" in profiler.components
        prop = profiler.get_timing("propagation")
        search = profiler.get_timing("search")
        assert prop is not None and search is not None
        assert prop.total_time > search.total_time

    def test_get_time_fraction(self) -> None:
        """Time fraction should sum to approximately 1.0."""
        profiler = SolverProfiler()
        with profiler.measure("a"):
            time.sleep(0.01)
        with profiler.measure("b"):
            time.sleep(0.01)

        frac_a = profiler.get_time_fraction("a")
        frac_b = profiler.get_time_fraction("b")
        assert 0.0 < frac_a < 1.0
        assert 0.0 < frac_b < 1.0
        assert frac_a + frac_b == pytest.approx(1.0, abs=0.01)

    def test_get_time_fraction_unknown_component(self) -> None:
        """Unknown component should return 0.0 fraction."""
        profiler = SolverProfiler()
        with profiler.measure("known"):
            pass
        assert profiler.get_time_fraction("unknown") == 0.0

    def test_bottleneck_detection(self) -> None:
        """Bottleneck should be the component with most time."""
        profiler = SolverProfiler()
        with profiler.measure("fast"):
            time.sleep(0.005)
        with profiler.measure("slow"):
            time.sleep(0.02)

        assert profiler.get_bottleneck() == "slow"

    def test_bottleneck_empty(self) -> None:
        """No bottleneck when no measurements exist."""
        profiler = SolverProfiler()
        assert profiler.get_bottleneck() is None

    def test_top_components(self) -> None:
        """Top components should be ordered by time."""
        profiler = SolverProfiler()
        with profiler.measure("medium"):
            time.sleep(0.01)
        with profiler.measure("fast"):
            time.sleep(0.005)
        with profiler.measure("slow"):
            time.sleep(0.02)

        top = profiler.get_top_components(k=2)
        assert len(top) == 2
        assert top[0][0] == "slow"
        assert top[1][0] == "medium"

    def test_propagation_search_ratio(self) -> None:
        """Propagation/search ratio should be computed correctly."""
        profiler = SolverProfiler()
        with profiler.measure("propagation"):
            time.sleep(0.02)
        with profiler.measure("search"):
            time.sleep(0.01)

        ratio = profiler.get_propagation_search_ratio()
        assert ratio > 1.0  # Propagation took longer

    def test_propagation_search_ratio_no_search(self) -> None:
        """Ratio should be inf when no search time recorded."""
        profiler = SolverProfiler()
        with profiler.measure("propagation"):
            time.sleep(0.01)
        assert profiler.get_propagation_search_ratio() == float("inf")

    def test_propagation_search_ratio_empty(self) -> None:
        """Ratio should be 0 when nothing recorded."""
        profiler = SolverProfiler()
        assert profiler.get_propagation_search_ratio() == 0.0

    def test_disabled_profiler(self) -> None:
        """Disabled profiler should not record anything."""
        profiler = SolverProfiler(enabled=False)
        with profiler.measure("test"):
            time.sleep(0.01)
        assert profiler.get_timing("test") is None
        assert profiler.total_time == 0.0

    def test_enable_disable(self) -> None:
        """Enable/disable should control recording."""
        profiler = SolverProfiler()
        profiler.disable()
        with profiler.measure("disabled"):
            pass
        assert profiler.get_timing("disabled") is None

        profiler.enable()
        with profiler.measure("enabled"):
            pass
        assert profiler.get_timing("enabled") is not None

    def test_reset(self) -> None:
        """Reset should clear all data."""
        profiler = SolverProfiler()
        with profiler.measure("test"):
            time.sleep(0.01)
        profiler.reset()
        assert profiler.total_time == 0.0
        assert profiler.components == []
        assert profiler.get_bottleneck() is None

    def test_manual_start_stop(self) -> None:
        """Manual start/stop timers should work."""
        profiler = SolverProfiler()
        profiler.start_timer("manual")
        time.sleep(0.01)
        elapsed = profiler.stop_timer("manual")
        assert elapsed >= 0.005
        timing = profiler.get_timing("manual")
        assert timing is not None
        assert timing.call_count == 1


class TestProfilerReport:
    """Tests for report generation."""

    def test_report_contains_components(self) -> None:
        """Report should list all measured components."""
        profiler = SolverProfiler()
        with profiler.measure("propagation"):
            time.sleep(0.005)
        with profiler.measure("search"):
            time.sleep(0.005)

        report = profiler.report()
        assert "propagation" in report
        assert "search" in report

    def test_report_contains_total_time(self) -> None:
        """Report should show total profiled time."""
        profiler = SolverProfiler()
        with profiler.measure("test"):
            time.sleep(0.01)

        report = profiler.report()
        assert "Total profiled time" in report

    def test_report_empty_profiler(self) -> None:
        """Report for empty profiler should indicate no data."""
        profiler = SolverProfiler()
        report = profiler.report()
        assert "No timing data" in report

    def test_report_contains_ratios(self) -> None:
        """Report should include key ratios section."""
        profiler = SolverProfiler()
        with profiler.measure("propagation"):
            time.sleep(0.005)
        with profiler.measure("search"):
            time.sleep(0.005)

        report = profiler.report()
        assert "Key Ratios" in report
        assert "Bottleneck" in report


class TestProfilerHook:
    """Tests for ProfilerHook integration."""

    def test_hook_creation(self) -> None:
        """Profiler should create a valid hook."""
        profiler = SolverProfiler()
        hook = profiler.create_hook()
        assert isinstance(hook, ProfilerHook)

    def test_hook_tracks_decisions(self) -> None:
        """Hook should count decisions."""
        profiler = SolverProfiler()
        hook = profiler.create_hook()
        var = Variable("x", range(5))
        hook.on_decision(var, 1, 0)
        hook.on_decision(var, 2, 1)
        assert hook.decision_count == 2

    def test_hook_tracks_backtracks(self) -> None:
        """Hook should count backtracks."""
        profiler = SolverProfiler()
        hook = profiler.create_hook()
        var = Variable("x", range(5))
        hook.on_decision(var, 1, 0)
        hook.on_backtrack(var, 0)
        assert hook.backtrack_count == 1

    def test_hook_on_start_finish(self) -> None:
        """Hook should record total solve time."""
        profiler = SolverProfiler()
        hook = profiler.create_hook()
        hook.on_start(5, 3)
        time.sleep(0.01)
        hook.on_finish(True, 10)
        timing = profiler.get_timing("total_solve")
        assert timing is not None
        assert timing.total_time >= 0.005
