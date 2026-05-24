"""Tests for optimizer module: restarts, adaptive heuristics, portfolio."""

import pytest
from solveengine.optimizer.restarts import (
    GeometricRestart,
    LubyRestart,
    FixedRestart,
    NoRestart,
    NestedRestart,
)
from solveengine.optimizer.adaptive import AdaptiveHeuristic
from solveengine.optimizer.portfolio import PortfolioSolver, SolverConfig
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.heuristics.variable_ordering import MRVSelector
from solveengine.heuristics.value_ordering import AscendingOrderer


class TestGeometricRestart:
    def test_cutoff_grows(self):
        """Cutoff increases with each restart."""
        policy = GeometricRestart(base=100, factor=2.0)
        cutoffs = []
        for _ in range(5):
            cutoffs.append(policy.next_cutoff())
            policy.record_restart(policy.next_cutoff())
        assert cutoffs == sorted(cutoffs)

    def test_should_restart_at_cutoff(self):
        """Should restart when nodes reach cutoff."""
        policy = GeometricRestart(base=10, factor=1.5)
        assert not policy.should_restart(5)
        assert policy.should_restart(10)
        assert policy.should_restart(15)

    def test_restart_count_increments(self):
        """Restart count increases with each restart."""
        policy = GeometricRestart()
        assert policy.restart_count == 0
        policy.record_restart(100)
        assert policy.restart_count == 1


class TestLubyRestart:
    def test_luby_sequence_values(self):
        """Luby sequence starts with 1, 1, 2, 1, 1, 2, 4."""
        expected = [1, 1, 2, 1, 1, 2, 4]
        for i, exp in enumerate(expected, 1):
            assert LubyRestart._luby(i) == exp

    def test_cutoff_uses_unit(self):
        """Cutoff is unit * luby(restart_count + 1)."""
        policy = LubyRestart(unit=50)
        assert policy.next_cutoff() == 50  # luby(1) = 1

    def test_no_restart_never_triggers(self):
        """NoRestart never triggers."""
        policy = NoRestart()
        assert not policy.should_restart(999999)


class TestFixedRestart:
    def test_fixed_cutoff(self):
        """Fixed restart always uses same cutoff."""
        policy = FixedRestart(cutoff=200)
        policy.record_restart(200)
        policy.record_restart(200)
        assert policy.next_cutoff() == 200


class TestAdaptiveHeuristic:
    def test_selects_from_unassigned(self):
        """Adaptive heuristic returns an unassigned variable."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(3)]
        adaptive = AdaptiveHeuristic()
        selected = adaptive.select(vars, [], {})
        assert selected in vars

    def test_stats_tracking(self):
        """Stats are tracked per heuristic."""
        adaptive = AdaptiveHeuristic(window_size=100)
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(3)]
        adaptive.select(vars, [], {})
        adaptive.record_node()
        adaptive.record_backtrack()
        stats = adaptive.stats
        assert any(s.total_nodes > 0 for s in stats)


class TestPortfolioSolver:
    def test_solves_simple_problem(self):
        """Portfolio solver finds solution for simple problem."""
        x = Variable("x", range(1, 5))
        y = Variable("y", range(1, 5))
        cstr = BinaryConstraint(x, y, lambda a, b: a + b == 5)

        portfolio = PortfolioSolver()
        result = portfolio.solve([x, y], [cstr])
        assert result.is_solved

    def test_reports_winning_config(self):
        """Portfolio reports which config found the solution."""
        x = Variable("x", [1, 2, 3])
        y = Variable("y", [1, 2, 3])
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)

        portfolio = PortfolioSolver()
        result = portfolio.solve([x, y], [cstr])
        assert result.winning_config is not None

    def test_unsatisfiable_returns_no_solution(self):
        """Portfolio returns no solution for unsat problems."""
        x = Variable("x", [1])
        y = Variable("y", [1])
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)

        configs = [SolverConfig(
            name="test", var_selector=MRVSelector(),
            val_orderer=AscendingOrderer(), node_budget=1000
        )]
        portfolio = PortfolioSolver(configs)
        result = portfolio.solve([x, y], [cstr])
        assert not result.is_solved
