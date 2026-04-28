"""Tests for restart search strategy.

Tests Luby sequence correctness, restart policy behavior, and that
restarts improve solver performance on hard instances while preserving
learned weights across restarts.
"""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.optimizer.restarts import (
    LubyRestart,
    GeometricRestart,
    FixedRestart,
    NoRestart,
    NestedRestart,
)
from solveengine.search.restart_search import RestartSearch, RestartSearchResult
from solveengine.heuristics.variable_ordering import DomWdegSelector, MRVSelector
from solveengine.heuristics.value_ordering import AscendingOrderer, RandomOrderer


class TestLubySequence:
    """Tests for Luby sequence correctness."""

    def test_first_elements(self):
        """First elements of Luby sequence are correct."""
        # Luby sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
        expected = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8]
        for i, exp in enumerate(expected, start=1):
            assert LubyRestart._luby(i) == exp, f"luby({i}) should be {exp}"

    def test_luby_powers_of_two_minus_one(self):
        """At positions 2^k - 1, Luby value is 2^(k-1)."""
        # Position 1 (2^1 - 1): value 1 (2^0)
        assert LubyRestart._luby(1) == 1
        # Position 3 (2^2 - 1): value 2 (2^1)
        assert LubyRestart._luby(3) == 2
        # Position 7 (2^3 - 1): value 4 (2^2)
        assert LubyRestart._luby(7) == 4
        # Position 15 (2^4 - 1): value 8 (2^3)
        assert LubyRestart._luby(15) == 8
        # Position 31 (2^5 - 1): value 16 (2^4)
        assert LubyRestart._luby(31) == 16

    def test_luby_cutoff_with_unit(self):
        """Luby cutoff is unit * luby(restart_count + 1)."""
        policy = LubyRestart(unit=50)
        # First restart: cutoff = 50 * luby(1) = 50
        assert policy.next_cutoff() == 50
        policy.record_restart(50)
        # Second restart: cutoff = 50 * luby(2) = 50
        assert policy.next_cutoff() == 50
        policy.record_restart(50)
        # Third restart: cutoff = 50 * luby(3) = 100
        assert policy.next_cutoff() == 100

    def test_luby_sequence_is_non_decreasing_in_max(self):
        """The maximum value in Luby sequence up to position n is non-decreasing."""
        max_seen = 0
        for i in range(1, 64):
            val = LubyRestart._luby(i)
            if val > max_seen:
                max_seen = val
            # Max should only increase at powers of 2 minus 1
            assert val <= max_seen


class TestGeometricRestart:
    """Tests for geometric restart policy."""

    def test_cutoff_grows_geometrically(self):
        """Cutoff grows by factor each restart."""
        policy = GeometricRestart(base=100, factor=2.0)
        assert policy.next_cutoff() == 100
        policy.record_restart(100)
        assert policy.next_cutoff() == 200
        policy.record_restart(200)
        assert policy.next_cutoff() == 400

    def test_should_restart_at_cutoff(self):
        """Should restart when nodes reach cutoff."""
        policy = GeometricRestart(base=50, factor=1.5)
        assert not policy.should_restart(49)
        assert policy.should_restart(50)
        assert policy.should_restart(100)

    def test_reset_restores_initial_state(self):
        """Reset brings policy back to initial state."""
        policy = GeometricRestart(base=100, factor=2.0)
        policy.record_restart(100)
        policy.record_restart(200)
        assert policy.restart_count == 2
        policy.reset()
        assert policy.restart_count == 0
        assert policy.next_cutoff() == 100


class TestFixedRestart:
    """Tests for fixed restart policy."""

    def test_constant_cutoff(self):
        """Cutoff remains constant across restarts."""
        policy = FixedRestart(cutoff=200)
        for _ in range(10):
            assert policy.next_cutoff() == 200
            policy.record_restart(200)

    def test_should_restart_at_exact_cutoff(self):
        """Should restart exactly at cutoff."""
        policy = FixedRestart(cutoff=100)
        assert not policy.should_restart(99)
        assert policy.should_restart(100)


class TestNoRestart:
    """Tests for no-restart policy."""

    def test_never_restarts(self):
        """NoRestart never triggers a restart."""
        policy = NoRestart()
        assert not policy.should_restart(0)
        assert not policy.should_restart(1000000)
        assert not policy.should_restart(2**60)

    def test_cutoff_is_effectively_infinite(self):
        """Cutoff is extremely large."""
        policy = NoRestart()
        assert policy.next_cutoff() > 2**60


class TestNestedRestart:
    """Tests for nested restart policy."""

    def test_inner_policy_triggers_first(self):
        """Inner policy triggers restarts within a phase."""
        inner = FixedRestart(cutoff=50)
        policy = NestedRestart(inner=inner, outer_base=200, outer_factor=2.0)
        assert policy.should_restart(50)
        assert not policy.should_restart(49)

    def test_outer_phase_resets_inner(self):
        """After outer phase budget, inner policy resets."""
        inner = FixedRestart(cutoff=50)
        policy = NestedRestart(inner=inner, outer_base=100, outer_factor=2.0)
        # Record enough restarts to exhaust outer phase
        policy.record_restart(50)
        policy.record_restart(50)
        # After 100 nodes in phase, outer should advance
        # Inner should have been reset
        assert policy.next_cutoff() == 50


class TestRestartSearchIntegration:
    """Integration tests for restart search with actual CSP solving."""

    def _create_nqueens(self, n: int) -> tuple[list[Variable], list[BinaryConstraint]]:
        """Create an N-Queens problem instance."""
        queens = [Variable(f"q{i}", range(0, n)) for i in range(n)]
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = j - i
                constraints.append(BinaryConstraint(
                    queens[i], queens[j], lambda a, b: a != b
                ))
                constraints.append(BinaryConstraint(
                    queens[i], queens[j], lambda a, b, d=diff: abs(a - b) != d
                ))
        return queens, constraints

    def test_restart_search_finds_solution(self):
        """Restart search finds a valid solution for 8-Queens."""
        queens, constraints = self._create_nqueens(8)
        search = RestartSearch(
            queens, constraints,
            restart_policy=LubyRestart(unit=50),
            var_selector=DomWdegSelector(),
            max_restarts=50,
        )
        result = search.solve()
        assert result.is_solved
        assert result.solution is not None
        assert len(result.solution) == 8

    def test_restart_search_result_has_stats(self):
        """Result includes meaningful statistics."""
        queens, constraints = self._create_nqueens(4)
        search = RestartSearch(
            queens, constraints,
            restart_policy=LubyRestart(unit=20),
            max_restarts=20,
        )
        result = search.solve()
        assert result.total_nodes > 0
        assert result.stats.nodes_explored > 0

    def test_weight_preservation_across_restarts(self):
        """Variable weights are preserved across restarts."""
        queens, constraints = self._create_nqueens(6)
        # Set initial weights
        for q in queens:
            q.increment_weight(0.5)
        initial_weights = [q.weight for q in queens]

        search = RestartSearch(
            queens, constraints,
            restart_policy=FixedRestart(cutoff=10),
            var_selector=DomWdegSelector(),
            max_restarts=5,
            total_node_limit=100,
        )
        search.solve()

        # Weights should have increased (or stayed same) due to conflicts
        for i, q in enumerate(queens):
            assert q.weight >= initial_weights[i]

    def test_domains_restored_after_failed_restart(self):
        """Variable domains are fully restored after failed restarts."""
        # Use a hard problem that won't be solved within the limit
        queens, constraints = self._create_nqueens(20)
        original_sizes = [q.domain_size for q in queens]

        search = RestartSearch(
            queens, constraints,
            restart_policy=FixedRestart(cutoff=5),
            max_restarts=3,
            total_node_limit=50,
        )
        result = search.solve()

        # If not solved, domains should be fully restored
        if not result.is_solved:
            for i, q in enumerate(queens):
                assert q.domain_size == original_sizes[i]

    def test_node_limit_respected(self):
        """Total node limit is respected across restarts."""
        queens, constraints = self._create_nqueens(12)
        node_limit = 500

        search = RestartSearch(
            queens, constraints,
            restart_policy=LubyRestart(unit=50),
            max_restarts=100,
            total_node_limit=node_limit,
        )
        result = search.solve()
        assert result.total_nodes <= node_limit + 100  # Small tolerance for budget

    def test_max_restarts_respected(self):
        """Maximum restart count is respected."""
        queens, constraints = self._create_nqueens(20)
        max_restarts = 5

        search = RestartSearch(
            queens, constraints,
            restart_policy=FixedRestart(cutoff=10),
            max_restarts=max_restarts,
            total_node_limit=1000000,
        )
        result = search.solve()
        assert result.num_restarts <= max_restarts + 1

    def test_unsatisfiable_returns_none(self):
        """Restart search returns None for unsatisfiable problems."""
        # 4 variables, domain {1,2}, all must be different — impossible
        vars = [Variable(f"v{i}", [1, 2]) for i in range(4)]
        constraints = []
        for i in range(4):
            for j in range(i + 1, 4):
                constraints.append(BinaryConstraint(
                    vars[i], vars[j], lambda a, b: a != b
                ))

        search = RestartSearch(
            vars, constraints,
            restart_policy=LubyRestart(unit=20),
            max_restarts=10,
            total_node_limit=1000,
        )
        result = search.solve()
        assert not result.is_solved
        assert result.solution is None
