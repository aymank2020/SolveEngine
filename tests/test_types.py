"""Tests for core types and protocols."""

import pytest
from solveengine.core.types import (
    SolverStatus,
    PropagationLevel,
    SearchPhase,
    SolverConfig,
    SolverResult,
)


class TestSolverConfig:
    def test_default_config(self):
        config = SolverConfig()
        assert config.propagation_level == PropagationLevel.ARC
        assert not config.use_restarts
        assert not config.has_node_limit
        assert not config.find_all

    def test_node_limit(self):
        config = SolverConfig(node_limit=1000)
        assert config.has_node_limit
        assert config.node_limit == 1000

    def test_find_all(self):
        config = SolverConfig(solution_limit=0)
        assert config.find_all

    def test_time_limit(self):
        config = SolverConfig(time_limit_ms=5000)
        assert config.has_time_limit


class TestSolverResult:
    def test_solved_result(self):
        result = SolverResult(
            status=SolverStatus.SOLVED,
            solutions=[{"x": 1, "y": 2}],
            nodes_explored=10,
        )
        assert result.is_solved
        assert not result.is_unsat
        assert result.first_solution == {"x": 1, "y": 2}
        assert result.num_solutions == 1

    def test_unsat_result(self):
        result = SolverResult(
            status=SolverStatus.UNSATISFIABLE,
            solutions=[],
        )
        assert result.is_unsat
        assert not result.is_solved
        assert result.first_solution is None

    def test_timeout_result(self):
        result = SolverResult(status=SolverStatus.TIMEOUT, solutions=[])
        assert not result.is_solved
        assert not result.is_unsat
