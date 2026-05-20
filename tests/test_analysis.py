"""Tests for analysis module: metrics and tightness."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.analysis.metrics import GraphMetrics
from solveengine.analysis.tightness import TightnessAnalyzer


class TestGraphMetrics:
    def test_basic_metrics(self):
        """Metrics are computed correctly for simple problem."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(3)]
        constraints = [
            BinaryConstraint(vars[0], vars[1], lambda a, b: a != b),
            BinaryConstraint(vars[1], vars[2], lambda a, b: a != b),
        ]
        metrics = GraphMetrics(vars, constraints)
        result = metrics.compute()
        assert result.num_variables == 3
        assert result.num_constraints == 2
        assert result.avg_domain_size == 4.0

    def test_density_range(self):
        """Graph density is between 0 and 1."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(4)]
        constraints = [BinaryConstraint(vars[0], vars[1], lambda a, b: a != b)]
        metrics = GraphMetrics(vars, constraints)
        result = metrics.compute()
        assert 0.0 <= result.graph_density <= 1.0

    def test_kappa_non_negative(self):
        """Constrainedness parameter is non-negative."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(3)]
        constraints = [BinaryConstraint(vars[0], vars[1], lambda a, b: a != b)]
        metrics = GraphMetrics(vars, constraints)
        result = metrics.compute()
        assert result.kappa >= 0.0

    def test_suggest_heuristic(self):
        """Heuristic suggestion returns a valid string."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(3)]
        constraints = [BinaryConstraint(vars[0], vars[1], lambda a, b: a != b)]
        metrics = GraphMetrics(vars, constraints)
        suggestion = metrics.suggest_heuristic()
        assert suggestion in ("mrv", "mrv_degree", "dom_wdeg", "component_decomposition")


class TestTightnessAnalyzer:
    def test_not_equal_tightness(self):
        """Not-equal constraint has tightness 1/d for domain size d."""
        x = Variable("x", range(1, 5))
        y = Variable("y", range(1, 5))
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)
        analyzer = TightnessAnalyzer([x, y], [cstr])
        results = analyzer.analyze_all()
        assert len(results) == 1
        # Tightness of != with domain 4: 4/(4*4) = 0.25
        assert abs(results[0].tightness - 0.25) < 0.01

    def test_bottleneck_detection(self):
        """Tight constraints are flagged as bottlenecks."""
        x = Variable("x", [1, 2])
        y = Variable("y", [1, 2])
        # Very tight: only (1,2) and (2,1) allowed = tightness 0.5
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)
        analyzer = TightnessAnalyzer([x, y], [cstr], bottleneck_threshold=0.4)
        results = analyzer.analyze_all()
        assert results[0].is_bottleneck

    def test_ranking_order(self):
        """Ranking returns tightest constraint first."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(3)]
        loose = BinaryConstraint(vars[0], vars[1], lambda a, b: a != b)  # 0.25
        tight = BinaryConstraint(vars[1], vars[2], lambda a, b: a < b)  # ~0.5
        analyzer = TightnessAnalyzer(vars, [loose, tight])
        ranked = analyzer.rank_by_tightness()
        assert ranked[0][1] >= ranked[1][1]  # First is tighter
