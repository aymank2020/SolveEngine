"""Tests for constraint graph decomposition — structural invariants."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.decomposition.graph import ConstraintGraph


class TestConstraintGraph:
    """Qualitative tests for constraint graph structure."""

    def test_edge_count_consistency(self):
        """Number of edges matches constraint connections."""
        vars = [Variable(f"v{i}", [1, 2, 3]) for i in range(4)]
        constraints = [
            BinaryConstraint(vars[0], vars[1], lambda a, b: a != b),
            BinaryConstraint(vars[1], vars[2], lambda a, b: a != b),
            BinaryConstraint(vars[2], vars[3], lambda a, b: a != b),
        ]
        graph = ConstraintGraph(vars, constraints)
        assert graph.num_edges == 3

    def test_neighbor_symmetry(self):
        """If A is neighbor of B, then B is neighbor of A."""
        vars = [Variable(f"v{i}", [1, 2]) for i in range(3)]
        constraints = [
            BinaryConstraint(vars[0], vars[1], lambda a, b: a != b),
        ]
        graph = ConstraintGraph(vars, constraints)
        assert vars[1] in graph.neighbors(vars[0])
        assert vars[0] in graph.neighbors(vars[1])

    def test_disconnected_components(self):
        """Disconnected variables form separate components."""
        vars = [Variable(f"v{i}", [1, 2]) for i in range(4)]
        constraints = [
            BinaryConstraint(vars[0], vars[1], lambda a, b: a != b),
            BinaryConstraint(vars[2], vars[3], lambda a, b: a != b),
        ]
        graph = ConstraintGraph(vars, constraints)
        components = graph.connected_components()
        assert len(components) == 2

    def test_complete_graph_density(self):
        """Complete graph has density 1.0."""
        vars = [Variable(f"v{i}", [1, 2]) for i in range(4)]
        constraints = []
        for i in range(4):
            for j in range(i + 1, 4):
                constraints.append(BinaryConstraint(vars[i], vars[j], lambda a, b: a != b))
        graph = ConstraintGraph(vars, constraints)
        assert abs(graph.density() - 1.0) < 0.01

    def test_tree_detection(self):
        """A chain of constraints forms a tree."""
        vars = [Variable(f"v{i}", [1, 2]) for i in range(4)]
        constraints = [
            BinaryConstraint(vars[0], vars[1], lambda a, b: a != b),
            BinaryConstraint(vars[1], vars[2], lambda a, b: a != b),
            BinaryConstraint(vars[2], vars[3], lambda a, b: a != b),
        ]
        graph = ConstraintGraph(vars, constraints)
        assert graph.is_tree()

    def test_cycle_not_tree(self):
        """A cycle is not a tree."""
        vars = [Variable(f"v{i}", [1, 2, 3]) for i in range(3)]
        constraints = [
            BinaryConstraint(vars[0], vars[1], lambda a, b: a != b),
            BinaryConstraint(vars[1], vars[2], lambda a, b: a != b),
            BinaryConstraint(vars[0], vars[2], lambda a, b: a != b),
        ]
        graph = ConstraintGraph(vars, constraints)
        assert not graph.is_tree()

    def test_degree_non_negative(self):
        """Degree is always >= 0."""
        vars = [Variable(f"v{i}", [1, 2]) for i in range(3)]
        graph = ConstraintGraph(vars, [])
        for v in vars:
            assert graph.degree(v) >= 0
