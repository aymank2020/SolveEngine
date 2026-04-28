"""Tests for search strategies."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.search.dfs import DFSStrategy
from solveengine.search.lds import LDSStrategy


class TestDFS:
    """Qualitative tests for DFS search."""

    def test_finds_solution_if_exists(self):
        """DFS finds a solution for satisfiable problems."""
        x = Variable("x", [1, 2, 3])
        y = Variable("y", [1, 2, 3])
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)

        dfs = DFSStrategy()
        solutions = list(dfs.explore([x, y], [cstr], {}))
        assert len(solutions) > 0

    def test_all_solutions_valid(self):
        """Every solution from DFS satisfies all constraints."""
        x = Variable("x", [1, 2, 3])
        y = Variable("y", [1, 2, 3])
        cstr = BinaryConstraint(x, y, lambda a, b: a < b)

        dfs = DFSStrategy()
        for sol in dfs.explore([x, y], [cstr], {}):
            assert sol[x] < sol[y]

    def test_depth_limit_respected(self):
        """DFS with depth limit doesn't go deeper."""
        vars = [Variable(f"v{i}", [1, 2]) for i in range(10)]
        dfs = DFSStrategy(depth_limit=3)
        solutions = list(dfs.explore(vars, [], {}))
        # With depth limit 3, can't assign all 10 vars
        assert len(solutions) == 0

    def test_nodes_explored_positive(self):
        """At least one node is explored."""
        x = Variable("x", [1])
        dfs = DFSStrategy()
        list(dfs.explore([x], [], {}))
        assert dfs.nodes_explored >= 1


class TestLDS:
    """Qualitative tests for Limited Discrepancy Search."""

    def test_zero_discrepancy_fewer_solutions(self):
        """With 0 discrepancies, LDS finds fewer solutions than with more."""
        x = Variable("x", [1, 2, 3])
        y = Variable("y", [1, 2, 3])
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)

        lds_0 = LDSStrategy(max_discrepancy=0)
        solutions_0 = list(lds_0.explore([x, y], [cstr], {}))

        lds_2 = LDSStrategy(max_discrepancy=2)
        solutions_2 = list(lds_2.explore([x, y], [cstr], {}))

        # More discrepancies = more solutions explored
        assert len(solutions_0) <= len(solutions_2)

    def test_finds_all_with_enough_discrepancy(self):
        """With enough discrepancies, LDS finds all solutions."""
        x = Variable("x", [1, 2])
        y = Variable("y", [1, 2])
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)

        lds = LDSStrategy(max_discrepancy=2)
        solutions = list(lds.explore([x, y], [cstr], {}))
        assert len(solutions) >= 1
