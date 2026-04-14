"""Tests for the backtracking solver — solution validity invariants."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.global_cstr.linear import SumConstraint, ComparisonOp
from solveengine.solver.backtrack import BacktrackSolver


class TestSolverInvariants:
    """Qualitative tests: any solution must satisfy all constraints."""

    def test_solution_satisfies_all_constraints(self):
        """Every returned solution satisfies all constraints."""
        x = Variable("x", range(1, 5))
        y = Variable("y", range(1, 5))
        z = Variable("z", range(1, 5))
        constraints = [
            BinaryConstraint(x, y, lambda a, b: a != b),
            BinaryConstraint(y, z, lambda a, b: a != b),
            BinaryConstraint(x, z, lambda a, b: a < b),
        ]

        solver = BacktrackSolver([x, y, z], constraints)
        result = solver.solve()

        assert result is not None
        for cstr in constraints:
            assert cstr.is_satisfied(result)

    def test_solution_assigns_all_variables(self):
        """A solution assigns every variable."""
        variables = [Variable(f"v{i}", range(1, 4)) for i in range(3)]
        constraints = [AllDifferent(*variables)]

        solver = BacktrackSolver(variables, constraints)
        result = solver.solve()

        assert result is not None
        assert len(result) == len(variables)

    def test_unsatisfiable_returns_none(self):
        """Unsatisfiable problem returns None."""
        x = Variable("x", [1])
        y = Variable("y", [1])
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)

        solver = BacktrackSolver([x, y], [cstr])
        result = solver.solve()
        assert result is None

    def test_all_solutions_distinct(self):
        """solve_all returns distinct solutions."""
        x = Variable("x", [1, 2, 3])
        y = Variable("y", [1, 2, 3])
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)

        solver = BacktrackSolver([x, y], [cstr])
        solutions = solver.solve_all()

        # Convert to tuples for comparison
        tuples = [tuple(sorted(s.items(), key=lambda p: p[0].index)) for s in solutions]
        assert len(tuples) == len(set(tuples))

    def test_node_count_non_negative(self):
        """Stats always have non-negative values."""
        x = Variable("x", range(1, 4))
        y = Variable("y", range(1, 4))
        cstr = BinaryConstraint(x, y, lambda a, b: a + b == 4)

        solver = BacktrackSolver([x, y], [cstr])
        solver.solve()

        assert solver.stats.nodes_explored >= 0
        assert solver.stats.backtracks >= 0

    def test_4queens_has_solution(self):
        """4-Queens problem is satisfiable (structural test)."""
        n = 4
        queens = [Variable(f"q{i}", range(0, n)) for i in range(n)]
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                # Different columns
                constraints.append(BinaryConstraint(
                    queens[i], queens[j], lambda a, b: a != b
                ))
                # Different diagonals
                diff = j - i
                constraints.append(BinaryConstraint(
                    queens[i], queens[j], lambda a, b, d=diff: abs(a - b) != d
                ))

        solver = BacktrackSolver(queens, constraints, use_forward_check=True)
        result = solver.solve()
        assert result is not None

    def test_sum_constraint_solution_valid(self):
        """Sum constraint solutions actually sum to target."""
        x = Variable("x", range(1, 10))
        y = Variable("y", range(1, 10))
        z = Variable("z", range(1, 10))
        cstr = SumConstraint([x, y, z], 15, ComparisonOp.EQ)

        solver = BacktrackSolver([x, y, z], [cstr])
        result = solver.solve()

        if result is not None:
            total = sum(result[v] for v in [x, y, z])
            assert total == 15
