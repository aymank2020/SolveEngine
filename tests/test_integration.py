"""Integration tests — end-to-end solving of classic CSP problems."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.global_cstr.linear import SumConstraint, ComparisonOp
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.heuristics.variable_ordering import DomWdegSelector
from solveengine.heuristics.value_ordering import LCVOrderer


class TestNQueens:
    """N-Queens problem as integration test."""

    def _make_nqueens(self, n: int):
        queens = [Variable(f"q{i}", range(0, n)) for i in range(n)]
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = j - i
                constraints.append(BinaryConstraint(
                    queens[i], queens[j], lambda a, b, _i=i, _j=j: a != b
                ))
                constraints.append(BinaryConstraint(
                    queens[i], queens[j], lambda a, b, d=diff: abs(a - b) != d
                ))
        return queens, constraints

    def test_4queens_solvable(self):
        """4-Queens has solutions."""
        queens, constraints = self._make_nqueens(4)
        solver = BacktrackSolver(queens, constraints, use_forward_check=True)
        result = solver.solve()
        assert result is not None

    def test_4queens_solution_valid(self):
        """4-Queens solution has no conflicts."""
        queens, constraints = self._make_nqueens(4)
        solver = BacktrackSolver(queens, constraints, use_forward_check=True)
        result = solver.solve()
        assert result is not None
        # Verify no two queens attack each other
        positions = [result[q] for q in queens]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                assert positions[i] != positions[j]  # Same column
                assert abs(positions[i] - positions[j]) != j - i  # Diagonal

    def test_3queens_unsatisfiable(self):
        """3-Queens has no solution."""
        queens, constraints = self._make_nqueens(3)
        solver = BacktrackSolver(queens, constraints, use_forward_check=True)
        result = solver.solve()
        assert result is None


class TestGraphColoring:
    """Graph coloring as integration test."""

    def test_3color_triangle(self):
        """Triangle is 3-colorable."""
        v1 = Variable("v1", [1, 2, 3])
        v2 = Variable("v2", [1, 2, 3])
        v3 = Variable("v3", [1, 2, 3])
        constraints = [
            BinaryConstraint(v1, v2, lambda a, b: a != b),
            BinaryConstraint(v2, v3, lambda a, b: a != b),
            BinaryConstraint(v1, v3, lambda a, b: a != b),
        ]
        solver = BacktrackSolver([v1, v2, v3], constraints, use_forward_check=True)
        result = solver.solve()
        assert result is not None
        assert result[v1] != result[v2]
        assert result[v2] != result[v3]
        assert result[v1] != result[v3]

    def test_2color_triangle_impossible(self):
        """Triangle is not 2-colorable."""
        v1 = Variable("v1", [1, 2])
        v2 = Variable("v2", [1, 2])
        v3 = Variable("v3", [1, 2])
        constraints = [
            BinaryConstraint(v1, v2, lambda a, b: a != b),
            BinaryConstraint(v2, v3, lambda a, b: a != b),
            BinaryConstraint(v1, v3, lambda a, b: a != b),
        ]
        solver = BacktrackSolver([v1, v2, v3], constraints, use_forward_check=True)
        result = solver.solve()
        assert result is None


class TestMagicSquare:
    """Magic square as integration test (3x3)."""

    def test_3x3_magic_square(self):
        """3x3 magic square with sum=15 is solvable."""
        # Variables for 3x3 grid, values 1-9
        cells = [Variable(f"c{i}", range(1, 10)) for i in range(9)]

        constraints = []
        # All different
        constraints.append(AllDifferent(*cells))

        # Row sums = 15
        for row in range(3):
            row_vars = cells[row * 3:(row + 1) * 3]
            constraints.append(SumConstraint(row_vars, 15, ComparisonOp.EQ))

        # Column sums = 15
        for col in range(3):
            col_vars = [cells[row * 3 + col] for row in range(3)]
            constraints.append(SumConstraint(col_vars, 15, ComparisonOp.EQ))

        # Diagonal sums = 15
        diag1 = [cells[0], cells[4], cells[8]]
        diag2 = [cells[2], cells[4], cells[6]]
        constraints.append(SumConstraint(diag1, 15, ComparisonOp.EQ))
        constraints.append(SumConstraint(diag2, 15, ComparisonOp.EQ))

        solver = BacktrackSolver(
            cells, constraints,
            var_selector=DomWdegSelector(),
            val_orderer=LCVOrderer(),
            use_forward_check=True,
        )
        result = solver.solve()
        assert result is not None

        # Verify solution
        grid = [result[cells[i]] for i in range(9)]
        assert len(set(grid)) == 9  # All different
        for row in range(3):
            assert sum(grid[row * 3:(row + 1) * 3]) == 15
