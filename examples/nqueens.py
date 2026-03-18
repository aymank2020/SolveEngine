"""N-Queens problem solver using SolveEngine."""

from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.heuristics.variable_ordering import DomWdegSelector
from solveengine.heuristics.value_ordering import LCVOrderer


def solve_nqueens(n: int) -> list[int] | None:
    """Solve the N-Queens problem.

    Returns a list of column positions for each row, or None if unsolvable.
    """
    queens = [Variable(f"q{i}", range(0, n)) for i in range(n)]
    constraints = []

    for i in range(n):
        for j in range(i + 1, n):
            diff = j - i
            # Different columns
            constraints.append(BinaryConstraint(
                queens[i], queens[j], lambda a, b: a != b
            ))
            # Different diagonals
            constraints.append(BinaryConstraint(
                queens[i], queens[j], lambda a, b, d=diff: abs(a - b) != d
            ))

    solver = BacktrackSolver(
        queens, constraints,
        var_selector=DomWdegSelector(),
        val_orderer=LCVOrderer(),
        use_forward_check=True,
    )

    result = solver.solve()
    if result is None:
        return None

    return [result[q] for q in queens]


def print_board(positions: list[int]) -> None:
    """Print a chessboard with queens placed."""
    n = len(positions)
    for row in range(n):
        line = ""
        for col in range(n):
            if positions[row] == col:
                line += " Q"
            else:
                line += " ."
        print(line)


if __name__ == "__main__":
    for n in [4, 8, 12]:
        print(f"\n{'='*20} {n}-Queens {'='*20}")
        solution = solve_nqueens(n)
        if solution:
            print_board(solution)
        else:
            print(f"No solution for {n}-Queens")
