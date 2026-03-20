"""Sudoku solver using SolveEngine."""

from solveengine.core.variable import Variable
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.heuristics.variable_ordering import DomWdegSelector


def solve_sudoku(grid: list[list[int]]) -> list[list[int]] | None:
    """Solve a 9x9 Sudoku puzzle.

    Args:
        grid: 9x9 grid where 0 represents empty cells.

    Returns:
        Completed grid, or None if unsolvable.
    """
    variables: list[list[Variable]] = []
    all_vars: list[Variable] = []
    constraints = []

    # Create variables
    for i in range(9):
        row_vars = []
        for j in range(9):
            if grid[i][j] != 0:
                var = Variable(f"c{i}{j}", [grid[i][j]])
            else:
                var = Variable(f"c{i}{j}", range(1, 10))
            row_vars.append(var)
            all_vars.append(var)
        variables.append(row_vars)

    # Row constraints
    for i in range(9):
        constraints.append(AllDifferent(*variables[i]))

    # Column constraints
    for j in range(9):
        col_vars = [variables[i][j] for i in range(9)]
        constraints.append(AllDifferent(*col_vars))

    # Box constraints
    for box_row in range(3):
        for box_col in range(3):
            box_vars = []
            for i in range(3):
                for j in range(3):
                    box_vars.append(variables[box_row * 3 + i][box_col * 3 + j])
            constraints.append(AllDifferent(*box_vars))

    solver = BacktrackSolver(
        all_vars, constraints,
        var_selector=DomWdegSelector(),
        use_forward_check=True,
    )

    result = solver.solve()
    if result is None:
        return None

    solution = [[0] * 9 for _ in range(9)]
    for i in range(9):
        for j in range(9):
            solution[i][j] = result[variables[i][j]]

    return solution


# Example puzzle (easy)
EXAMPLE_PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

if __name__ == "__main__":
    print("Solving Sudoku...")
    solution = solve_sudoku(EXAMPLE_PUZZLE)
    if solution:
        for row in solution:
            print(" ".join(str(v) for v in row))
    else:
        print("No solution found")
