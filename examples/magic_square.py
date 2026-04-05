"""Magic Square solver example.

A magic square is an n×n grid filled with distinct integers from 1 to n²
such that the sum of each row, each column, and both main diagonals
equals the same constant (the magic constant).

Magic constant for n×n: M = n(n² + 1) / 2

This example demonstrates:
- Creating a grid of variables
- AllDifferent constraint for uniqueness
- Sum constraints for rows, columns, and diagonals
- Solving and displaying the result

Supports 3×3 and 4×4 magic squares.
"""

from __future__ import annotations

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.global_cstr.linear import SumConstraint, ComparisonOp
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.heuristics.variable_ordering import DomWdegSelector
from solveengine.heuristics.value_ordering import LCVOrderer


def magic_constant(n: int) -> int:
    """Compute the magic constant for an n×n magic square.

    The magic constant M = n(n² + 1) / 2.
    """
    return n * (n * n + 1) // 2


def create_magic_square(n: int) -> tuple[list[list[Variable]], list[Constraint]]:
    """Create variables and constraints for an n×n magic square.

    Args:
        n: Size of the magic square.

    Returns:
        Tuple of (grid of variables, list of constraints).
    """
    m = magic_constant(n)
    n_sq = n * n

    # Create n×n grid of variables, each with domain [1, n²]
    grid: list[list[Variable]] = []
    all_vars: list[Variable] = []
    for i in range(n):
        row: list[Variable] = []
        for j in range(n):
            var = Variable(f"cell_{i}_{j}", range(1, n_sq + 1))
            row.append(var)
            all_vars.append(var)
        grid.append(row)

    constraints: list[Constraint] = []

    # All values must be distinct (1 to n² each used exactly once)
    alldiff = AllDifferent(*all_vars)
    constraints.append(alldiff)

    # Row sum constraints
    for i in range(n):
        row_vars = grid[i]
        row_sum = SumConstraint(row_vars, m, ComparisonOp.EQ)
        constraints.append(row_sum)

    # Column sum constraints
    for j in range(n):
        col_vars = [grid[i][j] for i in range(n)]
        col_sum = SumConstraint(col_vars, m, ComparisonOp.EQ)
        constraints.append(col_sum)

    # Main diagonal (top-left to bottom-right)
    diag1_vars = [grid[i][i] for i in range(n)]
    diag1_sum = SumConstraint(diag1_vars, m, ComparisonOp.EQ)
    constraints.append(diag1_sum)

    # Anti-diagonal (top-right to bottom-left)
    diag2_vars = [grid[i][n - 1 - i] for i in range(n)]
    diag2_sum = SumConstraint(diag2_vars, m, ComparisonOp.EQ)
    constraints.append(diag2_sum)

    return grid, constraints


def solve_magic_square(n: int) -> dict[Variable, int] | None:
    """Solve an n×n magic square.

    Args:
        n: Size of the magic square.

    Returns:
        Solution assignment or None if no solution exists.
    """
    grid, constraints = create_magic_square(n)
    all_vars = [var for row in grid for var in row]

    solver = BacktrackSolver(
        all_vars,
        constraints,
        var_selector=DomWdegSelector(),
        val_orderer=LCVOrderer(),
        use_forward_check=True,
    )

    return solver.solve()


def display_magic_square(grid: list[list[Variable]], solution: dict[Variable, int]) -> str:
    """Format a magic square solution for display.

    Args:
        grid: The variable grid.
        solution: The solution assignment.

    Returns:
        Formatted string representation of the magic square.
    """
    n = len(grid)
    m = magic_constant(n)
    n_sq = n * n
    cell_width = len(str(n_sq)) + 1

    lines: list[str] = []
    lines.append(f"Magic Square {n}×{n} (magic constant = {m})")
    lines.append("+" + ("-" * cell_width + "+") * n)

    for i in range(n):
        row_str = "|"
        for j in range(n):
            val = solution[grid[i][j]]
            row_str += f"{val:>{cell_width}}|"
        lines.append(row_str)
        lines.append("+" + ("-" * cell_width + "+") * n)

    return "\n".join(lines)


def verify_magic_square(grid: list[list[Variable]], solution: dict[Variable, int]) -> bool:
    """Verify that a solution is a valid magic square.

    Checks:
    1. All values are distinct and in range [1, n²]
    2. All row sums equal the magic constant
    3. All column sums equal the magic constant
    4. Both diagonal sums equal the magic constant

    Args:
        grid: The variable grid.
        solution: The solution to verify.

    Returns:
        True if the solution is a valid magic square.
    """
    n = len(grid)
    m = magic_constant(n)
    n_sq = n * n

    # Extract values
    values: list[list[int]] = []
    all_values: list[int] = []
    for i in range(n):
        row_vals: list[int] = []
        for j in range(n):
            val = solution[grid[i][j]]
            row_vals.append(val)
            all_values.append(val)
        values.append(row_vals)

    # Check distinct values in range
    if sorted(all_values) != list(range(1, n_sq + 1)):
        return False

    # Check row sums
    for i in range(n):
        if sum(values[i]) != m:
            return False

    # Check column sums
    for j in range(n):
        col_sum = sum(values[i][j] for i in range(n))
        if col_sum != m:
            return False

    # Check diagonal sums
    diag1_sum = sum(values[i][i] for i in range(n))
    if diag1_sum != m:
        return False

    diag2_sum = sum(values[i][n - 1 - i] for i in range(n))
    if diag2_sum != m:
        return False

    return True


def main() -> None:
    """Run the magic square solver for 3×3 and 4×4."""
    print("=" * 50)
    print("SolveEngine Magic Square Example")
    print("=" * 50)

    # Solve 3×3 magic square
    print("\nSolving 3×3 magic square...")
    print(f"Magic constant: {magic_constant(3)}")

    grid_3, constraints_3 = create_magic_square(3)
    all_vars_3 = [var for row in grid_3 for var in row]

    solver_3 = BacktrackSolver(
        all_vars_3,
        constraints_3,
        var_selector=DomWdegSelector(),
        val_orderer=LCVOrderer(),
        use_forward_check=True,
    )
    solution_3 = solver_3.solve()

    if solution_3:
        print(display_magic_square(grid_3, solution_3))
        valid = verify_magic_square(grid_3, solution_3)
        print(f"Valid: {valid}")
        print(f"Nodes explored: {solver_3.stats.nodes_explored}")
    else:
        print("No solution found!")

    # Solve 4×4 magic square
    print("\n" + "=" * 50)
    print("\nSolving 4×4 magic square...")
    print(f"Magic constant: {magic_constant(4)}")

    grid_4, constraints_4 = create_magic_square(4)
    all_vars_4 = [var for row in grid_4 for var in row]

    solver_4 = BacktrackSolver(
        all_vars_4,
        constraints_4,
        var_selector=DomWdegSelector(),
        val_orderer=LCVOrderer(),
        use_forward_check=True,
    )
    solver_4.set_node_limit(500000)
    solution_4 = solver_4.solve()

    if solution_4:
        print(display_magic_square(grid_4, solution_4))
        valid = verify_magic_square(grid_4, solution_4)
        print(f"Valid: {valid}")
        print(f"Nodes explored: {solver_4.stats.nodes_explored}")
    else:
        print("No solution found within node limit.")
        print(f"Nodes explored: {solver_4.stats.nodes_explored}")

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
