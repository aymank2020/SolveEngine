"""Graph coloring problem using SolveEngine."""

from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.heuristics.variable_ordering import DomWdegSelector


def solve_coloring(
    num_nodes: int,
    edges: list[tuple[int, int]],
    num_colors: int,
) -> dict[int, int] | None:
    """Solve a graph coloring problem.

    Args:
        num_nodes: Number of nodes in the graph.
        edges: List of (node_i, node_j) edges.
        num_colors: Number of available colors.

    Returns:
        Dict mapping node -> color, or None if not colorable.
    """
    variables = [Variable(f"n{i}", range(1, num_colors + 1)) for i in range(num_nodes)]
    constraints = []

    for i, j in edges:
        constraints.append(BinaryConstraint(
            variables[i], variables[j], lambda a, b: a != b,
            name=f"n{i}!=n{j}",
        ))

    solver = BacktrackSolver(
        variables, constraints,
        var_selector=DomWdegSelector(),
        use_forward_check=True,
    )

    result = solver.solve()
    if result is None:
        return None

    return {i: result[variables[i]] for i in range(num_nodes)}


# Petersen graph (chromatic number = 3)
PETERSEN_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),  # Spokes
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner pentagram
]

if __name__ == "__main__":
    print("Coloring Petersen graph with 3 colors...")
    result = solve_coloring(10, PETERSEN_EDGES, 3)
    if result:
        for node, color in sorted(result.items()):
            print(f"  Node {node}: color {color}")
    else:
        print("Not 3-colorable")
