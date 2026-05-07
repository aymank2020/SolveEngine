"""Tree decomposition for constraint graphs.

Tree decomposition breaks a constraint graph into overlapping clusters
(bags) arranged in a tree structure. The width of the decomposition
determines the complexity of solving the CSP on the decomposed structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.decomposition.graph import ConstraintGraph


@dataclass
class TreeNode:
    """A node (bag) in the tree decomposition."""

    bag_id: int
    variables: set[Variable] = field(default_factory=set)
    children: list[TreeNode] = field(default_factory=list)
    parent: TreeNode | None = None

    @property
    def width(self) -> int:
        """Width of this bag (number of variables - 1)."""
        return len(self.variables) - 1

    def __repr__(self) -> str:
        var_names = ", ".join(v.name for v in self.variables)
        return f"TreeNode({self.bag_id}: {{{var_names}}})"


class TreeDecomposition:
    """Tree decomposition of a constraint graph.

    Uses a greedy min-fill heuristic to compute an elimination ordering,
    then builds the tree decomposition from that ordering.
    """

    def __init__(self, graph: ConstraintGraph, variables: list[Variable]) -> None:
        self._graph = graph
        self._variables = variables
        self._root: TreeNode | None = None
        self._nodes: list[TreeNode] = []
        self._width: int = 0

    @property
    def root(self) -> TreeNode | None:
        return self._root

    @property
    def width(self) -> int:
        """Treewidth of the decomposition."""
        return self._width

    @property
    def num_bags(self) -> int:
        return len(self._nodes)

    def decompose(self) -> None:
        """Compute the tree decomposition using min-fill elimination."""
        ordering = self._min_fill_ordering()
        self._build_from_ordering(ordering)

    def _min_fill_ordering(self) -> list[Variable]:
        """Compute elimination ordering using min-fill heuristic.

        At each step, eliminate the variable that would add the fewest
        fill edges (edges between its neighbors that don't already exist).
        """
        # Build working adjacency
        adj: dict[int, set[int]] = {}
        for var in self._variables:
            adj[var.index] = set()
        for var in self._variables:
            for neighbor in self._graph.neighbors(var):
                adj[var.index].add(neighbor.index)
                adj[neighbor.index].add(var.index)

        remaining = set(var.index for var in self._variables)
        var_by_index = {var.index: var for var in self._variables}
        ordering: list[Variable] = []

        while remaining:
            # Find variable with minimum fill
            best_var = -1
            best_fill = float("inf")

            for var_idx in remaining:
                fill = self._count_fill(var_idx, adj, remaining)
                if fill < best_fill:
                    best_fill = fill
                    best_var = var_idx

            # Eliminate: add fill edges and remove from graph
            neighbors_in_remaining = adj[best_var] & remaining
            for n1 in neighbors_in_remaining:
                for n2 in neighbors_in_remaining:
                    if n1 != n2:
                        adj[n1].add(n2)
                        adj[n2].add(n1)

            remaining.discard(best_var)
            ordering.append(var_by_index[best_var])

        return ordering

    def _count_fill(self, var_idx: int, adj: dict[int, set[int]], remaining: set[int]) -> int:
        """Count fill edges that would be added by eliminating var_idx."""
        neighbors = adj[var_idx] & remaining
        fill = 0
        neighbor_list = list(neighbors)
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                if neighbor_list[j] not in adj[neighbor_list[i]]:
                    fill += 1
        return fill

    def _build_from_ordering(self, ordering: list[Variable]) -> None:
        """Build tree decomposition from elimination ordering."""
        # Build bags from elimination ordering
        adj: dict[int, set[int]] = {}
        for var in self._variables:
            adj[var.index] = set()
        for var in self._variables:
            for neighbor in self._graph.neighbors(var):
                adj[var.index].add(neighbor.index)
                adj[neighbor.index].add(var.index)

        var_by_index = {var.index: var for var in self._variables}
        bags: list[set[Variable]] = []
        remaining = set(var.index for var in self._variables)

        for var in ordering:
            neighbors_in_remaining = adj[var.index] & remaining
            bag = {var} | {var_by_index[n] for n in neighbors_in_remaining}
            bags.append(bag)

            # Add fill edges
            for n1 in neighbors_in_remaining:
                for n2 in neighbors_in_remaining:
                    if n1 != n2:
                        adj[n1].add(n2)
                        adj[n2].add(n1)

            remaining.discard(var.index)

        # Create tree nodes
        self._nodes = []
        for i, bag in enumerate(bags):
            node = TreeNode(bag_id=i, variables=bag)
            self._nodes.append(node)

        # Connect nodes into a tree (each bag connects to the first later bag
        # that contains at least one of its variables)
        for i in range(len(self._nodes) - 1):
            for j in range(i + 1, len(self._nodes)):
                if self._nodes[i].variables & self._nodes[j].variables:
                    self._nodes[i].parent = self._nodes[j]
                    self._nodes[j].children.append(self._nodes[i])
                    break

        # Root is the last node (no parent)
        if self._nodes:
            self._root = self._nodes[-1]
            self._width = max(node.width for node in self._nodes)
