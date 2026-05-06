"""Constraint graph representation.

The constraint graph has variables as nodes and edges between variables
that share a constraint. This structure is used for decomposition,
connected component detection, and ordering heuristics.
"""

from __future__ import annotations

from collections import deque

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class ConstraintGraph:
    """Graph representation of constraint relationships between variables.

    Nodes are variables, edges connect variables that appear together
    in at least one constraint.
    """

    def __init__(self, variables: list[Variable], constraints: list[Constraint]) -> None:
        self._variables = variables
        self._constraints = constraints
        self._adjacency: dict[int, set[int]] = {}  # var.index -> set of neighbor indices
        self._var_by_index: dict[int, Variable] = {}
        self._build()

    def _build(self) -> None:
        """Build adjacency list from constraints."""
        for var in self._variables:
            self._adjacency[var.index] = set()
            self._var_by_index[var.index] = var

        for cstr in self._constraints:
            vars_in_cstr = cstr.variables
            for i in range(len(vars_in_cstr)):
                for j in range(i + 1, len(vars_in_cstr)):
                    vi = vars_in_cstr[i].index
                    vj = vars_in_cstr[j].index
                    self._adjacency[vi].add(vj)
                    self._adjacency[vj].add(vi)

    @property
    def num_nodes(self) -> int:
        return len(self._variables)

    @property
    def num_edges(self) -> int:
        total = sum(len(neighbors) for neighbors in self._adjacency.values())
        return total // 2

    def neighbors(self, var: Variable) -> list[Variable]:
        """Get all variables connected to var by a constraint."""
        return [
            self._var_by_index[idx]
            for idx in self._adjacency.get(var.index, set())
        ]

    def degree(self, var: Variable) -> int:
        """Number of neighbors of var in the constraint graph."""
        return len(self._adjacency.get(var.index, set()))

    def connected_components(self) -> list[list[Variable]]:
        """Find connected components in the constraint graph.

        Independent components can be solved separately.
        """
        visited: set[int] = set()
        components: list[list[Variable]] = []

        for var in self._variables:
            if var.index in visited:
                continue
            component = self._bfs_component(var.index, visited)
            components.append(component)

        return components

    def _bfs_component(self, start: int, visited: set[int]) -> list[Variable]:
        """BFS to find all nodes in the component containing start."""
        queue = deque([start])
        visited.add(start)
        component = []

        while queue:
            node = queue.popleft()
            component.append(self._var_by_index[node])
            for neighbor in self._adjacency.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return component

    def is_tree(self) -> bool:
        """Check if the constraint graph is a tree (acyclic and connected).

        Tree-structured CSPs can be solved in polynomial time.
        """
        components = self.connected_components()
        if len(components) != 1:
            return False
        # A tree has exactly n-1 edges
        return self.num_edges == self.num_nodes - 1

    def density(self) -> float:
        """Graph density: ratio of actual edges to maximum possible edges."""
        n = self.num_nodes
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1) / 2
        return self.num_edges / max_edges

    def bandwidth(self) -> int:
        """Compute the bandwidth of the constraint graph.

        Bandwidth is the maximum difference between indices of adjacent nodes.
        Lower bandwidth means better locality for propagation.
        """
        max_diff = 0
        for var_idx, neighbors in self._adjacency.items():
            for neighbor_idx in neighbors:
                diff = abs(var_idx - neighbor_idx)
                if diff > max_diff:
                    max_diff = diff
        return max_diff

    def cutset(self) -> list[Variable]:
        """Find a small cycle cutset (greedy approximation).

        A cycle cutset is a set of variables whose removal makes the
        graph acyclic. Useful for hybrid solving approaches.
        """
        # Greedy: repeatedly remove the highest-degree node until acyclic
        remaining = set(var.index for var in self._variables)
        adj_copy = {k: set(v) for k, v in self._adjacency.items()}
        cutset_indices: list[int] = []

        while not self._is_acyclic(remaining, adj_copy):
            # Find highest degree node in remaining
            max_deg = -1
            max_node = -1
            for node in remaining:
                deg = len(adj_copy.get(node, set()) & remaining)
                if deg > max_deg:
                    max_deg = deg
                    max_node = node

            if max_node == -1:
                break

            remaining.discard(max_node)
            cutset_indices.append(max_node)

        return [self._var_by_index[idx] for idx in cutset_indices]

    def _is_acyclic(self, nodes: set[int], adj: dict[int, set[int]]) -> bool:
        """Check if the subgraph induced by nodes is acyclic."""
        if not nodes:
            return True

        visited: set[int] = set()

        for start in nodes:
            if start in visited:
                continue
            # BFS with parent tracking
            queue = deque([(start, -1)])
            visited.add(start)
            while queue:
                node, parent = queue.popleft()
                for neighbor in adj.get(node, set()):
                    if neighbor not in nodes:
                        continue
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        return False  # Cycle detected
                    visited.add(neighbor)
                    queue.append((neighbor, node))

        return True
