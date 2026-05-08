"""Hypertree decomposition for non-binary constraint networks.

Extends tree decomposition to hypergraphs where constraints can involve
more than two variables. A hypertree decomposition assigns each bag a
set of hyperedges (constraints) whose variables cover the bag, enabling
efficient solving of CSPs with bounded hypertree width.

Hypertree width is a more refined measure than treewidth for constraint
networks with non-binary constraints, as it accounts for the structure
of the hyperedges rather than just the primal graph.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


@dataclass
class Hyperedge:
    """A hyperedge in the constraint hypergraph.

    Each hyperedge corresponds to a constraint and contains all
    variables in that constraint's scope.
    """

    edge_id: int
    variables: frozenset[int]  # Variable indices
    constraint: Constraint

    def covers(self, var_indices: set[int]) -> bool:
        """Check if this hyperedge covers all given variable indices."""
        return var_indices.issubset(self.variables)

    def intersects(self, var_indices: set[int]) -> bool:
        """Check if this hyperedge shares any variables with the given set."""
        return bool(self.variables & var_indices)


@dataclass
class HypertreeNode:
    """A node in the hypertree decomposition.

    Each node has:
    - A bag of variables (chi)
    - A set of covering hyperedges (lambda)
    - Parent/children forming the tree structure
    """

    node_id: int
    bag: set[int] = field(default_factory=set)  # Variable indices in chi
    covering_edges: list[Hyperedge] = field(default_factory=list)  # Lambda
    children: list[HypertreeNode] = field(default_factory=list)
    parent: HypertreeNode | None = None

    @property
    def width(self) -> int:
        """Width of this node = number of covering hyperedges."""
        return len(self.covering_edges)

    @property
    def bag_size(self) -> int:
        """Number of variables in the bag."""
        return len(self.bag)

    def covered_variables(self) -> set[int]:
        """Variables covered by the lambda (covering edges)."""
        covered: set[int] = set()
        for edge in self.covering_edges:
            covered.update(edge.variables)
        return covered

    def is_valid_cover(self) -> bool:
        """Check if the covering edges actually cover the bag."""
        return self.bag.issubset(self.covered_variables())


class ConstraintHypergraph:
    """Hypergraph representation of a constraint network.

    Nodes are variables, hyperedges are constraints (each connecting
    all variables in the constraint's scope).
    """

    def __init__(
        self, variables: list[Variable], constraints: list[Constraint]
    ) -> None:
        self._variables = variables
        self._constraints = constraints
        self._var_by_index: dict[int, Variable] = {
            v.index: v for v in variables
        }
        self._hyperedges: list[Hyperedge] = []
        self._var_to_edges: dict[int, list[int]] = {}
        self._build()

    def _build(self) -> None:
        """Build the hypergraph from constraints."""
        for i, cstr in enumerate(self._constraints):
            var_indices = frozenset(v.index for v in cstr.variables)
            edge = Hyperedge(edge_id=i, variables=var_indices, constraint=cstr)
            self._hyperedges.append(edge)
            for vi in var_indices:
                self._var_to_edges.setdefault(vi, []).append(i)

    @property
    def num_nodes(self) -> int:
        return len(self._variables)

    @property
    def num_edges(self) -> int:
        return len(self._hyperedges)

    @property
    def hyperedges(self) -> list[Hyperedge]:
        return self._hyperedges

    def edges_for_variable(self, var_index: int) -> list[Hyperedge]:
        """Get all hyperedges containing a variable."""
        edge_ids = self._var_to_edges.get(var_index, [])
        return [self._hyperedges[eid] for eid in edge_ids]

    def neighbors(self, var_index: int) -> set[int]:
        """Get all variable indices connected to var_index via any hyperedge."""
        neighbor_set: set[int] = set()
        for edge in self.edges_for_variable(var_index):
            neighbor_set.update(edge.variables)
        neighbor_set.discard(var_index)
        return neighbor_set

    def primal_edges(self) -> list[tuple[int, int]]:
        """Get edges of the primal graph (pairs of variables sharing a constraint)."""
        edges: set[tuple[int, int]] = set()
        for hyperedge in self._hyperedges:
            var_list = sorted(hyperedge.variables)
            for i in range(len(var_list)):
                for j in range(i + 1, len(var_list)):
                    edges.add((var_list[i], var_list[j]))
        return sorted(edges)

    def max_edge_size(self) -> int:
        """Maximum number of variables in any single hyperedge."""
        if not self._hyperedges:
            return 0
        return max(len(e.variables) for e in self._hyperedges)

    def find_covering_edges(self, var_indices: set[int]) -> list[Hyperedge]:
        """Find a minimal set of hyperedges that covers all given variables.

        Uses a greedy set cover approximation.
        """
        uncovered = set(var_indices)
        covering: list[Hyperedge] = []

        while uncovered:
            # Find the edge that covers the most uncovered variables
            best_edge: Hyperedge | None = None
            best_cover_count = 0

            for edge in self._hyperedges:
                cover_count = len(edge.variables & uncovered)
                if cover_count > best_cover_count:
                    best_cover_count = cover_count
                    best_edge = edge

            if best_edge is None or best_cover_count == 0:
                break  # Cannot cover remaining variables

            covering.append(best_edge)
            uncovered -= best_edge.variables

        return covering


class HypertreeDecomposition:
    """Hypertree decomposition of a constraint hypergraph.

    Computes a tree decomposition where each bag is annotated with
    covering hyperedges. The hypertree width is the maximum number
    of covering edges at any node.

    Uses a greedy approach based on elimination ordering on the
    primal graph, then assigns covering edges to each bag.
    """

    def __init__(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> None:
        self._variables = variables
        self._constraints = constraints
        self._hypergraph = ConstraintHypergraph(variables, constraints)
        self._root: HypertreeNode | None = None
        self._nodes: list[HypertreeNode] = []
        self._width: int = 0

    @property
    def root(self) -> HypertreeNode | None:
        return self._root

    @property
    def hypertree_width(self) -> int:
        """Hypertree width of the decomposition."""
        return self._width

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def hypergraph(self) -> ConstraintHypergraph:
        return self._hypergraph

    def decompose(self) -> None:
        """Compute the hypertree decomposition.

        Steps:
        1. Compute elimination ordering on the primal graph
        2. Build bags from the elimination ordering
        3. Assign covering hyperedges to each bag
        4. Connect bags into a tree
        """
        ordering = self._compute_elimination_ordering()
        bags = self._build_bags(ordering)
        self._build_tree(bags)
        self._assign_covers()
        self._compute_width()

    def _compute_elimination_ordering(self) -> list[int]:
        """Compute a min-fill elimination ordering on the primal graph."""
        # Build adjacency for primal graph
        adj: dict[int, set[int]] = {v.index: set() for v in self._variables}
        for edge in self._hypergraph.hyperedges:
            var_list = list(edge.variables)
            for i in range(len(var_list)):
                for j in range(i + 1, len(var_list)):
                    adj[var_list[i]].add(var_list[j])
                    adj[var_list[j]].add(var_list[i])

        remaining = set(v.index for v in self._variables)
        ordering: list[int] = []

        while remaining:
            # Min-fill: eliminate variable adding fewest fill edges
            best_var = -1
            best_fill = float("inf")

            for var_idx in remaining:
                fill = self._count_fill_edges(var_idx, adj, remaining)
                if fill < best_fill:
                    best_fill = fill
                    best_var = var_idx

            # Add fill edges
            neighbors_in_remaining = adj[best_var] & remaining
            for n1 in neighbors_in_remaining:
                for n2 in neighbors_in_remaining:
                    if n1 != n2:
                        adj[n1].add(n2)
                        adj[n2].add(n1)

            remaining.discard(best_var)
            ordering.append(best_var)

        return ordering

    def _count_fill_edges(
        self, var_idx: int, adj: dict[int, set[int]], remaining: set[int]
    ) -> int:
        """Count fill edges that would be added by eliminating var_idx."""
        neighbors = list(adj[var_idx] & remaining)
        fill = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] not in adj[neighbors[i]]:
                    fill += 1
        return fill

    def _build_bags(self, ordering: list[int]) -> list[set[int]]:
        """Build bags from the elimination ordering."""
        adj: dict[int, set[int]] = {v.index: set() for v in self._variables}
        for edge in self._hypergraph.hyperedges:
            var_list = list(edge.variables)
            for i in range(len(var_list)):
                for j in range(i + 1, len(var_list)):
                    adj[var_list[i]].add(var_list[j])
                    adj[var_list[j]].add(var_list[i])

        remaining = set(v.index for v in self._variables)
        bags: list[set[int]] = []

        for var_idx in ordering:
            neighbors_in_remaining = adj[var_idx] & remaining
            bag = {var_idx} | neighbors_in_remaining

            # Add fill edges
            for n1 in neighbors_in_remaining:
                for n2 in neighbors_in_remaining:
                    if n1 != n2:
                        adj[n1].add(n2)
                        adj[n2].add(n1)

            remaining.discard(var_idx)
            bags.append(bag)

        return bags

    def _build_tree(self, bags: list[set[int]]) -> None:
        """Build tree structure from bags."""
        self._nodes = []
        for i, bag in enumerate(bags):
            node = HypertreeNode(node_id=i, bag=bag)
            self._nodes.append(node)

        # Connect: each bag connects to the first later bag sharing a variable
        for i in range(len(self._nodes) - 1):
            for j in range(i + 1, len(self._nodes)):
                if self._nodes[i].bag & self._nodes[j].bag:
                    self._nodes[i].parent = self._nodes[j]
                    self._nodes[j].children.append(self._nodes[i])
                    break

        if self._nodes:
            self._root = self._nodes[-1]

    def _assign_covers(self) -> None:
        """Assign covering hyperedges to each bag using greedy set cover."""
        for node in self._nodes:
            covering = self._hypergraph.find_covering_edges(node.bag)
            node.covering_edges = covering

    def _compute_width(self) -> None:
        """Compute the hypertree width (max covering edges at any node)."""
        if not self._nodes:
            self._width = 0
        else:
            self._width = max(node.width for node in self._nodes)

    def validate(self) -> bool:
        """Validate the hypertree decomposition.

        Checks:
        1. Every variable appears in at least one bag
        2. Every constraint's variables appear together in some bag
        3. The running intersection property holds
        4. Each bag is covered by its lambda (covering edges)
        """
        all_var_indices = {v.index for v in self._variables}

        # Check 1: all variables covered
        covered_vars: set[int] = set()
        for node in self._nodes:
            covered_vars.update(node.bag)
        if not all_var_indices.issubset(covered_vars):
            return False

        # Check 2: constraint coverage
        for edge in self._hypergraph.hyperedges:
            found = False
            for node in self._nodes:
                if edge.variables.issubset(node.bag):
                    found = True
                    break
            if not found:
                return False

        # Check 3: running intersection property
        for var_idx in all_var_indices:
            nodes_with_var = [n for n in self._nodes if var_idx in n.bag]
            if not self._is_connected_subtree(nodes_with_var):
                return False

        # Check 4: covering validity
        for node in self._nodes:
            if not node.is_valid_cover():
                return False

        return True

    def _is_connected_subtree(self, nodes: list[HypertreeNode]) -> bool:
        """Check if a set of nodes forms a connected subtree."""
        if len(nodes) <= 1:
            return True

        node_ids = {n.node_id for n in nodes}
        # BFS from first node, only following edges within the set
        visited: set[int] = set()
        queue = deque([nodes[0].node_id])
        visited.add(nodes[0].node_id)

        while queue:
            current_id = queue.popleft()
            current_node = self._nodes[current_id]

            # Check parent
            if current_node.parent and current_node.parent.node_id in node_ids:
                if current_node.parent.node_id not in visited:
                    visited.add(current_node.parent.node_id)
                    queue.append(current_node.parent.node_id)

            # Check children
            for child in current_node.children:
                if child.node_id in node_ids and child.node_id not in visited:
                    visited.add(child.node_id)
                    queue.append(child.node_id)

        return len(visited) == len(node_ids)

    def get_bag_for_constraint(self, constraint: Constraint) -> HypertreeNode | None:
        """Find the bag that contains all variables of a constraint."""
        var_indices = {v.index for v in constraint.variables}
        for node in self._nodes:
            if var_indices.issubset(node.bag):
                return node
        return None

    def get_separator(
        self, node: HypertreeNode
    ) -> set[int]:
        """Get the separator between a node and its parent.

        The separator is the intersection of the node's bag with its parent's bag.
        """
        if node.parent is None:
            return set()
        return node.bag & node.parent.bag
