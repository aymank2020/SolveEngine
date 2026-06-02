"""Circuit (Hamiltonian cycle) global constraint.

The Circuit constraint ensures that a set of successor variables forms
a single Hamiltonian cycle visiting all nodes exactly once. Variable x[i]
represents the successor of node i in the cycle.

For n nodes: x[0], x[1], ..., x[n-1] where x[i] ∈ {0, ..., n-1}
The constraint requires:
1. All values are distinct (permutation)
2. Following successors from any node visits all nodes before returning

This is fundamental for TSP and vehicle routing problems.
"""

from __future__ import annotations

from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


class CircuitConstraint(Constraint):
    """Hamiltonian cycle constraint on successor variables.

    Each variable x[i] represents the successor of node i.
    The constraint ensures all variables form a single cycle
    covering all nodes.

    Args:
        successor_vars: Variables representing successor of each node.
            successor_vars[i] = j means node i's successor is node j.
        name: Optional constraint name.
    """

    def __init__(
        self,
        successor_vars: Sequence[Variable],
        name: str = "",
    ) -> None:
        if len(successor_vars) < 2:
            raise ValueError("Circuit constraint requires at least 2 nodes")
        super().__init__(list(successor_vars), name or "Circuit")
        self._n = len(successor_vars)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the circuit."""
        return self._n

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        """Check if the assignment forms a valid circuit.

        For partial assignments, checks:
        1. No self-loops (x[i] != i)
        2. All assigned values are distinct
        3. No subtours in the assigned portion

        For complete assignments, additionally verifies a single cycle.
        """
        assigned_successors: dict[int, int] = {}
        for i, var in enumerate(self._variables):
            if var in assignment:
                succ = assignment[var]
                if succ == i:
                    return False
                if succ < 0 or succ >= self._n:
                    return False
                assigned_successors[i] = succ

        values = list(assigned_successors.values())
        if len(values) != len(set(values)):
            return False

        if len(assigned_successors) == self._n:
            return self._is_single_cycle(assigned_successors)

        if len(assigned_successors) > 1:
            if self._has_subtour(assigned_successors):
                return False

        return True

    def _is_single_cycle(self, successors: dict[int, int]) -> bool:
        """Check if a complete successor mapping forms a single cycle."""
        visited = set()
        current = 0
        for _ in range(self._n):
            if current in visited:
                return False
            visited.add(current)
            current = successors[current]

        return len(visited) == self._n and current == 0

    def _has_subtour(self, successors: dict[int, int]) -> bool:
        """Check if partial assignment contains a subtour.

        A subtour is a cycle that doesn't include all assigned nodes.
        """
        assigned_nodes = set(successors.keys())
        if len(assigned_nodes) < 2:
            return False

        for start in assigned_nodes:
            visited: set[int] = set()
            current = start
            while current in successors:
                if current in visited:
                    if len(visited) < len(assigned_nodes):
                        return True
                    break
                visited.add(current)
                current = successors[current]

        return False

    def get_supported_values(self, var: Variable, assignment: dict[Variable, int]) -> set[int]:
        """Get successor values for var that don't violate the circuit.

        Filters out:
        1. Self-loops
        2. Values already used by other variables
        3. Values that would create a subtour
        """
        var_idx = list(self._variables).index(var)
        assigned_successors: dict[int, int] = {}
        used_values: set[int] = set()

        for i, v in enumerate(self._variables):
            if v is var:
                continue
            if v in assignment:
                assigned_successors[i] = assignment[v]
                used_values.add(assignment[v])

        supported: set[int] = set()
        for value in var.domain.values():
            if value == var_idx:
                continue
            if value in used_values:
                continue
            if value < 0 or value >= self._n:
                continue

            test_successors = dict(assigned_successors)
            test_successors[var_idx] = value

            if not self._has_subtour(test_successors):
                supported.add(value)

        return supported

    def propagate_no_self_loops(self) -> dict[Variable, list[int]]:
        """Remove self-loop values from all variable domains.

        For each variable x[i], remove value i from its domain.
        Returns mapping of variable to removed values.
        """
        pruned: dict[Variable, list[int]] = {}
        for i, var in enumerate(self._variables):
            if var.domain.contains(i):
                if var.domain.remove(i):
                    pruned[var] = [i]
        return pruned

    def propagate_subtour_elimination(
        self, assignment: dict[Variable, int]
    ) -> dict[Variable, list[int]]:
        """Apply subtour elimination propagation.

        For each unassigned variable, remove values that would create
        a subtour with the current partial assignment.
        """
        pruned: dict[Variable, list[int]] = {}

        assigned_successors: dict[int, int] = {}
        used_values: set[int] = set()
        for i, var in enumerate(self._variables):
            if var in assignment:
                assigned_successors[i] = assignment[var]
                used_values.add(assignment[var])

        for i, var in enumerate(self._variables):
            if var in assignment:
                continue

            removed: list[int] = []
            for value in list(var.domain):
                if value == i:
                    if var.domain.remove(value):
                        removed.append(value)
                    continue

                if value in used_values:
                    if var.domain.remove(value):
                        removed.append(value)
                    continue

                test_successors = dict(assigned_successors)
                test_successors[i] = value
                if self._would_create_subtour(test_successors, i):
                    if var.domain.remove(value):
                        removed.append(value)

            if removed:
                pruned[var] = removed

        return pruned

    def _would_create_subtour(self, successors: dict[int, int], new_node: int) -> bool:
        """Check if adding new_node's successor creates a subtour."""
        total_assigned = len(successors)
        if total_assigned < 2:
            return False

        visited: set[int] = set()
        current = new_node
        while current in successors:
            if current in visited:
                return len(visited) < total_assigned
            visited.add(current)
            current = successors[current]

        return False

    def compute_reachability(self, assignment: dict[Variable, int]) -> dict[int, set[int]]:
        """Compute reachability sets from the current partial assignment.

        Returns a mapping from each node to the set of nodes reachable
        from it by following assigned successors.
        """
        assigned_successors: dict[int, int] = {}
        for i, var in enumerate(self._variables):
            if var in assignment:
                assigned_successors[i] = assignment[var]

        reachable: dict[int, set[int]] = {}
        for node in range(self._n):
            reached: set[int] = set()
            current = node
            while current in assigned_successors:
                next_node = assigned_successors[current]
                if next_node in reached:
                    break
                reached.add(next_node)
                current = next_node
            reachable[node] = reached

        return reachable

    def find_chain_endpoints(self, assignment: dict[Variable, int]) -> list[tuple[int, int]]:
        """Find chain start and end nodes in the partial assignment.

        A chain is a maximal path in the partial successor graph.
        Returns list of (chain_start, chain_end) pairs.
        """
        assigned_successors: dict[int, int] = {}
        has_predecessor: set[int] = set()

        for i, var in enumerate(self._variables):
            if var in assignment:
                assigned_successors[i] = assignment[var]
                has_predecessor.add(assignment[var])

        chain_starts = [
            node for node in assigned_successors
            if node not in has_predecessor
        ]

        chains: list[tuple[int, int]] = []
        for start in chain_starts:
            current = start
            while current in assigned_successors:
                next_node = assigned_successors[current]
                if next_node == start:
                    break
                current = next_node
            chains.append((start, current))

        return chains
