"""Tests for the Circuit (Hamiltonian cycle) global constraint.

Tests cover:
- Basic cycle detection and validation
- Subtour elimination propagation
- Self-loop removal
- Reachability computation
- Chain endpoint detection
- Edge cases (2-node, 3-node circuits)
- Integration with the solver
"""

from __future__ import annotations

import pytest

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.global_cstr.circuit import CircuitConstraint
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.heuristics.variable_ordering import MRVSelector
from solveengine.heuristics.value_ordering import AscendingOrderer


class TestCircuitSatisfaction:
    """Tests for is_satisfied method."""

    def test_valid_cycle_3_nodes(self) -> None:
        """A valid 3-node cycle: 0->1->2->0."""
        vars_ = [Variable(f"x{i}", range(3)) for i in range(3)]
        circuit = CircuitConstraint(vars_)
        # x[0]=1, x[1]=2, x[2]=0 means 0->1->2->0
        assignment = {vars_[0]: 1, vars_[1]: 2, vars_[2]: 0}
        assert circuit.is_satisfied(assignment) is True

    def test_valid_cycle_4_nodes(self) -> None:
        """A valid 4-node cycle: 0->2->3->1->0."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)
        assignment = {vars_[0]: 2, vars_[1]: 0, vars_[2]: 3, vars_[3]: 1}
        assert circuit.is_satisfied(assignment) is True

    def test_self_loop_rejected(self) -> None:
        """Self-loop x[i]=i is always invalid."""
        vars_ = [Variable(f"x{i}", range(3)) for i in range(3)]
        circuit = CircuitConstraint(vars_)
        assignment = {vars_[0]: 0, vars_[1]: 2, vars_[2]: 1}
        assert circuit.is_satisfied(assignment) is False

    def test_duplicate_values_rejected(self) -> None:
        """Two variables with the same successor is invalid."""
        vars_ = [Variable(f"x{i}", range(3)) for i in range(3)]
        circuit = CircuitConstraint(vars_)
        assignment = {vars_[0]: 1, vars_[1]: 1, vars_[2]: 0}
        assert circuit.is_satisfied(assignment) is False

    def test_subtour_rejected(self) -> None:
        """A subtour (not visiting all nodes) is invalid."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)
        # 0->1->0 is a subtour, 2->3->2 is another
        assignment = {vars_[0]: 1, vars_[1]: 0, vars_[2]: 3, vars_[3]: 2}
        assert circuit.is_satisfied(assignment) is False

    def test_partial_assignment_no_violation(self) -> None:
        """Partial assignment that doesn't violate yet."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)
        assignment = {vars_[0]: 1, vars_[1]: 2}
        assert circuit.is_satisfied(assignment) is True

    def test_partial_assignment_with_subtour(self) -> None:
        """Partial assignment with two disjoint cycles among assigned nodes."""
        vars_ = [Variable(f"x{i}", range(6)) for i in range(6)]
        circuit = CircuitConstraint(vars_)
        # Nodes 0->1->0 and 2->3->2 form two subtours among 4 assigned nodes
        # _has_subtour detects cycle of length 2 < 4 assigned nodes
        assignment = {vars_[0]: 1, vars_[1]: 0, vars_[2]: 3, vars_[3]: 2}
        assert circuit.is_satisfied(assignment) is False

    def test_out_of_range_value_rejected(self) -> None:
        """Values outside [0, n-1] are invalid."""
        vars_ = [Variable(f"x{i}", range(5)) for i in range(3)]
        circuit = CircuitConstraint(vars_)
        assignment = {vars_[0]: 4, vars_[1]: 2, vars_[2]: 0}
        assert circuit.is_satisfied(assignment) is False

    def test_two_node_cycle(self) -> None:
        """Minimum valid circuit: 0->1->0."""
        vars_ = [Variable(f"x{i}", range(2)) for i in range(2)]
        circuit = CircuitConstraint(vars_)
        assignment = {vars_[0]: 1, vars_[1]: 0}
        assert circuit.is_satisfied(assignment) is True


class TestCircuitPropagation:
    """Tests for propagation methods."""

    def test_no_self_loops_propagation(self) -> None:
        """Self-loop values should be removed from all domains."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)
        pruned = circuit.propagate_no_self_loops()

        # Each variable should have its own index removed
        for i, var in enumerate(vars_):
            assert i not in var.domain.values()
            assert var in pruned
            assert pruned[var] == [i]

    def test_subtour_elimination_removes_closing_value(self) -> None:
        """Subtour elimination should remove self-loops and used values."""
        vars_ = [Variable(f"x{i}", range(5)) for i in range(5)]
        circuit = CircuitConstraint(vars_)

        # Assign 0->1, 1->2. Propagation should remove:
        # - Self-loops (x2!=2, x3!=3, x4!=4)
        # - Used values (1 and 2 from unassigned vars)
        assignment = {vars_[0]: 1, vars_[1]: 2}
        pruned = circuit.propagate_subtour_elimination(assignment)

        # x2 should not have self-loop value 2 or used value 1
        assert 2 not in vars_[2].domain.values()
        assert 1 not in vars_[2].domain.values()
        # x3 should not have self-loop 3 or used values 1, 2
        assert 3 not in vars_[3].domain.values()
        assert 1 not in vars_[3].domain.values()
        assert 2 not in vars_[3].domain.values()

    def test_subtour_elimination_preserves_valid_values(self) -> None:
        """Valid successor values should not be pruned."""
        vars_ = [Variable(f"x{i}", range(5)) for i in range(5)]
        circuit = CircuitConstraint(vars_)

        assignment = {vars_[0]: 1, vars_[1]: 2}
        circuit.propagate_subtour_elimination(assignment)

        # x[2] can still go to 3 or 4 (extends the chain)
        assert 3 in vars_[2].domain.values()
        assert 4 in vars_[2].domain.values()

    def test_used_values_removed(self) -> None:
        """Values already used as successors should be removed."""
        vars_ = [Variable(f"x{i}", range(5)) for i in range(5)]
        circuit = CircuitConstraint(vars_)

        assignment = {vars_[0]: 1, vars_[1]: 2}
        pruned = circuit.propagate_subtour_elimination(assignment)

        # x[3] and x[4] should not have values 1 or 2 (already used)
        assert 1 not in vars_[3].domain.values()
        assert 2 not in vars_[3].domain.values()
        assert 1 not in vars_[4].domain.values()
        assert 2 not in vars_[4].domain.values()


class TestCircuitSupportedValues:
    """Tests for get_supported_values method."""

    def test_self_loop_not_supported(self) -> None:
        """A variable's own index should never be supported."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)
        supported = circuit.get_supported_values(vars_[0], {})
        assert 0 not in supported

    def test_used_values_not_supported(self) -> None:
        """Values already assigned to other variables are not supported."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)
        assignment = {vars_[1]: 2}
        supported = circuit.get_supported_values(vars_[0], assignment)
        assert 2 not in supported

    def test_valid_successors_supported(self) -> None:
        """Valid successor values should be in the supported set."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)
        supported = circuit.get_supported_values(vars_[0], {})
        # x[0] can go to 1, 2, or 3 (not 0)
        assert 1 in supported
        assert 2 in supported
        assert 3 in supported


class TestCircuitReachability:
    """Tests for reachability computation."""

    def test_empty_assignment_no_reachability(self) -> None:
        """With no assignments, no nodes are reachable."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)
        reachable = circuit.compute_reachability({})
        for node in range(4):
            assert reachable[node] == set()

    def test_single_assignment_reachability(self) -> None:
        """Single assignment creates one-step reachability."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)
        assignment = {vars_[0]: 2}
        reachable = circuit.compute_reachability(assignment)
        assert 2 in reachable[0]

    def test_chain_reachability(self) -> None:
        """Chain of assignments creates transitive reachability."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)
        assignment = {vars_[0]: 1, vars_[1]: 2, vars_[2]: 3}
        reachable = circuit.compute_reachability(assignment)
        assert reachable[0] == {1, 2, 3}
        assert reachable[1] == {2, 3}
        assert reachable[2] == {3}


class TestCircuitChainEndpoints:
    """Tests for chain endpoint detection."""

    def test_no_chains_empty_assignment(self) -> None:
        """No chains with empty assignment."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)
        chains = circuit.find_chain_endpoints({})
        assert chains == []

    def test_single_chain(self) -> None:
        """Single chain 0->1->2."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)
        assignment = {vars_[0]: 1, vars_[1]: 2}
        chains = circuit.find_chain_endpoints(assignment)
        assert (0, 2) in chains

    def test_multiple_chains(self) -> None:
        """Two separate chains."""
        vars_ = [Variable(f"x{i}", range(6)) for i in range(6)]
        circuit = CircuitConstraint(vars_)
        assignment = {vars_[0]: 1, vars_[2]: 3}
        chains = circuit.find_chain_endpoints(assignment)
        assert len(chains) == 2


class TestCircuitIntegration:
    """Integration tests with the solver."""

    def test_solve_3_node_circuit(self) -> None:
        """Solver should find a valid 3-node Hamiltonian cycle."""
        vars_ = [Variable(f"x{i}", range(3)) for i in range(3)]
        circuit = CircuitConstraint(vars_)

        solver = BacktrackSolver(
            vars_,
            [circuit],
            var_selector=MRVSelector(),
            val_orderer=AscendingOrderer(),
            use_forward_check=True,
        )
        solution = solver.solve()
        assert solution is not None
        assert circuit.is_satisfied(solution)

    def test_solve_4_node_circuit(self) -> None:
        """Solver should find a valid 4-node Hamiltonian cycle."""
        vars_ = [Variable(f"x{i}", range(4)) for i in range(4)]
        circuit = CircuitConstraint(vars_)

        solver = BacktrackSolver(
            vars_,
            [circuit],
            var_selector=MRVSelector(),
            val_orderer=AscendingOrderer(),
            use_forward_check=True,
        )
        solution = solver.solve()
        assert solution is not None
        assert circuit.is_satisfied(solution)

        # Verify it's a single cycle
        visited = set()
        current = 0
        for _ in range(4):
            assert current not in visited
            visited.add(current)
            current = solution[vars_[current]]
        assert current == 0
        assert len(visited) == 4

    def test_all_solutions_3_nodes(self) -> None:
        """3-node circuit has exactly 2 Hamiltonian cycles."""
        vars_ = [Variable(f"x{i}", range(3)) for i in range(3)]
        circuit = CircuitConstraint(vars_)

        solver = BacktrackSolver(
            vars_,
            [circuit],
            var_selector=MRVSelector(),
            val_orderer=AscendingOrderer(),
            use_forward_check=True,
        )
        solutions = solver.solve_all()
        # 3-node has 2 directed Hamiltonian cycles: 0->1->2->0 and 0->2->1->0
        assert len(solutions) == 2

    def test_minimum_circuit_size(self) -> None:
        """Circuit requires at least 2 nodes."""
        with pytest.raises(ValueError):
            vars_ = [Variable("x0", range(1))]
            CircuitConstraint(vars_)

    def test_num_nodes_property(self) -> None:
        """num_nodes should match the number of variables."""
        vars_ = [Variable(f"x{i}", range(5)) for i in range(5)]
        circuit = CircuitConstraint(vars_)
        assert circuit.num_nodes == 5
