"""Regular language constraint for CSP problems.

The Regular constraint ensures that a sequence of variables forms a word
accepted by a given Deterministic Finite Automaton (DFA). This is a powerful
global constraint that can encode many sequential patterns including
shift scheduling, sequence constraints, and grammar-based restrictions.

Propagation uses state reachability: for each variable position, only values
that appear on transitions reachable from the initial state AND that can
reach an accepting state are kept in the domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint


@dataclass(frozen=True)
class DFATransition:
    """A single transition in the DFA.

    Attributes:
        source: Source state index.
        symbol: Input symbol (integer value).
        target: Target state index.
    """

    source: int
    symbol: int
    target: int


@dataclass
class DFA:
    """Deterministic Finite Automaton for the Regular constraint.

    States are represented as integers from 0 to num_states-1.
    Symbols correspond to integer values in variable domains.

    Attributes:
        num_states: Total number of states.
        initial_state: The starting state index.
        accepting_states: Set of accepting (final) state indices.
        transitions: List of all transitions.
    """

    num_states: int
    initial_state: int
    accepting_states: set[int]
    transitions: list[DFATransition] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Build transition lookup tables after initialization."""
        self._transition_map: dict[tuple[int, int], int] = {}
        self._outgoing: dict[int, dict[int, int]] = {}
        for trans in self.transitions:
            self._transition_map[(trans.source, trans.symbol)] = trans.target
            if trans.source not in self._outgoing:
                self._outgoing[trans.source] = {}
            self._outgoing[trans.source][trans.symbol] = trans.target

    def get_target(self, state: int, symbol: int) -> int | None:
        """Get the target state for a given state and symbol.

        Returns None if no transition exists.
        """
        return self._transition_map.get((state, symbol))

    def get_outgoing_symbols(self, state: int) -> dict[int, int]:
        """Get all outgoing transitions from a state.

        Returns a mapping from symbol to target state.
        """
        return self._outgoing.get(state, {})

    def get_states_reaching(self, target_states: set[int]) -> set[int]:
        """Find all states that can reach any of the target states in one step."""
        reaching: set[int] = set()
        for trans in self.transitions:
            if trans.target in target_states:
                reaching.add(trans.source)
        return reaching

    def is_valid(self) -> bool:
        """Check if the DFA is well-formed."""
        if self.initial_state < 0 or self.initial_state >= self.num_states:
            return False
        for state in self.accepting_states:
            if state < 0 or state >= self.num_states:
                return False
        for trans in self.transitions:
            if trans.source < 0 or trans.source >= self.num_states:
                return False
            if trans.target < 0 or trans.target >= self.num_states:
                return False
        return True


class RegularConstraint(Constraint):
    """Regular language constraint using DFA-based filtering.

    Ensures that the sequence of values assigned to the variables forms
    a word accepted by the given DFA. Variable at position i corresponds
    to the i-th symbol in the input word.

    Propagation computes forward-reachable states from the initial state
    and backward-reachable states from accepting states. A value is
    supported at position i only if there exists a transition using that
    value from a forward-reachable state at position i to a state that
    is backward-reachable at position i+1.

    Args:
        variables: Sequence of variables representing the word positions.
        automaton: The DFA that defines the accepted language.
        name: Optional constraint name.
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        automaton: DFA,
        name: str = "",
    ) -> None:
        if not automaton.is_valid():
            raise ValueError("Invalid DFA: states or transitions out of range")
        super().__init__(list(variables), name or "Regular")
        self._automaton = automaton
        self._n = len(variables)
        self._forward_states: list[set[int]] = []
        self._backward_states: list[set[int]] = []

    @property
    def automaton(self) -> DFA:
        """The DFA used by this constraint."""
        return self._automaton

    @property
    def sequence_length(self) -> int:
        """Length of the variable sequence."""
        return self._n

    def is_satisfied(self, assignment: dict[Variable, int]) -> bool:
        """Check if the assigned values form an accepted word.

        For partial assignments, checks that the assigned prefix can
        still reach an accepting state through the remaining positions.
        """
        # Collect assigned values in sequence order
        values: list[int | None] = []
        for var in self._variables:
            if var in assignment:
                values.append(assignment[var])
            else:
                values.append(None)

        # Simulate the DFA on the assigned prefix
        current_states: set[int] = {self._automaton.initial_state}

        for i, val in enumerate(values):
            if not current_states:
                return False
            if val is None:
                # Expand to all reachable states via any symbol in domain
                next_states: set[int] = set()
                for state in current_states:
                    outgoing = self._automaton.get_outgoing_symbols(state)
                    for symbol, target in outgoing.items():
                        if self._variables[i].domain.contains(symbol):
                            next_states.add(target)
                current_states = next_states
            else:
                next_states = set()
                for state in current_states:
                    target = self._automaton.get_target(state, val)
                    if target is not None:
                        next_states.add(target)
                current_states = next_states

        if not current_states:
            return False

        # Check if any current state is accepting (for complete assignment)
        all_assigned = all(v is not None for v in values)
        if all_assigned:
            return bool(current_states & self._automaton.accepting_states)

        # For partial: check if accepting states are reachable
        return self._can_reach_accepting(current_states, values)

    def _can_reach_accepting(
        self, current_states: set[int], values: list[int | None]
    ) -> bool:
        """Check if accepting states are reachable from current states."""
        # Find first unassigned position after the last assigned
        last_assigned = -1
        for i in range(len(values) - 1, -1, -1):
            if values[i] is not None:
                last_assigned = i
                break

        remaining_positions = self._n - last_assigned - 1
        if remaining_positions <= 0:
            return bool(current_states & self._automaton.accepting_states)

        # BFS forward through remaining positions
        reachable = current_states
        for i in range(last_assigned + 1, self._n):
            next_reachable: set[int] = set()
            for state in reachable:
                outgoing = self._automaton.get_outgoing_symbols(state)
                for symbol, target in outgoing.items():
                    if self._variables[i].domain.contains(symbol):
                        next_reachable.add(target)
            reachable = next_reachable
            if not reachable:
                return False

        return bool(reachable & self._automaton.accepting_states)

    def get_supported_values(
        self, var: Variable, assignment: dict[Variable, int]
    ) -> set[int]:
        """Get values for var consistent with the regular language.

        Uses forward/backward state reachability to determine which
        symbols at this position can be part of an accepted word.
        """
        var_idx = list(self._variables).index(var)
        forward = self._compute_forward_states(assignment)
        backward = self._compute_backward_states(assignment)

        supported: set[int] = set()
        states_at_position = forward[var_idx]
        states_needed_after = backward[var_idx + 1] if var_idx + 1 <= self._n else self._automaton.accepting_states

        for state in states_at_position:
            outgoing = self._automaton.get_outgoing_symbols(state)
            for symbol, target in outgoing.items():
                if target in states_needed_after:
                    if var.domain.contains(symbol):
                        supported.add(symbol)

        return supported

    def propagate(self, assignment: dict[Variable, int]) -> dict[Variable, list[int]]:
        """Propagate the regular constraint using layered graph filtering.

        Computes forward and backward reachable states, then removes
        values from domains that cannot participate in any accepted word.

        Returns a mapping from variable to list of pruned values.
        """
        forward = self._compute_forward_states(assignment)
        backward = self._compute_backward_states(assignment)
        pruned: dict[Variable, list[int]] = {}

        for i, var in enumerate(self._variables):
            if var in assignment:
                continue

            states_before = forward[i]
            states_after = backward[i + 1]
            removed: list[int] = []

            for value in list(var.domain.values()):
                # Check if there's a valid transition using this value
                has_support = False
                for state in states_before:
                    target = self._automaton.get_target(state, value)
                    if target is not None and target in states_after:
                        has_support = True
                        break

                if not has_support:
                    if var.domain.remove(value):
                        removed.append(value)

            if removed:
                pruned[var] = removed

        return pruned

    def _compute_forward_states(
        self, assignment: dict[Variable, int]
    ) -> list[set[int]]:
        """Compute forward-reachable states at each position.

        forward[i] = set of states reachable after processing positions 0..i-1.
        forward[0] = {initial_state}.
        """
        forward: list[set[int]] = [set() for _ in range(self._n + 1)]
        forward[0] = {self._automaton.initial_state}

        for i in range(self._n):
            var = self._variables[i]
            if var in assignment:
                # Only follow the assigned value
                val = assignment[var]
                for state in forward[i]:
                    target = self._automaton.get_target(state, val)
                    if target is not None:
                        forward[i + 1].add(target)
            else:
                # Follow all values in the domain
                for state in forward[i]:
                    outgoing = self._automaton.get_outgoing_symbols(state)
                    for symbol, target in outgoing.items():
                        if var.domain.contains(symbol):
                            forward[i + 1].add(target)

        return forward

    def _compute_backward_states(
        self, assignment: dict[Variable, int]
    ) -> list[set[int]]:
        """Compute backward-reachable states at each position.

        backward[i] = set of states from which an accepting state can be
        reached by processing positions i..n-1.
        backward[n] = accepting_states.
        """
        backward: list[set[int]] = [set() for _ in range(self._n + 1)]
        backward[self._n] = set(self._automaton.accepting_states)

        for i in range(self._n - 1, -1, -1):
            var = self._variables[i]
            if var in assignment:
                val = assignment[var]
                for state in range(self._automaton.num_states):
                    target = self._automaton.get_target(state, val)
                    if target is not None and target in backward[i + 1]:
                        backward[i].add(state)
            else:
                for state in range(self._automaton.num_states):
                    outgoing = self._automaton.get_outgoing_symbols(state)
                    for symbol, target in outgoing.items():
                        if var.domain.contains(symbol) and target in backward[i + 1]:
                            backward[i].add(state)
                            break

        return backward

    def get_reachable_states_at(
        self, position: int, assignment: dict[Variable, int]
    ) -> set[int]:
        """Get the set of reachable states at a given position.

        Useful for debugging and understanding propagation behavior.
        """
        forward = self._compute_forward_states(assignment)
        return forward[position]

    def get_accepting_path(
        self, assignment: dict[Variable, int]
    ) -> list[tuple[int, int, int]] | None:
        """Find a path through the DFA for the given complete assignment.

        Returns a list of (state, symbol, next_state) triples, or None
        if the assignment is not accepted.
        """
        path: list[tuple[int, int, int]] = []
        current = self._automaton.initial_state

        for var in self._variables:
            if var not in assignment:
                return None
            symbol = assignment[var]
            target = self._automaton.get_target(current, symbol)
            if target is None:
                return None
            path.append((current, symbol, target))
            current = target

        if current not in self._automaton.accepting_states:
            return None
        return path
