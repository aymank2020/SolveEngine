"""Restart strategies for CSP solvers.

Restarts abandon the current search tree and start fresh with a different
variable/value ordering. This helps escape heavy-tailed runtime distributions
that are common in CSP solving.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import log2, floor


class RestartPolicy(ABC):
    """Abstract base for restart policies."""

    def __init__(self) -> None:
        self._restart_count: int = 0
        self._total_nodes: int = 0

    @property
    def restart_count(self) -> int:
        return self._restart_count

    @property
    def total_nodes(self) -> int:
        return self._total_nodes

    @abstractmethod
    def should_restart(self, nodes_since_restart: int) -> bool:
        """Check if the solver should restart now."""
        ...

    @abstractmethod
    def next_cutoff(self) -> int:
        """Return the node budget for the next restart."""
        ...

    def record_restart(self, nodes: int) -> None:
        """Record that a restart occurred after exploring 'nodes' nodes."""
        self._restart_count += 1
        self._total_nodes += nodes

    def reset(self) -> None:
        """Reset the policy state."""
        self._restart_count = 0
        self._total_nodes = 0


class GeometricRestart(RestartPolicy):
    """Geometric restart policy.

    The cutoff grows geometrically: base, base*factor, base*factor^2, ...
    Common choice: base=100, factor=1.5
    """

    def __init__(self, base: int = 100, factor: float = 1.5) -> None:
        super().__init__()
        self._base = base
        self._factor = factor

    @property
    def base(self) -> int:
        return self._base

    @property
    def factor(self) -> float:
        return self._factor

    def should_restart(self, nodes_since_restart: int) -> bool:
        return nodes_since_restart >= self.next_cutoff()

    def next_cutoff(self) -> int:
        return int(self._base * (self._factor ** self._restart_count))

    def __repr__(self) -> str:
        return f"GeometricRestart(base={self._base}, factor={self._factor})"


class LubyRestart(RestartPolicy):
    """Luby sequence restart policy.

    Uses the Luby sequence (1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...)
    multiplied by a base unit. Proven optimal for Las Vegas algorithms
    when the runtime distribution is unknown.
    """

    def __init__(self, unit: int = 100) -> None:
        super().__init__()
        self._unit = unit

    @property
    def unit(self) -> int:
        return self._unit

    def should_restart(self, nodes_since_restart: int) -> bool:
        return nodes_since_restart >= self.next_cutoff()

    def next_cutoff(self) -> int:
        return self._unit * self._luby(self._restart_count + 1)

    @staticmethod
    def _luby(i: int) -> int:
        """Compute the i-th element of the Luby sequence (1-indexed).

        The Luby sequence is defined as:
        - luby(1) = 1
        - If i = 2^k - 1 for some k, then luby(i) = 2^(k-1)
        - Otherwise, luby(i) = luby(i - 2^(k-1) + 1) where k = floor(log2(i))
        """
        if i <= 0:
            return 1

        # Find k such that 2^k - 1 >= i
        k = 1
        while (1 << k) - 1 < i:
            k += 1

        if i == (1 << k) - 1:
            return 1 << (k - 1)

        return LubyRestart._luby(i - (1 << (k - 1)) + 1)

    def __repr__(self) -> str:
        return f"LubyRestart(unit={self._unit})"


class FixedRestart(RestartPolicy):
    """Fixed cutoff restart policy.

    Restarts after a fixed number of nodes every time.
    Simple but effective for problems with known difficulty.
    """

    def __init__(self, cutoff: int = 500) -> None:
        super().__init__()
        self._cutoff = cutoff

    def should_restart(self, nodes_since_restart: int) -> bool:
        return nodes_since_restart >= self._cutoff

    def next_cutoff(self) -> int:
        return self._cutoff


class NoRestart(RestartPolicy):
    """No-restart policy (run to completion)."""

    def should_restart(self, nodes_since_restart: int) -> bool:
        return False

    def next_cutoff(self) -> int:
        return 2**63  # Effectively infinite


class NestedRestart(RestartPolicy):
    """Nested restart policy combining inner and outer restart sequences.

    The inner policy controls short restarts within a phase.
    The outer policy controls when to reset the inner policy and
    increase the overall budget.
    """

    def __init__(
        self,
        inner: RestartPolicy | None = None,
        outer_factor: float = 2.0,
        outer_base: int = 1000,
    ) -> None:
        super().__init__()
        self._inner = inner or LubyRestart(unit=50)
        self._outer_factor = outer_factor
        self._outer_base = outer_base
        self._phase: int = 0
        self._nodes_in_phase: int = 0

    def should_restart(self, nodes_since_restart: int) -> bool:
        if self._inner.should_restart(nodes_since_restart):
            return True
        # Check if outer phase budget is exhausted
        phase_budget = int(self._outer_base * (self._outer_factor ** self._phase))
        return self._nodes_in_phase >= phase_budget

    def next_cutoff(self) -> int:
        return self._inner.next_cutoff()

    def record_restart(self, nodes: int) -> None:
        super().record_restart(nodes)
        self._nodes_in_phase += nodes
        self._inner.record_restart(nodes)

        # Check if outer phase is complete
        phase_budget = int(self._outer_base * (self._outer_factor ** self._phase))
        if self._nodes_in_phase >= phase_budget:
            self._phase += 1
            self._nodes_in_phase = 0
            self._inner.reset()
