"""Load balancer for distributed CSP solving.

Distributes sub-problems across worker nodes, tracking their load and
health status. Assigns new work to the least-loaded worker and handles
worker failures by redistributing their pending work.

This module works with the splitter module to decompose problems and
the aggregator module to combine results from workers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence

from solveengine.distributed.splitter import SubProblem


class WorkerStatus(Enum):
    """Status of a worker node."""

    IDLE = auto()
    BUSY = auto()
    OVERLOADED = auto()
    FAILED = auto()
    DRAINING = auto()


@dataclass
class WorkerInfo:
    """Information about a worker node."""

    worker_id: str
    status: WorkerStatus = WorkerStatus.IDLE
    current_load: int = 0
    max_capacity: int = 4
    assigned_problems: list[int] = field(default_factory=list)
    completed_count: int = 0
    failed_count: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    total_solve_time: float = 0.0
    avg_solve_time: float = 0.0

    @property
    def load_ratio(self) -> float:
        """Current load as a fraction of capacity."""
        if self.max_capacity <= 0:
            return 1.0
        return self.current_load / self.max_capacity

    @property
    def is_available(self) -> bool:
        """Whether this worker can accept new work."""
        return (
            self.status in (WorkerStatus.IDLE, WorkerStatus.BUSY)
            and self.current_load < self.max_capacity
        )

    @property
    def time_since_heartbeat(self) -> float:
        """Seconds since last heartbeat."""
        return time.time() - self.last_heartbeat


@dataclass
class Assignment:
    """A work assignment mapping a sub-problem to a worker."""

    problem_id: int
    worker_id: str
    assigned_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    retries: int = 0
    max_retries: int = 3

    @property
    def is_pending(self) -> bool:
        return self.completed_at is None

    @property
    def elapsed_time(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def can_retry(self) -> bool:
        return self.retries < self.max_retries


class LoadBalancer:
    """Distributes sub-problems across worker nodes.

    Implements a least-loaded-first strategy with failure detection
    and automatic work redistribution.

    Args:
        heartbeat_timeout: Seconds before a worker is considered failed.
        overload_threshold: Load ratio above which a worker is overloaded.
        max_retries: Maximum retries for a failed assignment.
    """

    def __init__(
        self,
        heartbeat_timeout: float = 30.0,
        overload_threshold: float = 0.9,
        max_retries: int = 3,
    ) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        self._assignments: dict[int, Assignment] = {}
        self._pending_problems: list[SubProblem] = []
        self._completed_problems: list[int] = []
        self._heartbeat_timeout = heartbeat_timeout
        self._overload_threshold = overload_threshold
        self._max_retries = max_retries
        self._total_assigned: int = 0
        self._total_completed: int = 0
        self._total_failed: int = 0

    @property
    def num_workers(self) -> int:
        """Number of registered workers."""
        return len(self._workers)

    @property
    def active_workers(self) -> int:
        """Number of workers that are not failed."""
        return sum(
            1 for w in self._workers.values()
            if w.status != WorkerStatus.FAILED
        )

    @property
    def total_assigned(self) -> int:
        return self._total_assigned

    @property
    def total_completed(self) -> int:
        return self._total_completed

    @property
    def total_failed(self) -> int:
        return self._total_failed

    @property
    def pending_count(self) -> int:
        """Number of problems waiting to be assigned."""
        return len(self._pending_problems)

    def register_worker(
        self, worker_id: str, max_capacity: int = 4
    ) -> WorkerInfo:
        """Register a new worker node."""
        worker = WorkerInfo(
            worker_id=worker_id,
            max_capacity=max_capacity,
            last_heartbeat=time.time(),
        )
        self._workers[worker_id] = worker
        return worker

    def unregister_worker(self, worker_id: str) -> list[int]:
        """Remove a worker and return its pending problem IDs for reassignment."""
        worker = self._workers.get(worker_id)
        if worker is None:
            return []

        pending_ids = list(worker.assigned_problems)
        worker.status = WorkerStatus.FAILED
        del self._workers[worker_id]
        return pending_ids

    def record_heartbeat(self, worker_id: str, current_load: int | None = None) -> None:
        """Record a heartbeat from a worker."""
        worker = self._workers.get(worker_id)
        if worker is None:
            return

        worker.last_heartbeat = time.time()
        if current_load is not None:
            worker.current_load = current_load

        if worker.status == WorkerStatus.FAILED:
            worker.status = WorkerStatus.IDLE if worker.current_load == 0 else WorkerStatus.BUSY

        if worker.load_ratio >= self._overload_threshold:
            worker.status = WorkerStatus.OVERLOADED
        elif worker.current_load > 0:
            worker.status = WorkerStatus.BUSY
        else:
            worker.status = WorkerStatus.IDLE

    def submit_problems(self, problems: Sequence[SubProblem]) -> list[Assignment]:
        """Submit sub-problems for distribution to workers.

        Assigns problems to available workers immediately where possible,
        queues the rest for later assignment.
        """
        assignments: list[Assignment] = []

        for problem in problems:
            worker = self._select_worker()
            if worker is not None:
                assignment = self._assign_to_worker(problem, worker)
                assignments.append(assignment)
            else:
                self._pending_problems.append(problem)

        return assignments

    def assign_pending(self) -> list[Assignment]:
        """Try to assign pending problems to available workers.

        Called periodically to drain the pending queue as workers become free.
        """
        assignments: list[Assignment] = []
        remaining: list[SubProblem] = []

        for problem in self._pending_problems:
            worker = self._select_worker()
            if worker is not None:
                assignment = self._assign_to_worker(problem, worker)
                assignments.append(assignment)
            else:
                remaining.append(problem)

        self._pending_problems = remaining
        return assignments

    def report_completion(
        self, problem_id: int, worker_id: str, result: dict[str, Any] | None = None
    ) -> None:
        """Report that a worker completed a sub-problem."""
        assignment = self._assignments.get(problem_id)
        if assignment is not None:
            assignment.completed_at = time.time()
            assignment.result = result

        worker = self._workers.get(worker_id)
        if worker is not None:
            worker.current_load = max(0, worker.current_load - 1)
            if problem_id in worker.assigned_problems:
                worker.assigned_problems.remove(problem_id)
            worker.completed_count += 1
            elapsed = assignment.elapsed_time if assignment else 0.0
            worker.total_solve_time += elapsed
            total = worker.completed_count
            worker.avg_solve_time = worker.total_solve_time / total if total > 0 else 0.0

            if worker.current_load == 0:
                worker.status = WorkerStatus.IDLE
            else:
                worker.status = WorkerStatus.BUSY

        self._completed_problems.append(problem_id)
        self._total_completed += 1

    def report_failure(self, problem_id: int, worker_id: str) -> Assignment | None:
        """Report that a worker failed on a sub-problem.

        Returns a new assignment if the problem was retried, None otherwise.
        """
        assignment = self._assignments.get(problem_id)
        worker = self._workers.get(worker_id)

        if worker is not None:
            worker.current_load = max(0, worker.current_load - 1)
            if problem_id in worker.assigned_problems:
                worker.assigned_problems.remove(problem_id)
            worker.failed_count += 1

        self._total_failed += 1

        if assignment is not None and assignment.can_retry:
            assignment.retries += 1
            new_worker = self._select_worker(exclude=worker_id)
            if new_worker is not None:
                new_assignment = Assignment(
                    problem_id=problem_id,
                    worker_id=new_worker.worker_id,
                    retries=assignment.retries,
                    max_retries=assignment.max_retries,
                )
                new_worker.current_load += 1
                new_worker.assigned_problems.append(problem_id)
                self._assignments[problem_id] = new_assignment
                self._total_assigned += 1
                return new_assignment

        return None

    def check_health(self) -> list[str]:
        """Check worker health and detect failures.

        Returns list of worker IDs that have been marked as failed.
        """
        failed_workers: list[str] = []

        for worker_id, worker in self._workers.items():
            if worker.status == WorkerStatus.FAILED:
                continue
            if worker.time_since_heartbeat > self._heartbeat_timeout:
                worker.status = WorkerStatus.FAILED
                failed_workers.append(worker_id)

        return failed_workers

    def redistribute_failed_work(self) -> list[Assignment]:
        """Redistribute work from failed workers to healthy ones."""
        new_assignments: list[Assignment] = []

        for worker_id, worker in list(self._workers.items()):
            if worker.status != WorkerStatus.FAILED:
                continue
            for problem_id in list(worker.assigned_problems):
                assignment = self._assignments.get(problem_id)
                if assignment is None or not assignment.is_pending:
                    continue
                new_worker = self._select_worker(exclude=worker_id)
                if new_worker is not None:
                    new_assign = Assignment(
                        problem_id=problem_id,
                        worker_id=new_worker.worker_id,
                        retries=(assignment.retries + 1) if assignment else 0,
                    )
                    new_worker.current_load += 1
                    new_worker.assigned_problems.append(problem_id)
                    self._assignments[problem_id] = new_assign
                    new_assignments.append(new_assign)
            worker.assigned_problems.clear()

        return new_assignments

    def _select_worker(self, exclude: str | None = None) -> WorkerInfo | None:
        """Select the least-loaded available worker."""
        candidates = [
            w for w in self._workers.values()
            if w.is_available and w.worker_id != exclude
        ]
        if not candidates:
            return None

        return min(candidates, key=lambda w: (w.load_ratio, w.avg_solve_time))

    def _assign_to_worker(self, problem: SubProblem, worker: WorkerInfo) -> Assignment:
        """Create an assignment of a problem to a worker."""
        assignment = Assignment(
            problem_id=problem.problem_id,
            worker_id=worker.worker_id,
            max_retries=self._max_retries,
        )
        worker.current_load += 1
        worker.assigned_problems.append(problem.problem_id)
        if worker.load_ratio >= self._overload_threshold:
            worker.status = WorkerStatus.OVERLOADED
        else:
            worker.status = WorkerStatus.BUSY

        self._assignments[problem.problem_id] = assignment
        self._total_assigned += 1
        return assignment

    def get_statistics(self) -> dict[str, Any]:
        """Return load balancer statistics."""
        worker_loads = [w.load_ratio for w in self._workers.values()]
        avg_load = sum(worker_loads) / len(worker_loads) if worker_loads else 0.0

        return {
            "num_workers": self.num_workers,
            "active_workers": self.active_workers,
            "total_assigned": self._total_assigned,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "pending_count": self.pending_count,
            "avg_worker_load": avg_load,
        }
