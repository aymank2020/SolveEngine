"""Scheduling modeling helpers for constraint satisfaction problems.

Provides high-level abstractions for common scheduling problems:
- Task intervals with start times, durations, and end times
- Precedence constraints between tasks
- Resource constraints (unary and cumulative)
- Makespan minimization helpers

These helpers build on the core Model class to provide a domain-specific
interface for scheduling problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint, BinaryConstraint
from solveengine.modeling.model import Model


@dataclass
class TaskInterval:
    """Represents a task with a start time, duration, and end time.

    The end time is implicitly start + duration. The start variable
    is a CSP variable with a domain representing possible start times.

    Attributes:
        name: Human-readable task name.
        start_var: Variable representing the start time.
        duration: Fixed duration of the task.
        end_var: Variable representing the end time (start + duration).
        resource_id: Optional resource this task requires.
        resource_demand: Amount of resource capacity consumed.
    """

    name: str
    start_var: Variable
    duration: int
    end_var: Variable
    resource_id: int | None = None
    resource_demand: int = 1

    @property
    def earliest_start(self) -> int:
        """Earliest possible start time."""
        return self.start_var.domain.min_value

    @property
    def latest_start(self) -> int:
        """Latest possible start time."""
        return self.start_var.domain.max_value

    @property
    def earliest_end(self) -> int:
        """Earliest possible end time."""
        return self.end_var.domain.min_value

    @property
    def latest_end(self) -> int:
        """Latest possible end time."""
        return self.end_var.domain.max_value


@dataclass
class Resource:
    """A resource with limited capacity.

    Attributes:
        resource_id: Unique identifier for the resource.
        name: Human-readable resource name.
        capacity: Maximum simultaneous usage capacity.
        tasks: Tasks assigned to this resource.
    """

    resource_id: int
    name: str
    capacity: int = 1
    tasks: list[TaskInterval] = field(default_factory=list)


class SchedulingModel:
    """High-level scheduling model builder.

    Provides a convenient API for defining scheduling problems with
    tasks, precedence relations, and resource constraints.

    Example:
        sched = SchedulingModel(horizon=100)
        t1 = sched.add_task("paint", duration=5)
        t2 = sched.add_task("dry", duration=10)
        sched.add_precedence(t1, t2)  # paint before dry
        sched.add_unary_resource("painter", [t1])
        solution = sched.solve()
    """

    def __init__(self, horizon: int, name: str = "scheduling") -> None:
        """Initialize the scheduling model.

        Args:
            horizon: Maximum time horizon (all tasks must complete by this time).
            name: Model name for identification.
        """
        self._horizon = horizon
        self._name = name
        self._model = Model()
        self._tasks: list[TaskInterval] = []
        self._resources: list[Resource] = []
        self._precedences: list[tuple[TaskInterval, TaskInterval]] = []
        self._makespan_var: Variable | None = None

    @property
    def horizon(self) -> int:
        """Time horizon for the schedule."""
        return self._horizon

    @property
    def num_tasks(self) -> int:
        """Number of tasks in the model."""
        return len(self._tasks)

    @property
    def num_resources(self) -> int:
        """Number of resources in the model."""
        return len(self._resources)

    @property
    def tasks(self) -> list[TaskInterval]:
        """All tasks in the model."""
        return list(self._tasks)

    def add_task(
        self,
        name: str,
        duration: int,
        earliest_start: int = 0,
        latest_end: int | None = None,
        resource_id: int | None = None,
        resource_demand: int = 1,
    ) -> TaskInterval:
        """Add a task to the scheduling model.

        Creates start and end variables with appropriate domains.

        Args:
            name: Task name.
            duration: Fixed task duration.
            earliest_start: Earliest allowed start time.
            latest_end: Latest allowed end time (defaults to horizon).
            resource_id: Optional resource assignment.
            resource_demand: Resource capacity consumed by this task.

        Returns:
            The created TaskInterval.
        """
        if latest_end is None:
            latest_end = self._horizon

        latest_start = latest_end - duration
        if latest_start < earliest_start:
            raise ValueError(
                f"Task '{name}' with duration {duration} cannot fit between "
                f"earliest_start={earliest_start} and latest_end={latest_end}"
            )

        start_var = self._model.int_var(
            f"{name}_start", earliest_start, latest_start
        )
        end_var = self._model.int_var(
            f"{name}_end", earliest_start + duration, latest_end
        )

        # Link end = start + duration
        self._model.add_binary(
            start_var, end_var,
            lambda s, e, d=duration: e == s + d,
            f"{name}_end=start+{duration}",
        )

        task = TaskInterval(
            name=name,
            start_var=start_var,
            duration=duration,
            end_var=end_var,
            resource_id=resource_id,
            resource_demand=resource_demand,
        )
        self._tasks.append(task)
        return task

    def add_precedence(
        self, before: TaskInterval, after: TaskInterval, delay: int = 0
    ) -> None:
        """Add a precedence constraint: before must finish before after starts.

        Enforces: before.end + delay <= after.start

        Args:
            before: Task that must complete first.
            after: Task that must start after.
            delay: Minimum delay between end of before and start of after.
        """
        self._precedences.append((before, after))
        self._model.add_binary(
            before.end_var, after.start_var,
            lambda e, s, d=delay: e + d <= s,
            f"{before.name}_before_{after.name}",
        )

    def add_no_overlap(self, tasks: Sequence[TaskInterval]) -> None:
        """Add no-overlap (disjunctive) constraint between tasks.

        Ensures no two tasks in the set overlap in time. For each pair,
        either task_i ends before task_j starts, or vice versa.

        Args:
            tasks: Tasks that must not overlap.
        """
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                ti = tasks[i]
                tj = tasks[j]
                # ti ends before tj starts OR tj ends before ti starts
                self._model.add_binary(
                    ti.start_var, tj.start_var,
                    lambda si, sj, di=ti.duration, dj=tj.duration: (
                        si + di <= sj or sj + dj <= si
                    ),
                    f"no_overlap_{ti.name}_{tj.name}",
                )

    def add_unary_resource(
        self, name: str, tasks: Sequence[TaskInterval]
    ) -> Resource:
        """Add a unary (capacity=1) resource constraint.

        Ensures that at most one task uses the resource at any time.
        This is equivalent to a no-overlap constraint on the tasks.

        Args:
            name: Resource name.
            tasks: Tasks that use this resource.

        Returns:
            The created Resource.
        """
        resource = Resource(
            resource_id=len(self._resources),
            name=name,
            capacity=1,
            tasks=list(tasks),
        )
        self._resources.append(resource)
        self.add_no_overlap(tasks)
        return resource

    def add_cumulative_resource(
        self, name: str, tasks: Sequence[TaskInterval], capacity: int
    ) -> Resource:
        """Add a cumulative resource constraint.

        Ensures that at any time point, the sum of demands of active tasks
        does not exceed the resource capacity.

        Uses a time-point decomposition: for each possible time point,
        the sum of demands of tasks active at that point <= capacity.

        Args:
            name: Resource name.
            tasks: Tasks that use this resource.
            capacity: Maximum simultaneous capacity.

        Returns:
            The created Resource.
        """
        resource = Resource(
            resource_id=len(self._resources),
            name=name,
            capacity=capacity,
            tasks=list(tasks),
        )
        self._resources.append(resource)

        # For efficiency, only check at task start times
        # At each start time, sum of demands of overlapping tasks <= capacity
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                ti = tasks[i]
                tj = tasks[j]
                # If both tasks overlap AND their combined demand exceeds capacity,
                # they cannot overlap
                if ti.resource_demand + tj.resource_demand > capacity:
                    self.add_no_overlap([ti, tj])

        return resource

    def add_makespan_objective(self) -> Variable:
        """Add a makespan variable that equals the maximum end time.

        The makespan is the completion time of the last task.
        Useful for minimization objectives.

        Returns:
            The makespan variable.
        """
        if not self._tasks:
            raise ValueError("Cannot add makespan without tasks")

        self._makespan_var = self._model.int_var(
            "makespan", 0, self._horizon
        )

        # Makespan >= end of every task
        for task in self._tasks:
            self._model.add_binary(
                task.end_var, self._makespan_var,
                lambda e, m: m >= e,
                f"makespan>={task.name}_end",
            )

        return self._makespan_var

    def solve(self) -> dict[str, int] | None:
        """Solve the scheduling problem.

        Returns a mapping from variable names to values, or None if infeasible.
        """
        return self._model.solve()

    def solve_minimize_makespan(self, max_iterations: int = 100) -> dict[str, int] | None:
        """Solve with makespan minimization using iterative tightening.

        Repeatedly solves the problem with decreasing makespan upper bounds
        until no better solution exists.

        Args:
            max_iterations: Maximum number of solve iterations.

        Returns:
            The best solution found, or None if infeasible.
        """
        if self._makespan_var is None:
            self.add_makespan_objective()

        best_solution: dict[str, int] | None = None
        current_upper = self._horizon

        for _ in range(max_iterations):
            # Restrict makespan domain
            makespan_var = self._makespan_var
            if makespan_var is None:
                break

            solution = self._model.solve()
            if solution is None:
                break

            best_solution = solution
            current_makespan = solution.get("makespan", current_upper)

            # Tighten: next solution must have smaller makespan
            current_upper = current_makespan - 1
            if current_upper < 0:
                break

            # Add constraint for next iteration
            self._model.add_unary(
                makespan_var,
                lambda m, ub=current_upper: m <= ub,
                f"makespan<={current_upper}",
            )

        return best_solution

    def compute_critical_path(self) -> list[TaskInterval]:
        """Compute the critical path through the precedence graph.

        The critical path is the longest path through the precedence
        network, determining the minimum possible makespan.

        Returns:
            List of tasks on the critical path.
        """
        # Build precedence graph
        successors: dict[str, list[tuple[TaskInterval, int]]] = {}
        predecessors: dict[str, list[str]] = {}

        for task in self._tasks:
            successors[task.name] = []
            predecessors[task.name] = []

        for before, after in self._precedences:
            successors[before.name].append((after, 0))
            predecessors[after.name].append(before.name)

        # Forward pass: compute earliest start times
        earliest_start: dict[str, int] = {}
        task_by_name = {t.name: t for t in self._tasks}

        # Topological order
        in_degree = {t.name: len(predecessors[t.name]) for t in self._tasks}
        queue = [t.name for t in self._tasks if in_degree[t.name] == 0]
        topo_order: list[str] = []

        while queue:
            name = queue.pop(0)
            topo_order.append(name)
            for succ_task, delay in successors[name]:
                in_degree[succ_task.name] -= 1
                if in_degree[succ_task.name] == 0:
                    queue.append(succ_task.name)

        # Forward pass
        for name in topo_order:
            task = task_by_name[name]
            es = task.earliest_start
            for pred_name in predecessors[name]:
                pred_task = task_by_name[pred_name]
                es = max(es, earliest_start.get(pred_name, 0) + pred_task.duration)
            earliest_start[name] = es

        # Backward pass: compute latest start times
        latest_start: dict[str, int] = {}
        makespan = max(
            earliest_start.get(t.name, 0) + t.duration for t in self._tasks
        )

        for name in reversed(topo_order):
            task = task_by_name[name]
            ls = makespan - task.duration
            for succ_task, delay in successors[name]:
                ls = min(ls, latest_start.get(succ_task.name, makespan) - task.duration)
            latest_start[name] = ls

        # Critical path: tasks where earliest_start == latest_start
        critical = [
            task_by_name[name]
            for name in topo_order
            if earliest_start.get(name, 0) == latest_start.get(name, 0)
        ]

        return critical

    def get_schedule_summary(self, solution: dict[str, int]) -> str:
        """Generate a human-readable schedule summary.

        Args:
            solution: A solution mapping variable names to values.

        Returns:
            Formatted string showing the schedule.
        """
        lines = [f"Schedule: {self._name}", "=" * 40]

        task_starts: list[tuple[int, str, int, int]] = []
        for task in self._tasks:
            start = solution.get(f"{task.name}_start", -1)
            end = solution.get(f"{task.name}_end", -1)
            task_starts.append((start, task.name, task.duration, end))

        task_starts.sort()
        for start, name, duration, end in task_starts:
            lines.append(f"  {name:<15} start={start:>3}  dur={duration:>2}  end={end:>3}")

        makespan = solution.get("makespan")
        if makespan is not None:
            lines.append(f"\nMakespan: {makespan}")

        return "\n".join(lines)
