"""Job Shop Scheduling example using SolveEngine.

Solves a job shop scheduling problem with 3 machines and 5 jobs.
Each job consists of a sequence of operations, each requiring a specific
machine for a specific duration. The goal is to minimize the makespan
(total completion time).

Problem:
- 3 machines: M0, M1, M2
- 5 jobs, each with 3 operations (one per machine)
- Each operation has a fixed duration and machine assignment
- Operations within a job must be sequential
- Each machine can process only one operation at a time
- Objective: minimize makespan (time when all jobs complete)
"""

from __future__ import annotations

from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint, Constraint
from solveengine.global_cstr.cumulative import CumulativeConstraint
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.heuristics.variable_ordering import DomWdegSelector
from solveengine.heuristics.value_ordering import AscendingOrderer


# Problem data: jobs[job_id] = [(machine, duration), ...]
JOBS: list[list[tuple[int, int]]] = [
    [(0, 3), (1, 2), (2, 4)],   # Job 0: M0 for 3, M1 for 2, M2 for 4
    [(1, 4), (0, 3), (2, 2)],   # Job 1: M1 for 4, M0 for 3, M2 for 2
    [(2, 2), (1, 3), (0, 4)],   # Job 2: M2 for 2, M1 for 3, M0 for 4
    [(0, 2), (2, 3), (1, 4)],   # Job 3: M0 for 2, M2 for 3, M1 for 4
    [(1, 3), (2, 2), (0, 3)],   # Job 4: M1 for 3, M2 for 2, M0 for 3
]

NUM_MACHINES = 3
NUM_JOBS = 5


def compute_horizon(jobs: list[list[tuple[int, int]]]) -> int:
    """Compute an upper bound on the makespan (sum of all durations)."""
    total = 0
    for job in jobs:
        for _, duration in job:
            total += duration
    return total


def create_scheduling_model(
    jobs: list[list[tuple[int, int]]], horizon: int
) -> tuple[list[Variable], list[Constraint], list[Variable]]:
    """Create the CSP model for the job shop problem.

    Returns:
        Tuple of (all_start_vars, all_constraints, end_vars_per_job)
    """
    all_vars: list[Variable] = []
    all_constraints: list[Constraint] = []
    job_start_vars: list[list[Variable]] = []
    machine_operations: dict[int, list[tuple[Variable, int]]] = {
        m: [] for m in range(NUM_MACHINES)
    }

    # Create start time variables for each operation
    for job_id, job in enumerate(jobs):
        job_vars: list[Variable] = []
        for op_idx, (machine, duration) in enumerate(job):
            var_name = f"j{job_id}_op{op_idx}_m{machine}"
            # Start time can be 0 to horizon - duration
            start_var = Variable(var_name, range(0, horizon - duration + 1))
            job_vars.append(start_var)
            all_vars.append(start_var)
            machine_operations[machine].append((start_var, duration))
        job_start_vars.append(job_vars)

    # Precedence constraints: operations within a job must be sequential
    for job_id, job in enumerate(jobs):
        for op_idx in range(len(job) - 1):
            current_var = job_start_vars[job_id][op_idx]
            next_var = job_start_vars[job_id][op_idx + 1]
            current_duration = job[op_idx][1]

            # next_start >= current_start + current_duration
            precedence = BinaryConstraint(
                current_var, next_var,
                lambda s1, s2, d=current_duration: s2 >= s1 + d,
                name=f"prec_j{job_id}_op{op_idx}",
            )
            all_constraints.append(precedence)

    # Machine disjunctive constraints: no two operations on same machine overlap
    for machine_id, ops in machine_operations.items():
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                var_i, dur_i = ops[i]
                var_j, dur_j = ops[j]

                # Either op_i finishes before op_j starts, or vice versa
                disjunctive = BinaryConstraint(
                    var_i, var_j,
                    lambda si, sj, di=dur_i, dj=dur_j: (
                        si + di <= sj or sj + dj <= si
                    ),
                    name=f"disj_m{machine_id}_{var_i.name}_{var_j.name}",
                )
                all_constraints.append(disjunctive)

    # Compute end time variables for makespan
    end_vars: list[Variable] = []
    for job_id, job in enumerate(jobs):
        last_op_idx = len(job) - 1
        last_duration = job[last_op_idx][1]
        last_start = job_start_vars[job_id][last_op_idx]
        # End time = start + duration (we'll compute this from the solution)
        end_vars.append(last_start)

    return all_vars, all_constraints, end_vars


def solve_with_makespan_bound(
    jobs: list[list[tuple[int, int]]], makespan_bound: int
) -> dict[Variable, int] | None:
    """Try to solve the scheduling problem with a given makespan bound.

    Adds constraints that all operations must finish by makespan_bound.
    """
    all_vars, all_constraints, end_vars = create_scheduling_model(jobs, makespan_bound)

    # Add makespan constraints: each job must finish by the bound
    for job_id, job in enumerate(jobs):
        last_duration = job[-1][1]
        end_var = end_vars[job_id]
        # end_var + last_duration <= makespan_bound
        # So end_var <= makespan_bound - last_duration
        # This is already enforced by the domain upper bound if horizon = makespan_bound

    solver = BacktrackSolver(
        all_vars,
        all_constraints,
        var_selector=DomWdegSelector(),
        val_orderer=AscendingOrderer(),
        use_forward_check=True,
    )
    solver.set_node_limit(50000)

    return solver.solve()


def minimize_makespan(jobs: list[list[tuple[int, int]]]) -> tuple[int, dict[Variable, int]] | None:
    """Find the minimum makespan using binary search on the bound.

    Returns (makespan, solution) or None if no solution found.
    """
    # Lower bound: longest single job
    lower = max(sum(d for _, d in job) for job in jobs)
    # Upper bound: sum of all durations
    upper = compute_horizon(jobs)

    best_solution: dict[Variable, int] | None = None
    best_makespan = upper

    # Binary search on makespan
    while lower <= upper:
        mid = (lower + upper) // 2
        solution = solve_with_makespan_bound(jobs, mid)

        if solution is not None:
            best_solution = solution
            best_makespan = mid
            upper = mid - 1
        else:
            lower = mid + 1

    if best_solution is not None:
        return (best_makespan, best_solution)
    return None


def compute_makespan(solution: dict[Variable, int], jobs: list[list[tuple[int, int]]]) -> int:
    """Compute the actual makespan from a solution."""
    max_end = 0
    for var, start_time in solution.items():
        # Parse job and operation from variable name
        name = var.name
        parts = name.split("_")
        job_id = int(parts[0][1:])
        op_idx = int(parts[1][2:])
        duration = jobs[job_id][op_idx][1]
        end_time = start_time + duration
        if end_time > max_end:
            max_end = end_time
    return max_end


def print_schedule(solution: dict[Variable, int], jobs: list[list[tuple[int, int]]]) -> None:
    """Print a formatted schedule."""
    print("\nSchedule:")
    print("-" * 60)

    for job_id, job in enumerate(jobs):
        print(f"  Job {job_id}:")
        for op_idx, (machine, duration) in enumerate(job):
            var_name = f"j{job_id}_op{op_idx}_m{machine}"
            start = None
            for var, val in solution.items():
                if var.name == var_name:
                    start = val
                    break
            if start is not None:
                end = start + duration
                print(f"    Op{op_idx} on M{machine}: [{start:2d}, {end:2d}) (dur={duration})")

    print("-" * 60)


def print_gantt_text(solution: dict[Variable, int], jobs: list[list[tuple[int, int]]], makespan: int) -> None:
    """Print a text-based Gantt chart."""
    print("\nGantt Chart (by machine):")
    print("-" * 60)

    machine_schedule: dict[int, list[tuple[int, int, int, int]]] = {
        m: [] for m in range(NUM_MACHINES)
    }

    for var, start_time in solution.items():
        name = var.name
        parts = name.split("_")
        job_id = int(parts[0][1:])
        op_idx = int(parts[1][2:])
        machine = int(parts[2][1:])
        duration = jobs[job_id][op_idx][1]
        machine_schedule[machine].append((start_time, duration, job_id, op_idx))

    for machine_id in range(NUM_MACHINES):
        ops = sorted(machine_schedule[machine_id])
        timeline = ["."] * makespan
        for start, dur, job_id, _ in ops:
            for t in range(start, min(start + dur, makespan)):
                timeline[t] = str(job_id)

        chart = "".join(timeline)
        print(f"  M{machine_id}: |{chart}|")

    print(f"       {''.join(str(t % 10) for t in range(makespan))}")
    print("-" * 60)


def main() -> None:
    """Run the job shop scheduling example."""
    print("=" * 60)
    print("Job Shop Scheduling Problem")
    print(f"  {NUM_JOBS} jobs, {NUM_MACHINES} machines")
    print("=" * 60)

    print("\nProblem definition:")
    for job_id, job in enumerate(JOBS):
        ops_str = " → ".join(f"M{m}({d})" for m, d in job)
        print(f"  Job {job_id}: {ops_str}")

    horizon = compute_horizon(JOBS)
    print(f"\nHorizon (upper bound): {horizon}")
    print(f"Lower bound (longest job): {max(sum(d for _, d in job) for job in JOBS)}")

    print("\nSolving with binary search on makespan...")
    result = minimize_makespan(JOBS)

    if result is not None:
        makespan, solution = result
        print(f"\nOptimal makespan: {makespan}")
        actual = compute_makespan(solution, JOBS)
        print(f"Verified makespan: {actual}")
        print_schedule(solution, JOBS)
        print_gantt_text(solution, JOBS, actual)
    else:
        print("\nNo solution found within node limit.")
        print("Trying with relaxed makespan...")
        solution = solve_with_makespan_bound(JOBS, horizon)
        if solution is not None:
            makespan = compute_makespan(solution, JOBS)
            print(f"Found feasible schedule with makespan: {makespan}")
            print_schedule(solution, JOBS)
            print_gantt_text(solution, JOBS, makespan)
        else:
            print("Could not find any feasible schedule.")


if __name__ == "__main__":
    main()
