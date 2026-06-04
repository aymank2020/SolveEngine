"""Benchmark script for SolveEngine performance testing."""

import time
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.heuristics.variable_ordering import DomWdegSelector, MRVSelector
from solveengine.heuristics.value_ordering import LCVOrderer


def bench_nqueens(n: int, heuristic: str = "dom_wdeg") -> dict:
    """Benchmark N-Queens solving."""
    queens = [Variable(f"q{i}", range(0, n)) for i in range(n)]
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = j - i
            constraints.append(BinaryConstraint(
                queens[i], queens[j], lambda a, b: a != b
            ))
            constraints.append(BinaryConstraint(
                queens[i], queens[j], lambda a, b, d=diff: abs(a - b) != d
            ))

    selector = DomWdegSelector() if heuristic == "dom_wdeg" else MRVSelector()
    solver = BacktrackSolver(
        queens, constraints,
        var_selector=selector,
        val_orderer=LCVOrderer(),
        use_forward_check=True,
    )

    start = time.perf_counter()
    result = solver.solve()
    elapsed = time.perf_counter() - start

    return {
        "problem": f"{n}-Queens",
        "heuristic": heuristic,
        "solved": result is not None,
        "time_ms": elapsed * 1000,
        "nodes": solver.stats.nodes_explored,
        "backtracks": solver.stats.backtracks,
    }


def bench_alldiff(n: int) -> dict:
    """Benchmark AllDifferent with n variables, domain [1, n]."""
    variables = [Variable(f"v{i}", range(1, n + 1)) for i in range(n)]
    constraints = [AllDifferent(*variables)]

    solver = BacktrackSolver(
        variables, constraints,
        var_selector=MRVSelector(),
        use_forward_check=True,
    )

    start = time.perf_counter()
    result = solver.solve()
    elapsed = time.perf_counter() - start

    return {
        "problem": f"AllDiff-{n}",
        "solved": result is not None,
        "time_ms": elapsed * 1000,
        "nodes": solver.stats.nodes_explored,
    }


if __name__ == "__main__":
    print("SolveEngine Benchmarks")
    print("=" * 60)

    for n in [8, 12, 16, 20]:
        result = bench_nqueens(n)
        print(f"  {result['problem']:12s} | {result['time_ms']:8.1f}ms | "
              f"nodes={result['nodes']:6d} | bt={result['backtracks']}")

    print()
    for n in [8, 10, 12, 15]:
        result = bench_alldiff(n)
        print(f"  {result['problem']:12s} | {result['time_ms']:8.1f}ms | "
              f"nodes={result['nodes']:6d}")
