"""CLI runner for solving CSP problems from JSON files."""

from __future__ import annotations

import argparse
import json
import sys
import time

from solveengine.persistence.serializer import ProblemSerializer
from solveengine.solver.backtrack import BacktrackSolver
from solveengine.heuristics.variable_ordering import MRVSelector, DomWdegSelector
from solveengine.heuristics.value_ordering import LCVOrderer, AscendingOrderer


def run_cli(args: list[str] | None = None) -> int:
    """Run the CLI solver."""
    parser = argparse.ArgumentParser(
        prog="solveengine",
        description="Solve constraint satisfaction problems from JSON files",
    )
    parser.add_argument("input", help="Path to JSON problem file")
    parser.add_argument("--all", action="store_true", help="Find all solutions")
    parser.add_argument(
        "--heuristic",
        choices=["mrv", "dom_wdeg"],
        default="dom_wdeg",
        help="Variable ordering heuristic",
    )
    parser.add_argument(
        "--value-order",
        choices=["ascending", "lcv"],
        default="lcv",
        help="Value ordering strategy",
    )
    parser.add_argument("--limit", type=int, default=0, help="Node limit (0=unlimited)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parsed = parser.parse_args(args)

    try:
        with open(parsed.input, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        return 1

    serializer = ProblemSerializer()
    try:
        variables = serializer.deserialize_variables(data)
        constraints = serializer.deserialize_constraints(data, variables)
    except (KeyError, TypeError, ValueError) as e:
        print(f"Invalid problem: {e}", file=sys.stderr)
        return 1

    if parsed.verbose:
        print(f"Problem: {len(variables)} variables, {len(constraints)} constraints")

    var_selector = DomWdegSelector() if parsed.heuristic == "dom_wdeg" else MRVSelector()
    val_orderer = LCVOrderer() if parsed.value_order == "lcv" else AscendingOrderer()

    solver = BacktrackSolver(
        variables,
        constraints,
        var_selector=var_selector,
        val_orderer=val_orderer,
        use_forward_check=True,
    )

    if parsed.limit > 0:
        solver.set_node_limit(parsed.limit)

    start = time.perf_counter()

    if parsed.all:
        solutions = solver.solve_all()
        elapsed = time.perf_counter() - start
        suffix = " (incomplete: node limit reached)" if solver.node_limit_reached else ""
        print(f"Found {len(solutions)} solution(s) in {elapsed:.3f}s{suffix}")
        for i, sol in enumerate(solutions, 1):
            assignment = {var.name: val for var, val in sol.items()}
            print(f"  Solution {i}: {assignment}")
    else:
        result = solver.solve()
        elapsed = time.perf_counter() - start
        if result is None:
            if solver.node_limit_reached:
                print(f"UNKNOWN: node limit reached ({elapsed:.3f}s)")
                return 2
            print(f"UNSATISFIABLE ({elapsed:.3f}s)")
            return 1
        assignment = {var.name: val for var, val in result.items()}
        print(f"SATISFIABLE ({elapsed:.3f}s)")
        print(f"  {assignment}")

    if parsed.verbose:
        stats = solver.stats
        print(f"\nStats: nodes={stats.nodes_explored}, backtracks={stats.backtracks}")

    return 2 if solver.node_limit_reached else 0


if __name__ == "__main__":
    sys.exit(run_cli())
