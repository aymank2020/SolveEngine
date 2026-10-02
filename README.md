# SolveEngine

A constraint satisfaction problem (CSP) solver featuring arc consistency propagation,
intelligent backtracking with conflict-driven backjumping, and nogood learning.

## Overview

SolveEngine provides a modular framework for solving constraint satisfaction problems:

- **Variables** with finite domains
- **Constraints** (binary, unary, and global)
- **Propagation** via arc consistency (AC-3, AC-4) and node consistency
- **Search** with backtracking, forward checking, and backjumping
- **Heuristics** for variable ordering (MRV, degree) and value ordering (LCV)
- **Learning** from conflicts via nogood recording and clause-driven backjumping
- **Global constraints** (AllDifferent, Sum, Element, Cardinality)

## Architecture

```
src/solveengine/
├── core/           # Variable, Domain, Constraint primitives
├── propagation/    # Arc/node consistency enforcement
├── heuristics/     # Variable and value ordering strategies
├── solver/         # Backtracking engine with forward checking
├── decomposition/  # Constraint graph analysis, tree decomposition
├── global_cstr/    # Global constraint implementations
├── learning/       # Nogood learning, conflict analysis
├── search/         # Search strategies (DFS, LDS, iterative)
├── modeling/       # High-level modeling API
├── analysis/       # Graph metrics, bottleneck detection
├── transforms/     # Constraint reformulation, symmetry breaking
├── persistence/    # State checkpointing and restore
├── debug/          # Search tree tracing, failure explanation
└── cli/            # Command-line interface
```

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from solveengine import Variable, BacktrackSolver, BinaryConstraint, AllDifferent

x = Variable("x", range(1, 10))
y = Variable("y", range(1, 10))
z = Variable("z", range(1, 10))
constraints = [
    AllDifferent(x, y, z),
    BinaryConstraint(x, y, lambda a, b: a + b > 5),
]
solution = BacktrackSolver([x, y, z], constraints).solve()
print({var.name: value for var, value in solution.items()})
```

## Testing

```bash
pytest
```

## License

MIT

## JSON command line

Run `solveengine problem.json` after installation, or
`python -m solveengine.cli.runner problem.json` from a source checkout.
ProblemSerializer supports AllDifferent, Sum, and Table constraints. Unsupported
constraints fail explicitly rather than producing an incomplete saved model.

Legacy JSON `neq` constraints require two variables and are enforced. With `--limit`, incomplete single-solution search prints UNKNOWN; incomplete enumeration is marked explicitly. Both return exit status 2, while proven unsatisfiability returns status 1. The library exposes `solver.node_limit_reached` without changing the existing solution return types.
