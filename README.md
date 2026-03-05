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
from solveengine import Variable, Solver, AllDifferent

# Create variables
x = Variable("x", range(1, 10))
y = Variable("y", range(1, 10))
z = Variable("z", range(1, 10))

# Add constraints
solver = Solver()
solver.add_variables(x, y, z)
solver.add_constraint(AllDifferent(x, y, z))
solver.add_constraint(lambda x, y: x + y > 5)

# Solve
solution = solver.solve()
```

## Testing

```bash
pytest
```

## License

MIT
