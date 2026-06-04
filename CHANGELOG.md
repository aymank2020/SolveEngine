# Changelog

## 0.4.0 (2026-06-04)

- Add constraint graph decomposition module
- Add symmetry breaking transforms
- Improve backjumping with conflict-directed learning
- Add persistence module for state checkpointing
- Add CLI interface for solving from XCSP3 files

## 0.3.0 (2026-05-10)

- Add global constraints (AllDifferent, Sum, Element, Cardinality)
- Add nogood learning with clause minimization
- Add limited discrepancy search strategy
- Expand test coverage for propagation edge cases

## 0.2.0 (2026-04-15)

- Add arc consistency (AC-3 and AC-4) propagation
- Add variable ordering heuristics (MRV, degree, dom/wdeg)
- Add value ordering (LCV)
- Add forward checking to backtracking solver
- Add modeling API for high-level problem specification

## 0.1.0 (2026-03-12)

- Initial release
- Core data structures (Variable, Domain, Constraint)
- Basic backtracking solver
- Node consistency propagation
- DFS search strategy
