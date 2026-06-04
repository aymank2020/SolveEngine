# Development Notes

## Architecture Decisions

### Why separate propagation from solver?
The propagation module (AC-3, AC-4) is independent of the search strategy.
This allows swapping propagators without changing the solver, and testing
propagation in isolation.

### Why trail-based undo instead of copying?
Copying entire domains at each decision point is O(d*n) per decision.
Trail-based undo is O(1) per modification and O(k) to undo k modifications.
For deep search trees with large domains, this is significantly faster.

### Why sparse sets for domains?
Sparse sets provide O(1) remove and O(1) restore (by size increment).
This is critical for the inner loop of propagation where values are
frequently removed and restored during backtracking.

## Performance Considerations

- AC-3 is simpler but may re-revise arcs unnecessarily
- AC-4 has optimal worst-case but higher constant overhead
- For most practical problems, AC-3 with good queue ordering wins

## Future Work

- Implement MAC (Maintaining Arc Consistency) as default
- Add restart strategies (geometric, Luby)
- Implement watched literals for nogood propagation
- Add parallel portfolio solver
