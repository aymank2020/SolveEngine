"""Domain storage backends and restoration strategies."""

from solveengine.storage.trail import Trail, TrailEntry
from solveengine.storage.sparse_set import SparseSet

__all__ = ["Trail", "TrailEntry", "SparseSet"]
