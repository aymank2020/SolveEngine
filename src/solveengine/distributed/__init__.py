"""Distributed solving: work splitting and result aggregation."""

from solveengine.distributed.splitter import WorkSplitter, SubProblem
from solveengine.distributed.aggregator import ResultAggregator

__all__ = ["WorkSplitter", "SubProblem", "ResultAggregator"]
