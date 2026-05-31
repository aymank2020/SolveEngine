"""Tests for persistence module: checkpoints and serialization."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.global_cstr.linear import SumConstraint, ComparisonOp
from solveengine.persistence.checkpoint import CheckpointManager
from solveengine.persistence.serializer import ProblemSerializer


class TestCheckpointManager:
    def test_save_and_restore(self):
        """Can save and restore solver state."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(3)]
        manager = CheckpointManager(vars)

        assignment = {vars[0]: 1, vars[1]: 2}
        cp_id = manager.save(assignment, depth=2)
        assert manager.num_checkpoints == 1

        restored = manager.restore(cp_id)
        assert restored[vars[0].index] == 1
        assert restored[vars[1].index] == 2

    def test_multiple_checkpoints(self):
        """Multiple checkpoints can coexist."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(2)]
        manager = CheckpointManager(vars)

        cp1 = manager.save({vars[0]: 1}, depth=1)
        cp2 = manager.save({vars[0]: 2, vars[1]: 3}, depth=2)
        assert manager.num_checkpoints == 2

    def test_capacity_limit(self):
        """Checkpoints respect capacity limit."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(2)]
        manager = CheckpointManager(vars, max_checkpoints=3)

        for i in range(5):
            manager.save({vars[0]: i % 4 + 1}, depth=i)
        assert manager.num_checkpoints <= 3

    def test_discard(self):
        """Can discard a specific checkpoint."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(2)]
        manager = CheckpointManager(vars)
        cp_id = manager.save({vars[0]: 1}, depth=1)
        assert manager.discard(cp_id)
        assert manager.num_checkpoints == 0


class TestProblemSerializer:
    def test_serialize_variables(self):
        """Variables serialize to correct format."""
        vars = [Variable("x", [1, 2, 3]), Variable("y", [4, 5, 6])]
        serializer = ProblemSerializer()
        data = serializer.serialize(vars, [])
        assert len(data["variables"]) == 2
        assert data["variables"][0]["name"] == "x"

    def test_roundtrip_variables(self):
        """Variables survive serialize/deserialize roundtrip."""
        vars = [Variable("x", [1, 2, 3]), Variable("y", [4, 5, 6])]
        serializer = ProblemSerializer()
        data = serializer.serialize(vars, [])
        restored = serializer.deserialize_variables(data)
        assert len(restored) == 2
        assert restored[0].name == "x"
        assert restored[0].domain.values() == frozenset({1, 2, 3})

    def test_serialize_alldiff(self):
        """AllDifferent constraint serializes correctly."""
        vars = [Variable(f"v{i}", [1, 2, 3]) for i in range(3)]
        cstr = AllDifferent(*vars)
        serializer = ProblemSerializer()
        data = serializer.serialize(vars, [cstr])
        assert data["constraints"][0]["type"] == "AllDifferent"

    def test_roundtrip_constraints(self):
        """Constraints survive serialize/deserialize roundtrip."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(3)]
        cstr = SumConstraint(vars, 10, ComparisonOp.EQ)
        serializer = ProblemSerializer()
        data = serializer.serialize(vars, [cstr])

        restored_vars = serializer.deserialize_variables(data)
        restored_cstrs = serializer.deserialize_constraints(data, restored_vars)
        assert len(restored_cstrs) == 1

    def test_json_output(self):
        """to_json produces valid JSON string."""
        vars = [Variable("x", [1, 2])]
        serializer = ProblemSerializer()
        json_str = serializer.to_json(vars, [])
        assert '"variables"' in json_str
        assert '"x"' in json_str
