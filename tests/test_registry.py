"""Tests for variable and constraint registry."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint, UnaryConstraint
from solveengine.core.registry import VariableRegistry, ConstraintRegistry
from solveengine.global_cstr.alldiff import AllDifferent


class TestVariableRegistry:
    def test_register_and_lookup_by_name(self):
        reg = VariableRegistry()
        v = Variable("x", [1, 2, 3])
        reg.register(v)
        assert reg.get_by_name("x") is v

    def test_register_and_lookup_by_index(self):
        reg = VariableRegistry()
        v = Variable("y", [1, 2])
        reg.register(v)
        assert reg.get_by_index(v.index) is v

    def test_duplicate_name_raises(self):
        reg = VariableRegistry()
        v1 = Variable("x", [1, 2])
        reg.register(v1)
        v2 = Variable("x", [3, 4])
        with pytest.raises(ValueError):
            reg.register(v2)

    def test_size_tracking(self):
        reg = VariableRegistry()
        assert reg.size == 0
        reg.register(Variable("a", [1]))
        reg.register(Variable("b", [2]))
        assert reg.size == 2

    def test_get_unassigned(self):
        reg = VariableRegistry()
        v1 = Variable("a", [1, 2, 3])
        v2 = Variable("b", [1, 2, 3])
        reg.register(v1)
        reg.register(v2)
        v1.assign(1)
        unassigned = reg.get_unassigned()
        assert v2 in unassigned
        assert v1 not in unassigned

    def test_total_domain_size(self):
        reg = VariableRegistry()
        reg.register(Variable("a", [1, 2, 3]))
        reg.register(Variable("b", [1, 2]))
        assert reg.total_domain_size() == 5

    def test_min_domain_variable(self):
        reg = VariableRegistry()
        v1 = Variable("big", range(1, 100))
        v2 = Variable("small", [1, 2])
        reg.register(v1)
        reg.register(v2)
        assert reg.min_domain_variable() is v2

    def test_register_many(self):
        reg = VariableRegistry()
        vars = [Variable(f"v{i}", [1, 2]) for i in range(5)]
        reg.register_many(vars)
        assert reg.size == 5


class TestConstraintRegistry:
    def test_register_and_get_for_variable(self):
        reg = ConstraintRegistry()
        x = Variable("x", [1, 2, 3])
        y = Variable("y", [1, 2, 3])
        cstr = BinaryConstraint(x, y, lambda a, b: a != b)
        reg.register(cstr)
        assert cstr in reg.get_for_variable(x)
        assert cstr in reg.get_for_variable(y)

    def test_categorization(self):
        reg = ConstraintRegistry()
        x = Variable("x", [1, 2, 3])
        y = Variable("y", [1, 2, 3])
        z = Variable("z", [1, 2, 3])
        binary = BinaryConstraint(x, y, lambda a, b: a != b)
        unary = UnaryConstraint(x, lambda a: a > 1)
        glob = AllDifferent(x, y, z)
        reg.register(binary)
        reg.register(unary)
        reg.register(glob)
        assert binary in reg.binary_constraints
        assert unary in reg.unary_constraints
        assert glob in reg.global_constraints

    def test_get_between(self):
        reg = ConstraintRegistry()
        x = Variable("x", [1, 2])
        y = Variable("y", [1, 2])
        z = Variable("z", [1, 2])
        c1 = BinaryConstraint(x, y, lambda a, b: a != b)
        c2 = BinaryConstraint(y, z, lambda a, b: a != b)
        reg.register(c1)
        reg.register(c2)
        between_xy = reg.get_between(x, y)
        assert c1 in between_xy
        assert c2 not in between_xy

    def test_get_neighbors(self):
        reg = ConstraintRegistry()
        x = Variable("x", [1, 2])
        y = Variable("y", [1, 2])
        z = Variable("z", [1, 2])
        reg.register(BinaryConstraint(x, y, lambda a, b: a != b))
        reg.register(BinaryConstraint(x, z, lambda a, b: a != b))
        neighbors = reg.get_neighbors(x)
        assert y in neighbors
        assert z in neighbors
        assert x not in neighbors

    def test_constraint_degree(self):
        reg = ConstraintRegistry()
        x = Variable("x", [1, 2])
        y = Variable("y", [1, 2])
        z = Variable("z", [1, 2])
        reg.register(BinaryConstraint(x, y, lambda a, b: a != b))
        reg.register(BinaryConstraint(x, z, lambda a, b: a != b))
        assert reg.constraint_degree(x) == 2
        assert reg.constraint_degree(y) == 1
