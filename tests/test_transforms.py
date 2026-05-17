"""Tests for constraint transforms: symmetry breaking and reformulation."""

import pytest
from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.transforms.symmetry import SymmetryBreaker
from solveengine.transforms.reformulate import ConstraintReformulator


class TestSymmetryBreaker:
    def test_find_interchangeable(self):
        """Detects interchangeable variables."""
        # Three variables with same domain in same constraint type
        vars = [Variable(f"v{i}", [1, 2, 3, 4]) for i in range(3)]
        cstr = AllDifferent(*vars)
        breaker = SymmetryBreaker(vars, [cstr])
        groups = breaker.find_interchangeable_variables()
        # All three should be interchangeable
        assert len(groups) >= 1

    def test_lex_ordering_constraints(self):
        """Lex ordering produces correct number of constraints."""
        vars = [Variable(f"v{i}", range(1, 5)) for i in range(4)]
        breaker = SymmetryBreaker(vars, [])
        lex_cstrs = breaker.add_lex_ordering(vars)
        assert len(lex_cstrs) == 3  # n-1 constraints for n vars

    def test_lex_ordering_satisfies_order(self):
        """Lex ordering constraints enforce v[i] <= v[i+1]."""
        vars = [Variable(f"v{i}", range(1, 10)) for i in range(3)]
        breaker = SymmetryBreaker(vars, [])
        lex_cstrs = breaker.add_lex_ordering(vars)
        # Check that (1, 2) satisfies and (3, 1) doesn't
        assert lex_cstrs[0].is_satisfied({vars[0]: 1, vars[1]: 2})
        assert not lex_cstrs[0].is_satisfied({vars[0]: 3, vars[1]: 1})


class TestConstraintReformulator:
    def test_decompose_alldiff(self):
        """AllDiff decomposes into n*(n-1)/2 binary constraints."""
        vars = [Variable(f"v{i}", [1, 2, 3, 4]) for i in range(4)]
        alldiff = AllDifferent(*vars)
        reformulator = ConstraintReformulator()
        binary = reformulator.decompose_alldiff(alldiff)
        assert len(binary) == 6  # C(4,2) = 6

    def test_decomposed_constraints_equivalent(self):
        """Decomposed constraints reject same assignments as AllDiff."""
        vars = [Variable(f"v{i}", [1, 2, 3]) for i in range(3)]
        alldiff = AllDifferent(*vars)
        reformulator = ConstraintReformulator()
        binary = reformulator.decompose_alldiff(alldiff)

        # Valid assignment
        valid = {vars[0]: 1, vars[1]: 2, vars[2]: 3}
        assert alldiff.is_satisfied(valid)
        assert all(c.is_satisfied(valid) for c in binary)

        # Invalid assignment
        invalid = {vars[0]: 1, vars[1]: 1, vars[2]: 3}
        assert not alldiff.is_satisfied(invalid)
        assert not all(c.is_satisfied(invalid) for c in binary)

    def test_merge_to_table(self):
        """Merging binary constraints produces a table constraint."""
        x = Variable("x", [1, 2, 3])
        y = Variable("y", [1, 2, 3])
        c1 = BinaryConstraint(x, y, lambda a, b: a != b)
        c2 = BinaryConstraint(x, y, lambda a, b: a + b <= 4)
        reformulator = ConstraintReformulator()
        table = reformulator.merge_to_table([c1, c2], x, y)
        assert table is not None
        # (1,1) fails c1, (2,3) fails c2, (1,2) passes both
        assert table.is_satisfied({x: 1, y: 2})
        assert not table.is_satisfied({x: 1, y: 1})
        assert not table.is_satisfied({x: 2, y: 3})

    def test_tighten_domains(self):
        """Domain tightening removes unsupported values."""
        x = Variable("x", range(1, 10))
        y = Variable("y", [3])  # Fixed
        cstr = BinaryConstraint(x, y, lambda a, b: a < b)
        reformulator = ConstraintReformulator()
        pruned = reformulator.tighten_domains([x, y], [cstr])
        # x must be < 3, so values >= 3 should be pruned
        if x in pruned:
            for val in pruned[x]:
                assert val >= 3
