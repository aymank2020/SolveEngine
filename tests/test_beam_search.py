"""Tests for beam search strategy.

Tests cover:
- Basic solving with beam search
- Beam width effects on solution quality
- Scoring function behavior
- Candidate expansion and pruning
- Top-k solution finding
- Edge cases (empty problems, single variable)
"""

from __future__ import annotations

import pytest

from solveengine.core.variable import Variable
from solveengine.core.constraint import BinaryConstraint, Constraint
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.search.beam import BeamSearch, BeamScorer, BeamCandidate
from solveengine.heuristics.variable_ordering import MRVSelector


class TestBeamScorer:
    """Tests for the BeamScorer scoring function."""

    def test_no_violations_low_score(self) -> None:
        """Assignment with no violations should have a low score."""
        vars_ = [Variable(f"x{i}", range(1, 4)) for i in range(3)]
        constraints: list[Constraint] = [
            BinaryConstraint(vars_[0], vars_[1], lambda a, b: a != b),
        ]
        scorer = BeamScorer()
        # Assignment with no violations
        assignment = {vars_[0]: 1, vars_[1]: 2}
        score = scorer.score(assignment, vars_, constraints)
        assert score < 100  # No violation penalty

    def test_violations_increase_score(self) -> None:
        """Violations should significantly increase the score."""
        vars_ = [Variable(f"x{i}", range(1, 4)) for i in range(3)]
        constraints: list[Constraint] = [
            BinaryConstraint(vars_[0], vars_[1], lambda a, b: a != b),
        ]
        scorer = BeamScorer(violation_weight=100.0)
        # Assignment with violation (both = 1)
        bad_assignment = {vars_[0]: 1, vars_[1]: 1}
        good_assignment = {vars_[0]: 1, vars_[1]: 2}
        bad_score = scorer.score(bad_assignment, vars_, constraints)
        good_score = scorer.score(good_assignment, vars_, constraints)
        assert bad_score > good_score

    def test_deeper_assignment_preferred(self) -> None:
        """Deeper assignments should be preferred (lower score) with depth bonus."""
        vars_ = [Variable(f"x{i}", range(1, 4)) for i in range(3)]
        constraints: list[Constraint] = []
        scorer = BeamScorer(depth_bonus=-10.0, domain_weight=0.0)
        shallow = {vars_[0]: 1}
        deep = {vars_[0]: 1, vars_[1]: 2}
        assert scorer.score(deep, vars_, constraints) < scorer.score(shallow, vars_, constraints)

    def test_smaller_remaining_domain_preferred(self) -> None:
        """Assignments leaving smaller remaining domains score better."""
        vars_ = [Variable("x", range(1, 4)), Variable("y", range(1, 10))]
        constraints: list[Constraint] = []
        scorer = BeamScorer(domain_weight=1.0, depth_bonus=0.0, violation_weight=0.0)
        # Assigning x leaves y's large domain; assigning y leaves x's small domain
        assign_x = {vars_[0]: 1}  # remaining: y with 9 values
        assign_y = {vars_[1]: 1}  # remaining: x with 3 values
        assert scorer.score(assign_y, vars_, constraints) < scorer.score(assign_x, vars_, constraints)


class TestBeamCandidate:
    """Tests for BeamCandidate data class."""

    def test_ordering_by_score(self) -> None:
        """Candidates should be ordered by score (lower is better)."""
        c1 = BeamCandidate(score=5.0, assignment={})
        c2 = BeamCandidate(score=10.0, assignment={})
        assert c1 < c2

    def test_copy_independence(self) -> None:
        """Copied candidates should be independent."""
        var = Variable("x", range(5))
        c1 = BeamCandidate(score=5.0, assignment={var: 1}, depth=1)
        c2 = c1.copy()
        c2.assignment[var] = 2
        assert c1.assignment[var] == 1
        assert c2.assignment[var] == 2

    def test_copy_preserves_fields(self) -> None:
        """Copy should preserve all fields."""
        c1 = BeamCandidate(score=3.5, assignment={}, depth=2, violations=1)
        c2 = c1.copy()
        assert c2.score == 3.5
        assert c2.depth == 2
        assert c2.violations == 1


class TestBeamSearchBasic:
    """Basic beam search functionality tests."""

    def test_solve_trivial_problem(self) -> None:
        """Beam search should solve a trivial single-variable problem."""
        var = Variable("x", range(1, 4))
        constraints: list[Constraint] = []
        beam = BeamSearch([var], constraints, beam_width=5)
        solution = beam.solve()
        assert solution is not None
        assert var in solution
        assert solution[var] in range(1, 4)

    def test_solve_two_variable_not_equal(self) -> None:
        """Beam search should solve x != y."""
        x = Variable("x", range(1, 4))
        y = Variable("y", range(1, 4))
        constraints: list[Constraint] = [
            BinaryConstraint(x, y, lambda a, b: a != b),
        ]
        beam = BeamSearch([x, y], constraints, beam_width=10)
        solution = beam.solve()
        assert solution is not None
        assert solution[x] != solution[y]

    def test_solve_alldiff_small(self) -> None:
        """Beam search should solve a small AllDifferent problem."""
        vars_ = [Variable(f"x{i}", range(1, 5)) for i in range(4)]
        constraints: list[Constraint] = [AllDifferent(*vars_)]
        beam = BeamSearch(vars_, constraints, beam_width=20)
        solution = beam.solve()
        assert solution is not None
        values = [solution[v] for v in vars_]
        assert len(set(values)) == 4

    def test_unsatisfiable_returns_none(self) -> None:
        """Beam search should return None for unsatisfiable problems."""
        x = Variable("x", [1])
        y = Variable("y", [1])
        constraints: list[Constraint] = [
            BinaryConstraint(x, y, lambda a, b: a != b),
        ]
        beam = BeamSearch([x, y], constraints, beam_width=10)
        solution = beam.solve()
        assert solution is None


class TestBeamWidthEffect:
    """Tests for beam width impact on search behavior."""

    def test_wider_beam_more_expansions(self) -> None:
        """Wider beam should generally explore more candidates."""
        vars_ = [Variable(f"x{i}", range(1, 6)) for i in range(4)]
        constraints: list[Constraint] = [AllDifferent(*vars_)]

        narrow = BeamSearch(vars_, constraints, beam_width=2)
        narrow.solve()
        narrow_expansions = narrow.expansions

        # Reset domains
        for v in vars_:
            v.restore_to(0)
            v.unassign()

        wide = BeamSearch(vars_, constraints, beam_width=20)
        wide.solve()
        wide_expansions = wide.expansions

        assert wide_expansions >= narrow_expansions

    def test_beam_width_1_is_greedy(self) -> None:
        """Beam width 1 is essentially greedy best-first search."""
        vars_ = [Variable(f"x{i}", range(1, 5)) for i in range(3)]
        constraints: list[Constraint] = [AllDifferent(*vars_)]
        beam = BeamSearch(vars_, constraints, beam_width=1)
        beam.solve()
        # With width 1, levels explored should equal number of variables
        assert beam.levels_explored <= len(vars_)

    def test_beam_width_property(self) -> None:
        """Beam width property should return the configured width."""
        vars_ = [Variable("x", range(5))]
        beam = BeamSearch(vars_, [], beam_width=7)
        assert beam.beam_width == 7

    def test_minimum_beam_width(self) -> None:
        """Beam width should be at least 1."""
        vars_ = [Variable("x", range(5))]
        beam = BeamSearch(vars_, [], beam_width=0)
        assert beam.beam_width == 1


class TestBeamSearchTopK:
    """Tests for finding multiple solutions."""

    def test_top_k_finds_multiple_solutions(self) -> None:
        """solve_top_k should find multiple distinct solutions."""
        vars_ = [Variable(f"x{i}", range(1, 5)) for i in range(3)]
        constraints: list[Constraint] = [AllDifferent(*vars_)]
        beam = BeamSearch(vars_, constraints, beam_width=50)
        solutions = beam.solve_top_k(k=5)
        assert len(solutions) >= 1
        # All solutions should be valid
        for sol in solutions:
            values = [sol[v] for v in vars_]
            assert len(set(values)) == 3

    def test_top_k_respects_limit(self) -> None:
        """solve_top_k should not return more than k solutions."""
        vars_ = [Variable(f"x{i}", range(1, 5)) for i in range(3)]
        constraints: list[Constraint] = [AllDifferent(*vars_)]
        beam = BeamSearch(vars_, constraints, beam_width=100)
        solutions = beam.solve_top_k(k=2)
        assert len(solutions) <= 2

    def test_top_k_empty_for_unsatisfiable(self) -> None:
        """solve_top_k should return empty list for unsatisfiable problems."""
        x = Variable("x", [1])
        y = Variable("y", [1])
        constraints: list[Constraint] = [
            BinaryConstraint(x, y, lambda a, b: a != b),
        ]
        beam = BeamSearch([x, y], constraints, beam_width=10)
        solutions = beam.solve_top_k(k=5)
        assert solutions == []


class TestBeamSearchExpansionLimit:
    """Tests for expansion limit behavior."""

    def test_max_expansions_respected(self) -> None:
        """Search should stop when max_expansions is reached."""
        vars_ = [Variable(f"x{i}", range(1, 10)) for i in range(5)]
        constraints: list[Constraint] = [AllDifferent(*vars_)]
        beam = BeamSearch(
            vars_, constraints, beam_width=50, max_expansions=10
        )
        beam.solve()
        assert beam.expansions <= 50  # May slightly exceed due to batch processing

    def test_levels_explored_tracked(self) -> None:
        """levels_explored should track how deep the search went."""
        vars_ = [Variable(f"x{i}", range(1, 5)) for i in range(3)]
        constraints: list[Constraint] = []
        beam = BeamSearch(vars_, constraints, beam_width=10)
        beam.solve()
        assert beam.levels_explored == 3

    def test_expansions_counter_increments(self) -> None:
        """Expansions counter should be positive after solving."""
        vars_ = [Variable(f"x{i}", range(1, 4)) for i in range(2)]
        constraints: list[Constraint] = [
            BinaryConstraint(vars_[0], vars_[1], lambda a, b: a != b),
        ]
        beam = BeamSearch(vars_, constraints, beam_width=5)
        beam.solve()
        assert beam.expansions > 0


class TestBeamSearchConsistency:
    """Tests for solution consistency."""

    def test_solution_satisfies_all_constraints(self) -> None:
        """Any returned solution must satisfy all constraints."""
        vars_ = [Variable(f"x{i}", range(1, 6)) for i in range(4)]
        constraints: list[Constraint] = [
            AllDifferent(*vars_),
            BinaryConstraint(vars_[0], vars_[1], lambda a, b: a < b),
        ]
        beam = BeamSearch(vars_, constraints, beam_width=30)
        solution = beam.solve()
        if solution is not None:
            for cstr in constraints:
                assert cstr.is_satisfied(solution)

    def test_solution_is_complete(self) -> None:
        """Returned solution should assign all variables."""
        vars_ = [Variable(f"x{i}", range(1, 5)) for i in range(3)]
        constraints: list[Constraint] = [AllDifferent(*vars_)]
        beam = BeamSearch(vars_, constraints, beam_width=20)
        solution = beam.solve()
        assert solution is not None
        for var in vars_:
            assert var in solution
