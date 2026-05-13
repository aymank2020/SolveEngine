"""Tests for the high-level modeling API."""

import pytest
from solveengine.modeling.model import Model


class TestModel:
    """Integration tests for the Model API."""

    def test_simple_not_equal(self):
        """Simple not-equal problem has a solution."""
        model = Model()
        x = model.int_var("x", 1, 3)
        y = model.int_var("y", 1, 3)
        model.add_not_equal(x, y)
        result = model.solve()
        assert result is not None
        assert result["x"] != result["y"]

    def test_all_different_3vars(self):
        """AllDifferent with 3 vars and domain [1,3] has solutions."""
        model = Model()
        x = model.int_var("x", 1, 3)
        y = model.int_var("y", 1, 3)
        z = model.int_var("z", 1, 3)
        model.add_all_different(x, y, z)
        result = model.solve()
        assert result is not None
        assert len(set(result.values())) == 3

    def test_sum_constraint(self):
        """Sum constraint produces correct total."""
        model = Model()
        x = model.int_var("x", 1, 5)
        y = model.int_var("y", 1, 5)
        model.add_sum_eq([x, y], 6)
        result = model.solve()
        assert result is not None
        assert result["x"] + result["y"] == 6

    def test_unsatisfiable_model(self):
        """Unsatisfiable model returns None."""
        model = Model()
        x = model.int_var("x", 1, 2)
        y = model.int_var("y", 1, 2)
        z = model.int_var("z", 1, 2)
        model.add_all_different(x, y, z)  # 3 vars, domain size 2 → impossible
        result = model.solve()
        assert result is None

    def test_solve_all_completeness(self):
        """solve_all finds all valid solutions."""
        model = Model()
        x = model.int_var("x", 1, 2)
        y = model.int_var("y", 1, 2)
        model.add_not_equal(x, y)
        solutions = model.solve_all()
        # x=1,y=2 and x=2,y=1
        assert len(solutions) == 2

    def test_less_than_ordering(self):
        """Less-than constraint enforces ordering."""
        model = Model()
        x = model.int_var("x", 1, 5)
        y = model.int_var("y", 1, 5)
        model.add_less_than(x, y)
        result = model.solve()
        assert result is not None
        assert result["x"] < result["y"]
