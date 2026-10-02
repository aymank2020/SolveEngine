"""Bounded search distinguishes incomplete work and enforces every constraint."""
import json
import pytest
from solveengine import Variable, BinaryConstraint, BacktrackSolver
from solveengine.cli.runner import run_cli


def test_all_solutions_check_constraints_without_forward_check():
    x, y = Variable("x", [1, 2]), Variable("y", [1, 2])
    c = BinaryConstraint(x, y, lambda a, b: a != b)
    solver = BacktrackSolver([x, y], [c], use_forward_check=False)
    solutions = solver.solve_all()
    assert len(solutions) == 2
    assert all(c.is_satisfied(solution) for solution in solutions)


@pytest.mark.parametrize("all_solutions", [False, True])
def test_cli_node_limit_reports_incomplete(tmp_path, capsys, all_solutions):
    data = {"variables": [{"name": "x", "domain": [1, 2]}, {"name": "y", "domain": [1, 2]}]}
    path = tmp_path / "limited.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    args = [str(path), "--limit", "1"] + (["--all"] if all_solutions else [])
    assert run_cli(args) == 2
    output = capsys.readouterr().out
    assert "node limit reached" in output
    assert "UNSATISFIABLE" not in output


def test_solver_can_continue_after_raising_limit():
    variables = [Variable("x", [1, 2]), Variable("y", [1, 2])]
    solver = BacktrackSolver(variables, [])
    solver.set_node_limit(1)
    assert solver.solve() is None
    assert solver.node_limit_reached
    solver.set_node_limit(100)
    assert solver.solve() is not None
    assert not solver.node_limit_reached
