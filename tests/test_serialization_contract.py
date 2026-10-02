"""End-to-end protection against weakening malformed saved problems."""
import json

import pytest

from solveengine import BinaryConstraint, Variable
from solveengine.cli.runner import run_cli
from solveengine.persistence.serializer import ProblemSerializer


def test_callable_constraints_cannot_be_silently_lost():
    x, y = Variable("x", [1]), Variable("y", [1])
    constraint = BinaryConstraint(x, y, lambda a, b: a != b)
    with pytest.raises(ValueError, match="Unsupported constraint"):
        ProblemSerializer().serialize([x, y], [constraint])


@pytest.mark.parametrize("data", [
    {"variables": [{"name": "x", "domain": [1]}], "constraints": [{"type": "BinaryConstraint", "variables": ["x"]}]},
    {"variables": [{"name": "x", "domain": [1]}, {"name": "x", "domain": [2]}]},
    {"variables": [{"name": "x", "domain": [1]}], "constraints": [{"type": "AllDifferent", "variables": ["missing"]}]},
    {"version": "2.0", "variables": []},
    {"variables": [{"name": "x", "domain": [True]}]},
    {"variables": [{"name": "x", "domain": [1.5]}]},
    {"variables": {}},
    {"variables": [{"name": "x", "domain": [1]}], "constraints": [{"type": "Sum", "variables": ["x"], "target": "bad", "op": "=="}]},
    {"variables": [{"name": "x", "domain": [1]}], "constraints": [{"type": "Table", "variables": ["x"], "tuples": [[1, 2]]}]},
    [],
])
def test_invalid_problem_cli_fails_without_sat(tmp_path, capsys, data):
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert run_cli([str(path)]) == 1
    output = capsys.readouterr()
    assert "Invalid problem:" in output.err
    assert "SATISFIABLE" not in output.out


def test_utf8_problem_cli_preserves_constraint(tmp_path, capsys):
    data = {
        "version": "1.0",
        "variables": [{"name": "أ", "domain": [1]}, {"name": "ب", "domain": [1]}],
        "constraints": [{"type": "AllDifferent", "variables": ["أ", "ب"]}],
    }
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert run_cli([str(path)]) == 1
    assert "UNSATISFIABLE" in capsys.readouterr().out


def test_legacy_neq_constraint_is_enforced(tmp_path, capsys):
    data = {"variables": [{"name": "x", "domain": [1]}, {"name": "y", "domain": [1]}], "constraints": [{"type": "neq", "variables": ["x", "y"]}]}
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert run_cli([str(path)]) == 1
    assert "UNSATISFIABLE" in capsys.readouterr().out
