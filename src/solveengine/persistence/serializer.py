"""Problem serialization for saving/loading CSP instances."""

from __future__ import annotations

import json
from typing import Any

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint, BinaryConstraint, TableConstraint
from solveengine.global_cstr.alldiff import AllDifferent
from solveengine.global_cstr.linear import SumConstraint, ComparisonOp


class ProblemSerializer:
    """Serializes and deserializes CSP problem instances.

    Supports a JSON-based format for storing problem definitions.
    Note: Lambda-based constraints cannot be serialized (only table
    and named global constraints are supported).
    """

    def serialize(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> dict[str, Any]:
        """Serialize a CSP problem to a dictionary."""
        names = [v.name for v in variables]
        if len(set(names)) != len(names):
            raise ValueError("Variable names must be unique")
        known_variables = set(variables)
        for constraint in constraints:
            if any(v not in known_variables for v in constraint.variables):
                raise ValueError("Constraint references a variable outside the problem")
        return {
            "version": "1.0",
            "variables": [self._serialize_variable(v) for v in variables],
            "constraints": [self._serialize_constraint(c) for c in constraints],
        }

    def to_json(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
        indent: int = 2,
    ) -> str:
        """Serialize to JSON string."""
        data = self.serialize(variables, constraints)
        return json.dumps(data, indent=indent)

    def _serialize_variable(self, var: Variable) -> dict[str, Any]:
        return {
            "name": var.name,
            "index": var.index,
            "domain": sorted(var.domain.values()),
        }

    def _serialize_constraint(self, cstr: Constraint) -> dict[str, Any]:
        base = {
            "type": type(cstr).__name__,
            "variables": [v.name for v in cstr.variables],
            "name": cstr.name,
        }

        if isinstance(cstr, AllDifferent):
            base["type"] = "AllDifferent"
        elif isinstance(cstr, SumConstraint):
            base["type"] = "Sum"
            base["target"] = cstr.target
            base["op"] = cstr.op.value
        elif isinstance(cstr, TableConstraint):
            base["type"] = "Table"
            base["tuples"] = [list(t) for t in cstr.allowed_tuples]

        else:
            raise ValueError(f"Unsupported constraint type: {type(cstr).__name__}")

        return base

    def deserialize_variables(self, data: dict[str, Any]) -> list[Variable]:
        """Deserialize variables from a dictionary."""
        self._validate_document(data)
        variables = []
        names: set[str] = set()
        for var_data in data.get("variables", []):
            if not isinstance(var_data, dict):
                raise ValueError("Each variable must be an object")
            name = var_data.get("name")
            domain = var_data.get("domain")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Variable names must be nonempty and unique")
            if not isinstance(domain, list) or not domain or any(type(v) is not int for v in domain):
                raise ValueError(f"Variable {name!r} needs a nonempty integer domain")
            names.add(name)
            var = Variable(name, domain)
            variables.append(var)
        return variables

    def deserialize_constraints(
        self,
        data: dict[str, Any],
        variables: list[Variable],
    ) -> list[Constraint]:
        """Deserialize constraints from a dictionary.

        Requires the variables to already be created.
        """
        self._validate_document(data)
        var_by_name = {v.name: v for v in variables}
        if len(var_by_name) != len(variables):
            raise ValueError("Variable names must be unique")
        constraints: list[Constraint] = []

        for cstr_data in data.get("constraints", []):
            if not isinstance(cstr_data, dict):
                raise ValueError("Each constraint must be an object")
            cstr_type = cstr_data.get("type")
            if cstr_type not in ("AllDifferent", "Sum", "Table", "neq"):
                raise ValueError(f"Unsupported constraint type: {cstr_type!r}")
            var_names = cstr_data.get("variables")
            if not isinstance(var_names, list) or any(not isinstance(name, str) for name in var_names):
                raise ValueError("Constraint variables must be a list of names")
            unknown = [name for name in var_names if name not in var_by_name]
            if unknown:
                raise ValueError(f"Unknown constraint variable: {unknown[0]!r}")
            cstr_vars = [var_by_name[name] for name in var_names]

            if cstr_type == "neq":
                if len(cstr_vars) != 2:
                    raise ValueError("neq requires exactly two variables")
                constraints.append(BinaryConstraint(
                    cstr_vars[0], cstr_vars[1], lambda a, b: a != b,
                    name=cstr_data.get("name", ""),
                ))
            elif cstr_type == "AllDifferent":
                constraints.append(AllDifferent(*cstr_vars, name=cstr_data.get("name", "")))
            elif cstr_type == "Sum":
                if type(cstr_data.get("target")) is not int:
                    raise ValueError("Sum target must be an integer")
                op = ComparisonOp(cstr_data["op"])
                constraints.append(SumConstraint(
                    cstr_vars, cstr_data["target"], op, name=cstr_data.get("name", "")
                ))
            elif cstr_type == "Table":
                rows = cstr_data.get("tuples")
                if not isinstance(rows, list) or any(
                    not isinstance(row, list) or len(row) != len(cstr_vars)
                    or any(type(value) is not int for value in row)
                    for row in rows
                ):
                    raise ValueError("Table tuples must contain one integer per variable")
                tuples = {tuple(t) for t in cstr_data["tuples"]}
                constraints.append(TableConstraint(
                    cstr_vars, tuples, name=cstr_data.get("name", "")
                ))

        return constraints

    @staticmethod
    def _validate_document(data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("Problem must be a JSON object")
        if data.get("version", "1.0") != "1.0":
            raise ValueError("Unsupported problem format version")
        for field in ("variables", "constraints"):
            if not isinstance(data.get(field, []), list):
                raise ValueError(f"Problem {field} must be a list")
