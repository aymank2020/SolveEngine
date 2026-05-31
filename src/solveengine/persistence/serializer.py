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

        return base

    def deserialize_variables(self, data: dict[str, Any]) -> list[Variable]:
        """Deserialize variables from a dictionary."""
        variables = []
        for var_data in data.get("variables", []):
            var = Variable(var_data["name"], var_data["domain"])
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
        var_by_name = {v.name: v for v in variables}
        constraints: list[Constraint] = []

        for cstr_data in data.get("constraints", []):
            cstr_type = cstr_data["type"]
            var_names = cstr_data["variables"]
            cstr_vars = [var_by_name[name] for name in var_names]

            if cstr_type == "AllDifferent":
                constraints.append(AllDifferent(*cstr_vars, name=cstr_data.get("name", "")))
            elif cstr_type == "Sum":
                op = ComparisonOp(cstr_data["op"])
                constraints.append(SumConstraint(
                    cstr_vars, cstr_data["target"], op, name=cstr_data.get("name", "")
                ))
            elif cstr_type == "Table":
                tuples = {tuple(t) for t in cstr_data["tuples"]}
                constraints.append(TableConstraint(
                    cstr_vars, tuples, name=cstr_data.get("name", "")
                ))

        return constraints
