"""Combinatorial modeling helpers for constraint satisfaction problems.

Provides high-level abstractions for common combinatorial structures:
- Permutation constraints (bijection between variables and values)
- Subset selection (choose k items from n)
- Partition constraints (divide items into groups)
- Bin packing helpers (assign items to bins with capacity limits)

These helpers build on the core Model class to provide domain-specific
interfaces for combinatorial optimization problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from solveengine.core.variable import Variable
from solveengine.core.constraint import Constraint
from solveengine.modeling.model import Model
from solveengine.global_cstr.alldiff import AllDifferent


@dataclass
class BinPackingItem:
    """An item to be packed into a bin.

    Attributes:
        item_id: Unique identifier.
        name: Human-readable name.
        size: Size/weight of the item.
        bin_var: Variable representing which bin this item is assigned to.
    """

    item_id: int
    name: str
    size: int
    bin_var: Variable | None = None


@dataclass
class Bin:
    """A bin with limited capacity.

    Attributes:
        bin_id: Unique identifier.
        name: Human-readable name.
        capacity: Maximum total size of items that can fit.
    """

    bin_id: int
    name: str
    capacity: int


@dataclass
class Partition:
    """A partition of items into groups.

    Attributes:
        num_groups: Number of groups in the partition.
        group_vars: Variables indicating group assignment for each item.
        group_sizes: Optional constraints on group sizes.
    """

    num_groups: int
    group_vars: list[Variable] = field(default_factory=list)
    group_sizes: list[tuple[int, int]] = field(default_factory=list)


class PermutationModel:
    """Model for permutation constraints.

    Ensures that a set of variables forms a permutation of a given
    value set (bijection). Each value is used exactly once.

    Example:
        perm = PermutationModel(n=5)
        # Variables x[0]..x[4] form a permutation of {0,1,2,3,4}
        solution = perm.solve()
    """

    def __init__(self, n: int, values: Sequence[int] | None = None) -> None:
        """Create a permutation model.

        Args:
            n: Number of variables (and values).
            values: The value set to permute (defaults to 0..n-1).
        """
        self._n = n
        self._values = list(values) if values else list(range(n))
        if len(self._values) != n:
            raise ValueError(f"Need exactly {n} values for permutation, got {len(self._values)}")
        self._model = Model()
        self._variables: list[Variable] = []
        self._build()

    def _build(self) -> None:
        """Build the permutation model."""
        for i in range(self._n):
            var = self._model.int_var_values(f"perm_{i}", self._values)
            self._variables.append(var)
        self._model.add_all_different(*self._variables)

    @property
    def variables(self) -> list[Variable]:
        """The permutation variables."""
        return list(self._variables)

    def add_fixed_point(self, position: int, value: int) -> None:
        """Fix a specific position to a specific value.

        Args:
            position: Index of the variable to fix.
            value: Value to assign at that position.
        """
        self._model.add_unary(
            self._variables[position],
            lambda v, target=value: v == target,
            f"fixed_{position}={value}",
        )

    def add_position_constraint(
        self, position: int, allowed_values: Sequence[int]
    ) -> None:
        """Restrict a position to a subset of values.

        Args:
            position: Index of the variable to restrict.
            allowed_values: Allowed values at this position.
        """
        allowed_set = set(allowed_values)
        self._model.add_unary(
            self._variables[position],
            lambda v, s=allowed_set: v in s,
            f"restrict_{position}",
        )

    def add_adjacency_constraint(
        self, predicate: callable
    ) -> None:
        """Add a constraint between consecutive positions.

        Args:
            predicate: Function (val_i, val_i+1) -> bool for consecutive values.
        """
        for i in range(self._n - 1):
            self._model.add_binary(
                self._variables[i],
                self._variables[i + 1],
                predicate,
                f"adj_{i}_{i+1}",
            )

    def solve(self) -> list[int] | None:
        """Solve the permutation problem.

        Returns the permutation as a list of values, or None if infeasible.
        """
        solution = self._model.solve()
        if solution is None:
            return None
        return [solution[f"perm_{i}"] for i in range(self._n)]

    def solve_all(self) -> list[list[int]]:
        """Find all valid permutations."""
        solutions = self._model.solve_all()
        return [
            [sol[f"perm_{i}"] for i in range(self._n)]
            for sol in solutions
        ]


class SubsetSelectionModel:
    """Model for selecting k items from n with constraints.

    Uses binary indicator variables to represent selection.

    Example:
        subset = SubsetSelectionModel(items=10, select_count=3)
        subset.add_mutual_exclusion(0, 1)  # Can't select both 0 and 1
        solution = subset.solve()
    """

    def __init__(
        self,
        items: int,
        select_count: int,
        item_names: Sequence[str] | None = None,
    ) -> None:
        """Create a subset selection model.

        Args:
            items: Total number of items available.
            select_count: Exact number of items to select.
            item_names: Optional names for items.
        """
        self._n = items
        self._k = select_count
        self._names = list(item_names) if item_names else [f"item_{i}" for i in range(items)]
        self._model = Model()
        self._indicators: list[Variable] = []
        self._build()

    def _build(self) -> None:
        """Build the subset selection model with indicator variables."""
        for i in range(self._n):
            # Binary indicator: 0 = not selected, 1 = selected
            var = self._model.int_var(f"sel_{self._names[i]}", 0, 1)
            self._indicators.append(var)

        # Exactly k items selected: sum of indicators = k
        self._model.add_sum_eq(self._indicators, self._k)

    @property
    def indicators(self) -> list[Variable]:
        """Binary indicator variables for each item."""
        return list(self._indicators)

    def add_mutual_exclusion(self, item_a: int, item_b: int) -> None:
        """At most one of item_a and item_b can be selected.

        Args:
            item_a: First item index.
            item_b: Second item index.
        """
        self._model.add_binary(
            self._indicators[item_a],
            self._indicators[item_b],
            lambda a, b: a + b <= 1,
            f"mutex_{self._names[item_a]}_{self._names[item_b]}",
        )

    def add_implication(self, if_selected: int, then_selected: int) -> None:
        """If item if_selected is chosen, then_selected must also be chosen.

        Args:
            if_selected: Item whose selection implies the other.
            then_selected: Item that must be selected if the first is.
        """
        self._model.add_binary(
            self._indicators[if_selected],
            self._indicators[then_selected],
            lambda a, b: a <= b,  # if a=1 then b must be 1
            f"implies_{self._names[if_selected]}->{self._names[then_selected]}",
        )

    def add_at_least_one_of(self, items: Sequence[int]) -> None:
        """At least one of the given items must be selected.

        Args:
            items: Indices of items, at least one must be selected.
        """
        selected_vars = [self._indicators[i] for i in items]
        self._model.add_sum_ge(selected_vars, 1)

    def solve(self) -> list[int] | None:
        """Solve and return indices of selected items.

        Returns a list of selected item indices, or None if infeasible.
        """
        solution = self._model.solve()
        if solution is None:
            return None
        selected = []
        for i in range(self._n):
            if solution.get(f"sel_{self._names[i]}", 0) == 1:
                selected.append(i)
        return selected

    def solve_all(self) -> list[list[int]]:
        """Find all valid subsets."""
        solutions = self._model.solve_all()
        results = []
        for sol in solutions:
            selected = [
                i for i in range(self._n)
                if sol.get(f"sel_{self._names[i]}", 0) == 1
            ]
            results.append(selected)
        return results


class PartitionModel:
    """Model for partitioning items into groups.

    Each item is assigned to exactly one group. Constraints can be
    placed on group sizes and item co-occurrence.

    Example:
        part = PartitionModel(items=12, num_groups=3)
        part.set_group_size(min_size=3, max_size=5)
        part.add_same_group(0, 1)  # Items 0 and 1 must be together
        solution = part.solve()
    """

    def __init__(
        self,
        items: int,
        num_groups: int,
        item_names: Sequence[str] | None = None,
    ) -> None:
        """Create a partition model.

        Args:
            items: Number of items to partition.
            num_groups: Number of groups.
            item_names: Optional names for items.
        """
        self._n = items
        self._num_groups = num_groups
        self._names = list(item_names) if item_names else [f"item_{i}" for i in range(items)]
        self._model = Model()
        self._group_vars: list[Variable] = []
        self._min_size = 0
        self._max_size = items
        self._build()

    def _build(self) -> None:
        """Build the partition model."""
        for i in range(self._n):
            var = self._model.int_var(f"group_{self._names[i]}", 0, self._num_groups - 1)
            self._group_vars.append(var)

    @property
    def group_vars(self) -> list[Variable]:
        """Variables indicating group assignment for each item."""
        return list(self._group_vars)

    def set_group_size(self, min_size: int = 0, max_size: int | None = None) -> None:
        """Set bounds on the size of each group.

        Args:
            min_size: Minimum items per group.
            max_size: Maximum items per group.
        """
        if max_size is None:
            max_size = self._n
        self._min_size = min_size
        self._max_size = max_size

        # For each group, count items assigned to it
        for g in range(self._num_groups):
            # Create indicator variables for this group
            indicators = []
            for i in range(self._n):
                ind_var = self._model.int_var(f"in_g{g}_{self._names[i]}", 0, 1)
                indicators.append(ind_var)
                # Link: ind_var = 1 iff group_var[i] == g
                self._model.add_binary(
                    self._group_vars[i], ind_var,
                    lambda gv, iv, group=g: (iv == 1) == (gv == group),
                    f"link_g{g}_{self._names[i]}",
                )

            if min_size > 0:
                self._model.add_sum_ge(indicators, min_size)
            if max_size < self._n:
                self._model.add_sum_le(indicators, max_size)

    def add_same_group(self, item_a: int, item_b: int) -> None:
        """Force two items into the same group.

        Args:
            item_a: First item index.
            item_b: Second item index.
        """
        self._model.add_binary(
            self._group_vars[item_a],
            self._group_vars[item_b],
            lambda a, b: a == b,
            f"same_{self._names[item_a]}_{self._names[item_b]}",
        )

    def add_different_group(self, item_a: int, item_b: int) -> None:
        """Force two items into different groups.

        Args:
            item_a: First item index.
            item_b: Second item index.
        """
        self._model.add_binary(
            self._group_vars[item_a],
            self._group_vars[item_b],
            lambda a, b: a != b,
            f"diff_{self._names[item_a]}_{self._names[item_b]}",
        )

    def solve(self) -> list[list[int]] | None:
        """Solve and return the partition as a list of groups.

        Returns a list of groups, where each group is a list of item indices.
        Returns None if infeasible.
        """
        solution = self._model.solve()
        if solution is None:
            return None

        groups: list[list[int]] = [[] for _ in range(self._num_groups)]
        for i in range(self._n):
            group_id = solution.get(f"group_{self._names[i]}", 0)
            groups[group_id].append(i)
        return groups


class BinPackingModel:
    """Model for bin packing problems.

    Assigns items with sizes to bins with capacity constraints.
    Minimizes the number of bins used or satisfies a fixed bin count.

    Example:
        bp = BinPackingModel(num_bins=3, bin_capacity=10)
        bp.add_item("book", size=3)
        bp.add_item("laptop", size=5)
        bp.add_item("phone", size=2)
        solution = bp.solve()
    """

    def __init__(self, num_bins: int, bin_capacity: int) -> None:
        """Create a bin packing model.

        Args:
            num_bins: Number of available bins.
            bin_capacity: Capacity of each bin.
        """
        self._num_bins = num_bins
        self._bin_capacity = bin_capacity
        self._model = Model()
        self._items: list[BinPackingItem] = []
        self._bins: list[Bin] = []

        for b in range(num_bins):
            self._bins.append(Bin(bin_id=b, name=f"bin_{b}", capacity=bin_capacity))

    @property
    def num_bins(self) -> int:
        return self._num_bins

    @property
    def num_items(self) -> int:
        return len(self._items)

    @property
    def items(self) -> list[BinPackingItem]:
        return list(self._items)

    def add_item(self, name: str, size: int) -> BinPackingItem:
        """Add an item to be packed.

        Args:
            name: Item name.
            size: Item size/weight.

        Returns:
            The created BinPackingItem.
        """
        item_id = len(self._items)
        bin_var = self._model.int_var(f"bin_{name}", 0, self._num_bins - 1)
        item = BinPackingItem(item_id=item_id, name=name, size=size, bin_var=bin_var)
        self._items.append(item)
        return item

    def add_same_bin(self, item_a: BinPackingItem, item_b: BinPackingItem) -> None:
        """Force two items into the same bin."""
        if item_a.bin_var and item_b.bin_var:
            self._model.add_binary(
                item_a.bin_var, item_b.bin_var,
                lambda a, b: a == b,
                f"same_bin_{item_a.name}_{item_b.name}",
            )

    def add_different_bin(self, item_a: BinPackingItem, item_b: BinPackingItem) -> None:
        """Force two items into different bins."""
        if item_a.bin_var and item_b.bin_var:
            self._model.add_binary(
                item_a.bin_var, item_b.bin_var,
                lambda a, b: a != b,
                f"diff_bin_{item_a.name}_{item_b.name}",
            )

    def build_capacity_constraints(self) -> None:
        """Add capacity constraints for all bins.

        For each bin, the sum of sizes of items assigned to it must
        not exceed the bin capacity. Uses pairwise constraints for
        items that together would exceed capacity.
        """
        # For each pair of items, if their combined size exceeds capacity,
        # they cannot be in the same bin
        for i in range(len(self._items)):
            for j in range(i + 1, len(self._items)):
                item_i = self._items[i]
                item_j = self._items[j]
                if item_i.size + item_j.size > self._bin_capacity:
                    self.add_different_bin(item_i, item_j)

        # For triples that exceed capacity
        for i in range(len(self._items)):
            for j in range(i + 1, len(self._items)):
                for k in range(j + 1, len(self._items)):
                    total = self._items[i].size + self._items[j].size + self._items[k].size
                    if total > self._bin_capacity:
                        # At least two must be in different bins
                        # Add pairwise: not all three in same bin
                        vi = self._items[i].bin_var
                        vj = self._items[j].bin_var
                        vk = self._items[k].bin_var
                        if vi and vj and vk:
                            self._model.add_binary(
                                vi, vj,
                                lambda a, b, item_k=self._items[k], cap=self._bin_capacity, si=self._items[i].size, sj=self._items[j].size: not (a == b and si + sj > cap),
                                f"cap3_{self._items[i].name}_{self._items[j].name}",
                            )

    def solve(self) -> dict[str, int] | None:
        """Solve the bin packing problem.

        Returns a mapping from item names to bin indices, or None if infeasible.
        """
        self.build_capacity_constraints()
        solution = self._model.solve()
        if solution is None:
            return None
        result = {}
        for item in self._items:
            result[item.name] = solution.get(f"bin_{item.name}", 0)
        return result

    def get_packing_summary(self, assignment: dict[str, int]) -> str:
        """Generate a human-readable packing summary."""
        bins_content: dict[int, list[tuple[str, int]]] = {}
        for item in self._items:
            bin_id = assignment.get(item.name, 0)
            bins_content.setdefault(bin_id, []).append((item.name, item.size))

        lines = ["Bin Packing Solution", "=" * 30]
        for bin_id in sorted(bins_content.keys()):
            items_in_bin = bins_content[bin_id]
            total_size = sum(size for _, size in items_in_bin)
            lines.append(
                f"  Bin {bin_id} ({total_size}/{self._bin_capacity}): "
                + ", ".join(f"{name}({size})" for name, size in items_in_bin)
            )

        return "\n".join(lines)
