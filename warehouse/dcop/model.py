"""The constraint network ``N = <X, D, C>`` of Weiss Ch 12 section 2.1.

Variables are any hashable id (``"x1"`` for the textbook example, an agent id
for a repair episode).  Soft constraints are cost *functions*; hard constraints
are encoded as soft ones with a prohibitively large cost, exactly as the
textbook prescribes on p550:

    "we can always encode hard constraints using only soft constraints by using
    infinite values to penalize assignments that do not satisfy hard
    constraints"

A literal ``inf`` would make ADOPT's bound arithmetic degenerate (``inf - inf``),
so :data:`HARD` is a large finite stand-in.  Any solution whose cost reaches it
is reported as infeasible.
"""

from __future__ import annotations

from typing import Callable, Hashable, Iterable, Mapping, Sequence

Variable = Hashable
Value = Hashable

#: Finite stand-in for an infinite cost.  Large enough that no sum of genuine
#: costs can reach it, small enough that sums of several stay exact in float.
HARD = 1e6


class ConstraintNetwork:
    """A binary constraint network with unary and pairwise soft constraints.

    ADOPT and DPOP are both formulated for binary networks, and the textbook
    notes every network can be mapped to one, so binary is all that is supported
    here.  Unary costs are kept separate rather than folded into a neighbour's
    constraint because the repair formulation needs them explicitly (they carry
    the per-agent "did this agent change its plan" penalty).
    """

    __slots__ = ("_variables", "_domains", "_unary", "_binary", "_adj", "_index")

    def __init__(
        self,
        variables: Sequence[Variable],
        domains: Mapping[Variable, Sequence[Value]],
    ) -> None:
        self._variables: tuple[Variable, ...] = tuple(variables)
        self._index: dict[Variable, int] = {v: i for i, v in enumerate(self._variables)}
        if len(self._index) != len(self._variables):
            raise ValueError("duplicate variable in network")

        self._domains: dict[Variable, tuple[Value, ...]] = {}
        for var in self._variables:
            dom = tuple(domains[var])
            if not dom:
                raise ValueError(f"variable {var!r} has an empty domain")
            self._domains[var] = dom

        self._unary: dict[Variable, dict[Value, float]] = {v: {} for v in self._variables}
        self._binary: dict[tuple[Variable, Variable], Callable[[Value, Value], float]] = {}
        self._adj: dict[Variable, set[Variable]] = {v: set() for v in self._variables}

    # ------------------------------------------------------------------ shape

    @property
    def variables(self) -> tuple[Variable, ...]:
        return self._variables

    def domain(self, var: Variable) -> tuple[Value, ...]:
        return self._domains[var]

    def position(self, var: Variable) -> int:
        """Index of ``var`` in the declaration order -- used for tie-breaking."""
        return self._index[var]

    def neighbors(self, var: Variable) -> tuple[Variable, ...]:
        """Variables sharing a constraint with ``var``, in declaration order.

        This is exactly Ch 12's notion of a neighbour: "two agents are
        considered neighbors if there is at least one constraint that depends on
        variables that each controls.  Only neighboring agents can directly
        communicate with each other."
        """
        return tuple(sorted(self._adj[var], key=self._index.__getitem__))

    def edges(self) -> tuple[tuple[Variable, Variable], ...]:
        return tuple(self._binary)

    def __len__(self) -> int:
        return len(self._variables)

    # ------------------------------------------------------------- definition

    def add_unary(self, var: Variable, costs: Mapping[Value, float]) -> None:
        """Add (accumulate) a unary cost function on ``var``."""
        if var not in self._index:
            raise KeyError(f"unknown variable {var!r}")
        table = self._unary[var]
        for value, cost in costs.items():
            table[value] = table.get(value, 0.0) + float(cost)

    def add_binary(
        self,
        u: Variable,
        v: Variable,
        cost: Mapping[tuple[Value, Value], float] | Callable[[Value, Value], float],
    ) -> None:
        """Add a soft constraint between ``u`` and ``v``.

        ``cost`` is either a callable ``f(a, b)`` or a table keyed by ``(a, b)``,
        with ``a`` the value of ``u``.  Adding a second constraint to a pair
        that already has one sums them.
        """
        if u not in self._index or v not in self._index:
            raise KeyError(f"unknown variable in edge ({u!r}, {v!r})")
        if u == v:
            raise ValueError("a binary constraint needs two distinct variables")

        fn = cost if callable(cost) else _table_lookup(cost)
        existing = self._binary.pop((u, v), None)
        if existing is None:
            flipped = self._binary.pop((v, u), None)
            if flipped is not None:
                prev = flipped
                fn = _sum_fns(fn, lambda a, b, _p=prev: _p(b, a))
        else:
            fn = _sum_fns(fn, existing)

        self._binary[(u, v)] = fn
        self._adj[u].add(v)
        self._adj[v].add(u)

    # ------------------------------------------------------------- evaluation

    def unary_cost(self, var: Variable, value: Value) -> float:
        return self._unary[var].get(value, 0.0)

    def binary_cost(self, u: Variable, v: Variable, a: Value, b: Value) -> float:
        """Cost of assigning ``u = a`` and ``v = b``; ``0`` if unconstrained."""
        fn = self._binary.get((u, v))
        if fn is not None:
            return fn(a, b)
        fn = self._binary.get((v, u))
        if fn is not None:
            return fn(b, a)
        return 0.0

    def local_cost(
        self, var: Variable, value: Value, context: Mapping[Variable, Value]
    ) -> float:
        """Unary cost of ``var = value`` plus every constraint with an assigned neighbour.

        This is ADOPT's ``delta(d)`` when ``context`` holds the higher-priority
        neighbours (Weiss p558: "the sum of the values of local cost function for
        all the higher neighbors").
        """
        total = self._unary[var].get(value, 0.0)
        for other in self._adj[var]:
            if other in context:
                total += self.binary_cost(var, other, value, context[other])
        return total

    def evaluate(self, assignment: Mapping[Variable, Value]) -> float:
        """Total cost of a complete assignment."""
        total = 0.0
        for var in self._variables:
            total += self._unary[var].get(assignment[var], 0.0)
        for (u, v), fn in self._binary.items():
            total += fn(assignment[u], assignment[v])
        return total

    def is_feasible(self, assignment: Mapping[Variable, Value]) -> bool:
        """Whether the assignment avoids every hard (``HARD``-priced) constraint."""
        return self.evaluate(assignment) < HARD

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"ConstraintNetwork(|X|={len(self._variables)}, "
            f"|C|={len(self._binary)} binary, "
            f"max|D|={max(len(d) for d in self._domains.values())})"
        )


def _table_lookup(
    table: Mapping[tuple[Value, Value], float]
) -> Callable[[Value, Value], float]:
    frozen = dict(table)

    def lookup(a: Value, b: Value) -> float:
        return frozen.get((a, b), 0.0)

    return lookup


def _sum_fns(
    f: Callable[[Value, Value], float], g: Callable[[Value, Value], float]
) -> Callable[[Value, Value], float]:
    def summed(a: Value, b: Value) -> float:
        return f(a, b) + g(a, b)

    return summed
