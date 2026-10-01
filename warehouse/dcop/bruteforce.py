"""Centralised exhaustive DCOP solver -- the oracle ADOPT is checked against.

Only ever used in tests and for small repair episodes where confirming
optimality is cheap.  It has no place in the distributed story; its purpose is
to make "ADOPT found the optimum" a checkable claim rather than an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

from .model import ConstraintNetwork, Value, Variable


@dataclass(frozen=True)
class BruteForceResult:
    assignment: dict[Variable, Value]
    cost: float
    n_evaluated: int
    optima: tuple[dict[Variable, Value], ...]


def solve_bruteforce(
    network: ConstraintNetwork, max_combinations: int = 2_000_000
) -> BruteForceResult:
    """Minimise the network by exhaustive enumeration.

    Raises:
        ValueError: if the search space exceeds ``max_combinations``.
    """
    variables = network.variables
    domains = [network.domain(v) for v in variables]

    total = 1
    for dom in domains:
        total *= len(dom)
        if total > max_combinations:
            raise ValueError(
                f"search space of {total}+ combinations exceeds the "
                f"{max_combinations} limit -- too large to enumerate"
            )

    best_cost = float("inf")
    optima: list[dict[Variable, Value]] = []
    evaluated = 0

    for combo in product(*domains):
        assignment = dict(zip(variables, combo))
        cost = network.evaluate(assignment)
        evaluated += 1
        if cost < best_cost - 1e-9:
            best_cost = cost
            optima = [assignment]
        elif abs(cost - best_cost) <= 1e-9:
            optima.append(assignment)

    return BruteForceResult(
        assignment=optima[0] if optima else {},
        cost=best_cost,
        n_evaluated=evaluated,
        optima=tuple(optima),
    )
