"""Reference DCOP instances from Weiss Ch 12.

:func:`textbook_constraint_network` is the exemplar network of Figure 12.1,
used as a regression test on the ADOPT implementation.
:func:`target_tracking_network` is the target-tracking formulation of section
3.1.2, solved by the same engine.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from .model import ConstraintNetwork, Variable

#: The cost function shared by every edge of Figure 12.1.  Read off the figure's
#: table and confirmed against the worked trace on p558, where the context
#: ``{x1=0, x2=0}`` gives ``delta(x4=0) = 2 + 2 = 4`` and ``delta(x4=1) = 0``.
TEXTBOOK_COSTS: dict[tuple[int, int], float] = {
    (0, 0): 2.0,
    (0, 1): 0.0,
    (1, 0): 0.0,
    (1, 1): 1.0,
}

#: Edges of Figure 12.1.
TEXTBOOK_EDGES: tuple[tuple[str, str], ...] = (
    ("x1", "x2"),
    ("x1", "x3"),
    ("x1", "x4"),
    ("x2", "x4"),
)


def textbook_constraint_network() -> ConstraintNetwork:
    """The exemplar constraint network of Weiss Figure 12.1.

    Four binary variables over ``{0, 1}``.  The optimum has cost 1 and is
    attained three times: at ``(0, 1, 1, 1)``, ``(1, 0, 0, 1)`` and
    ``(1, 1, 0, 0)``.
    """
    variables = ("x1", "x2", "x3", "x4")
    network = ConstraintNetwork(variables, {v: (0, 1) for v in variables})
    for u, v in TEXTBOOK_EDGES:
        network.add_binary(u, v, TEXTBOOK_COSTS)
    return network


#: Cost function shared by all four constraints of Modi et al. Fig. 2(a).
#: Fixed by the two totals the paper quotes: F(all zeros) = 4 and F(all ones) = 0.
ADOPT_PAPER_COSTS: dict[tuple[int, int], float] = {
    (0, 0): 1.0,
    (0, 1): 2.0,
    (1, 0): 2.0,
    (1, 1): 0.0,
}

#: Edges of Modi et al. Fig. 2(a).  The paper states "x1 and x3 are neighbors
#: but x1 and x4 are not", and Fig. 2(b) makes x1 the parent of x2 and x2 the
#: parent of both x3 and x4 -- which pins the constraint graph to exactly these.
ADOPT_PAPER_EDGES: tuple[tuple[str, str], ...] = (
    ("x1", "x2"),
    ("x1", "x3"),
    ("x2", "x3"),
    ("x2", "x4"),
)


def adopt_paper_network() -> ConstraintNetwork:
    """The constraint network of Modi et al. (2005), Figure 2(a).

    ADOPT was taught from the paper rather than from Weiss, and the paper uses a
    *different* example from the textbook's Figure 12.1: different edges and a
    different cost table.  Both are reproduced here so the implementation can be
    checked against either.

    The paper states the answer outright: ``F({all 0}) = 4``, ``F({all 1}) = 0``,
    and the optimum ``A* = {(x1,1), (x2,1), (x3,1), (x4,1)}``.
    """
    variables = ("x1", "x2", "x3", "x4")
    network = ConstraintNetwork(variables, {v: (0, 1) for v in variables})
    for u, v in ADOPT_PAPER_EDGES:
        network.add_binary(u, v, ADOPT_PAPER_COSTS)
    return network


def target_tracking_network(
    coverage: Mapping[Variable, Sequence[Variable]],
    *,
    reward: float = 10.0,
    energy: float = 1.0,
) -> ConstraintNetwork:
    """The target-tracking DCOP of Weiss Ch 12 section 3.1.2.

    Following the textbook's formulation, "sensors are represented by agents,
    and variables encode the different sensing modalities of each sensor.
    Constraints are usually defined among sensors that have an overlapping
    sensing range.  Each constraint relates to a specific target and represents
    how the joint choice of sensor modalities impacts on the tracking
    performance for that target."

    Each sensor's modality is the target it aims at, or ``"idle"``.  A target is
    correctly identified when two sensors aim at it together; aiming at all
    costs ``energy``, which stops sensors piling onto a single target.

    The natural way to write this is a *reward* of ``-reward`` for a jointly
    covered target, but ADOPT requires non-negative costs -- it seeds every
    child's lower bound at 0, which is only admissible when no subtree can cost
    less than that.  So the constraint is expressed equivalently as the cost of
    *failing* to cover: ``0`` when both endpoints aim at the same target and
    ``reward`` otherwise.  That is a constant ``+reward`` shift per edge, so the
    optimal assignment is identical and only the reported total differs.

    Args:
        coverage: sensor id -> the targets within its sensing range.
        reward: value of a target being jointly tracked by a pair of sensors.
        energy: cost of a sensor being active rather than idle.
    """
    sensors = tuple(coverage)
    domains = {s: ("idle", *tuple(coverage[s])) for s in sensors}
    network = ConstraintNetwork(sensors, domains)

    for sensor in sensors:
        network.add_unary(
            sensor, {m: (0.0 if m == "idle" else energy) for m in domains[sensor]}
        )

    for i, a in enumerate(sensors):
        for b in sensors[i + 1 :]:
            shared = set(coverage[a]) & set(coverage[b])
            if not shared:
                continue
            table = {
                (ma, mb): (0.0 if (ma == mb and ma in shared) else reward)
                for ma in domains[a]
                for mb in domains[b]
            }
            network.add_binary(a, b, table)

    return network


def random_network(
    n_variables: int = 6,
    domain_size: int = 3,
    edge_probability: float = 0.5,
    max_cost: float = 9.0,
    seed: int | None = None,
) -> ConstraintNetwork:
    """A random binary network, used to cross-check ADOPT against brute force."""
    rng = random.Random(seed)
    variables = tuple(f"v{i}" for i in range(n_variables))
    domains = {v: tuple(range(domain_size)) for v in variables}
    network = ConstraintNetwork(variables, domains)

    for var in variables:
        network.add_unary(
            var, {d: float(rng.randint(0, int(max_cost))) for d in domains[var]}
        )

    for i, a in enumerate(variables):
        for b in variables[i + 1 :]:
            if rng.random() > edge_probability:
                continue
            table = {
                (da, db): float(rng.randint(0, int(max_cost)))
                for da in domains[a]
                for db in domains[b]
            }
            network.add_binary(a, b, table)

    return network
