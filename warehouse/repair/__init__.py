"""Plan repair -- responding to disruption without replanning from scratch.

The layer is organised around partial global planning (Weiss Ch 11 section 6.3)
and solved as a distributed constraint optimization problem (Ch 12), following
the bridge the textbook draws on p497:

    "this view of multiagent planning is compatible with a distributed
    constraint satisfaction formulation, where the variables are the agents'
    plans, and the constraints enforce that the plans dovetail together
    suitably."

So a repair episode *is* a DCOP.  Each involved agent controls one variable
whose domain is the handful of plans it could switch to; binary constraints
forbid pairs of plans that collide in space-time; and the requirement to
"minimise the number of agents whose original plans are altered" is a term in
the objective function rather than an afterthought.

Four strategies are provided and compared head to head:

``selfish``
    only the disrupted agent replans.  Lower bound on agents changed.
``greedy_pgp``
    PGP mechanism 3: agents hill-climb over their partial global plan.
``adopt``
    the same DCOP solved optimally by ADOPT.
``full_replan``
    re-run prioritized planning for the whole fleet -- the baseline the
    assignment says to avoid, kept so the cost of avoiding it is measurable.
"""

from .episode import (
    RepairConfig,
    RepairContext,
    RepairEpisode,
    RepairOutcome,
    build_episode,
)
from .candidates import Candidate, generate_candidates
from .solvers import SOLVERS, repair

__all__ = [
    "RepairConfig",
    "RepairContext",
    "RepairEpisode",
    "RepairOutcome",
    "build_episode",
    "Candidate",
    "generate_candidates",
    "SOLVERS",
    "repair",
]
