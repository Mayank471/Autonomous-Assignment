"""Disruptions -- the unexpected problems plan repair has to absorb.

Three kinds, matching the assignment:

``CellBlockage``
    a grid cell becomes impassable from some time onward.  Permanent, not
    transient, so agents cannot simply wait it out.
``AgentBreakdown``
    an agent stops moving.  Its cell becomes an obstacle for everyone else and
    its unfinished tasks are re-auctioned by contract net.
``EmergencyTask``
    a high-priority job appears somewhere on the floor and some agent must
    divert to it, deferring whatever it was doing.  Who diverts is settled by
    the target-tracking DCOP in :mod:`warehouse.emergency`.

Weiss Ch 11 section 6.1 frames detection as the first half of the problem --
"detecting deviations from anticipated trajectories is one challenge that is
significantly more difficult in multiagent settings" -- and
:func:`affected_agents` is that detection step: given a disruption, which
agents' committed plans are no longer executable.

The *density* parameter of the experiments is the fraction of free cells that
become blocked over a run.  Disruptions are drawn up front from a seeded RNG so
a scenario replays identically under every repair strategy, which is what makes
the strategies comparable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .grid import WarehouseGrid
from .plan import Plan


@dataclass(frozen=True)
class Disruption:
    """Base class: something that happens at a particular time."""

    time: int

    @property
    def kind(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class CellBlockage(Disruption):
    """A cell becomes permanently impassable from :attr:`time`."""

    cell: int

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"CellBlockage(t={self.time}, cell={self.cell})"


@dataclass(frozen=True)
class AgentBreakdown(Disruption):
    """An agent fails and becomes a static obstacle where it stands."""

    agent: int

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"AgentBreakdown(t={self.time}, agent={self.agent})"


@dataclass(frozen=True)
class EmergencyTask(Disruption):
    """A high-priority job that some nearby agent must divert to."""

    cell: int
    required_responders: int = 1
    #: How long servicing the emergency takes once an agent arrives.
    service_time: int = 3

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"EmergencyTask(t={self.time}, cell={self.cell}, "
            f"need={self.required_responders})"
        )


def affected_agents(
    disruption: Disruption,
    plans: Mapping[int, Plan],
    now: int,
    *,
    positions: Mapping[int, int] | None = None,
) -> set[int]:
    """Agents whose remaining plan is invalidated by ``disruption``.

    This is the monitoring step of Ch 11 section 6.1.  Only the *remaining*
    portion of a plan matters: an agent that already passed through a cell
    before it was blocked is unaffected.
    """
    hit: set[int] = set()

    if isinstance(disruption, CellBlockage):
        for agent, plan in plans.items():
            for t in range(max(now, plan.start_time), plan.end_time + 1):
                if plan.at(t) == disruption.cell:
                    hit.add(agent)
                    break
            else:
                # An agent parked on the cell for good is hit too.
                if plan.goal == disruption.cell and plan.end_time >= now:
                    hit.add(agent)

    elif isinstance(disruption, AgentBreakdown):
        stuck_cell = (positions or {}).get(disruption.agent)
        if stuck_cell is None:
            plan = plans.get(disruption.agent)
            stuck_cell = plan.at(now) if plan is not None else None
        hit.add(disruption.agent)
        if stuck_cell is not None:
            for agent, plan in plans.items():
                if agent == disruption.agent:
                    continue
                for t in range(max(now, plan.start_time), plan.end_time + 1):
                    if plan.at(t) == stuck_cell:
                        hit.add(agent)
                        break

    elif isinstance(disruption, EmergencyTask):
        # Nobody's plan is *invalid*; the question is who should divert, which
        # the emergency DCOP answers.  Seeding it with the agents already
        # heading that way keeps the candidate set small.
        for agent, plan in plans.items():
            for t in range(max(now, plan.start_time), plan.end_time + 1):
                if plan.at(t) == disruption.cell:
                    hit.add(agent)
                    break

    return hit


def generate_disruptions(
    grid: WarehouseGrid,
    plans: Mapping[int, Plan],
    *,
    density: float,
    horizon: int,
    rng: random.Random,
    breakdown_rate: float = 0.15,
    emergency_rate: float = 0.15,
    earliest: int = 5,
) -> list[Disruption]:
    """Draw a disruption schedule for one run.

    Args:
        density: fraction of free cells that become blocked over the run.  This
            is the experiments' dynamic-obstacle density axis.
        horizon: the planned makespan, used to spread events over the run.
        breakdown_rate: share of events that are agent breakdowns instead.
        emergency_rate: share of events that are emergency tasks instead.

    Blockages are biased toward cells agents actually intend to use.  A blockage
    on a cell nobody was going to visit is not a disruption at all, and sampling
    uniformly over a mostly-empty floor would produce runs where nothing
    interesting happens and the strategies are indistinguishable.
    """
    if density <= 0 or not plans:
        return []

    # Cells the fleet plans to occupy, and how contested each is.
    usage: dict[int, int] = {}
    for plan in plans.values():
        for cell in plan.cells:
            usage[cell] = usage.get(cell, 0) + 1
    if not usage:
        return []

    n_events = max(1, int(round(density * len(grid.free_cells))))
    latest = max(earliest + 1, int(horizon * 0.8))

    used_cells = list(usage)
    weights = [usage[c] for c in used_cells]
    agents = list(plans)
    disruptions: list[Disruption] = []
    blocked: set[int] = set()

    for _ in range(n_events):
        when = rng.randint(earliest, latest)
        roll = rng.random()

        if roll < breakdown_rate and agents:
            disruptions.append(AgentBreakdown(time=when, agent=rng.choice(agents)))
        elif roll < breakdown_rate + emergency_rate:
            disruptions.append(
                EmergencyTask(time=when, cell=rng.choices(used_cells, weights)[0])
            )
        else:
            for _ in range(8):  # a few tries to find a cell not already blocked
                cell = rng.choices(used_cells, weights)[0]
                if cell not in blocked:
                    blocked.add(cell)
                    disruptions.append(CellBlockage(time=when, cell=cell))
                    break

    disruptions.sort(key=lambda d: (d.time, d.kind))
    return disruptions


def summarise(disruptions: Sequence[Disruption]) -> dict[str, int]:
    """Count disruptions by kind, for the results table."""
    out: dict[str, int] = {}
    for d in disruptions:
        out[d.kind] = out.get(d.kind, 0) + 1
    return out
