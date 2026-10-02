"""Disruptions: blockages, breakdowns and emergency tasks.

Episode schedules are sampled in advance from the run's seed, so every
strategy faces the same events.  Each event type has its own random stream, so
changing the obstacle density does not move the breakdowns or emergencies.

Obstacle density ``rho`` is the average fraction of blockable cells blocked at
any moment: blockages arrive at rate ``rho * |blockable| / mean_duration`` per
step.

The single-disruption sampler (experiment E1) picks a disruption that is sure
to affect a running plan, given the current world.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scenario import STREAM_EVENTS, Instance, rng_for

BLOCKAGE, BREAKDOWN, EMERGENCY = "blockage", "breakdown", "emergency"
KINDS = (BLOCKAGE, BREAKDOWN, EMERGENCY)


@dataclass(frozen=True)
class Event:
    kind: str
    t: int
    eid: int
    cell: int = -1
    agent: int = -1
    duration: int = 0
    pickup: int = -1
    station: int = -1


@dataclass(frozen=True)
class EventRates:
    rho: float = 0.02                     # blocked fraction of blockable cells
    blockage_duration: tuple[int, int] = (10, 30)
    breakdowns_per_agent: float = 0.3
    breakdown_duration: tuple[int, int] = (5, 20)
    emergencies: int = 2


def make_schedule(inst: Instance, horizon: int, rates: EventRates, seed: int) -> list[Event]:
    g = inst.grid
    events: list[Event] = []
    # blockages
    rng = rng_for(seed, STREAM_EVENTS, 0)
    lo, hi = rates.blockage_duration
    mean_d = (lo + hi) / 2
    lam = rates.rho * len(g.blockable) / mean_d
    n = int(rng.poisson(lam * horizon)) if rates.rho > 0 else 0
    for _ in range(n):
        events.append(Event(BLOCKAGE, int(rng.integers(1, horizon + 1)), 0,
                            cell=int(rng.choice(g.blockable)),
                            duration=int(rng.integers(lo, hi + 1))))
    # breakdowns
    rng = rng_for(seed, STREAM_EVENTS, 1)
    lo, hi = rates.breakdown_duration
    n = int(rng.poisson(rates.breakdowns_per_agent * inst.n_agents))
    for _ in range(n):
        events.append(Event(BREAKDOWN, int(rng.integers(1, max(2, int(0.8 * horizon)) + 1)), 0,
                            agent=int(rng.integers(inst.n_agents)),
                            duration=int(rng.integers(lo, hi + 1))))
    # emergencies
    rng = rng_for(seed, STREAM_EVENTS, 2)
    for _ in range(rates.emergencies):
        events.append(Event(EMERGENCY, int(rng.integers(1, max(2, int(0.6 * horizon)) + 1)), 0,
                            agent=int(rng.integers(inst.n_agents)),
                            pickup=int(rng.choice(g.pickups)),
                            station=int(rng.choice(g.stations))))
    events.sort(key=lambda e: (e.t, KINDS.index(e.kind), e.cell, e.agent, e.pickup))
    return [Event(e.kind, e.t, i, e.cell, e.agent, e.duration, e.pickup, e.station)
            for i, e in enumerate(events)]


def sample_single(world, kind: str, rng: np.random.Generator, eid: int = 0) -> Event | None:
    """A disruption at the world's current time that is sure to hit a plan.

    ``world`` is a :class:`warehouse.simulator.World`.
    """
    g = world.grid
    t = world.t
    movers = [a for a in world.active_agents() if not world.is_broken(a)
              and world.plans[a].end > t]
    if not movers:
        return None
    if kind == BLOCKAGE:
        occupied = {world.position(a) for a in range(world.n_agents)}
        candidates = []
        for a in movers:
            p = world.plans[a]
            for tau in range(t + 2, min(p.end, t + 25) + 1):
                c = p.at(tau)
                if c in g.dock_set or c in g.station_set or c in occupied:
                    continue
                candidates.append(c)
        if not candidates:
            return None
        cells = sorted(set(candidates))
        lo, hi = 10, 30
        return Event(BLOCKAGE, t, eid, cell=int(rng.choice(cells)),
                     duration=int(rng.integers(lo, hi + 1)))
    if kind == BREAKDOWN:
        moving = [a for a in movers if any(world.plans[a].at(tau) != world.position(a)
                                            for tau in range(t, min(world.plans[a].end, t + 20) + 1))]
        if not moving:
            return None
        return Event(BREAKDOWN, t, eid, agent=int(rng.choice(moving)),
                     duration=int(rng.integers(5, 21)))
    if kind == EMERGENCY:
        return Event(EMERGENCY, t, eid, agent=int(rng.choice(movers)),
                     pickup=int(rng.choice(g.pickups)), station=int(rng.choice(g.stations)))
    raise ValueError(kind)
