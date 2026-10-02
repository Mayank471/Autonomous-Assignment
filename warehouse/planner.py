"""Prioritized planning (Cooperative A*): the multi-agent plan generation algorithm.

Agents are planned one at a time with Space-Time A*; each plan is posted to the
reservation board and every later agent treats it as a hard obstacle.  Order:
agents with a deadline first, then the longest remaining static tour first.  If
an agent cannot be planned it is moved to the front and planning restarts
(at most ``max_restarts`` times).

When agents start mid-aisle (during repairs) plain restarts can cycle, so a
second, *stationary-first* stage is tried: see :func:`_stationary_first`.

This runs once at t = 0.  During execution only the Global baseline and the
escalation steps re-run it; KR-CBS itself never does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .grid import WarehouseGrid
from .obstacles import ObstacleWindows
from .plan import Plan
from .reservation import ReservationBoard
from .stastar import DEFAULT_MAX_EXPANSIONS, Query, search


@dataclass
class PlanningOutcome:
    plans: dict[int, Plan] | None
    calls: int = 0
    expansions: int = 0
    restarts: int = 0
    failed_agent: int | None = None

    @property
    def ok(self) -> bool:
        return self.plans is not None


def default_order(grid: WarehouseGrid, queries: list[Query]) -> list[int]:
    def key(q: Query):
        has_deadline = 0 if q.deadline is not None else 1
        return (has_deadline, -grid.tour_length(q.start, q.goals), q.agent)
    return [q.agent for q in sorted(queries, key=key)]


STATIONARY_TRY_EXPANSIONS = 15_000


def _stationary_first(grid, by_agent, order, base, obstacles, out, max_expansions):
    """Stage 1 of prioritized planning: agents not yet planned are stationary
    obstacles at their current cell.

    In passes over the agents in priority order, plan every agent that can
    route around every committed path and every waiting agent.  Committed paths then never
    cross a waiting agent's cell, so nobody gets trapped by an earlier agent
    driving through it -- the usual way prioritized planning fails when
    agents start mid-aisle rather than in their docks.  Returns ``None`` if it
    gets stuck, and the caller falls back to plain restarts.
    """
    b = base.copy()
    for a in order:
        q = by_agent[a]
        b.add(a, Plan(q.t0, (q.start,)), q.t0)       # placeholder: stays put
    remaining = list(order)
    plans: dict[int, Plan] = {}
    while remaining:
        # One full pass in priority order; agents that fail wait for the next
        # pass, which only happens if this pass committed someone.
        still = []
        for a in remaining:
            q = by_agent[a]
            b.remove(a)
            res = search(grid, q, obstacles=obstacles, board=b, hard=None,
                         max_expansions=max_expansions)
            out.calls += 1
            out.expansions += res.expansions
            if res.ok:
                plans[a] = res.plan
                b.add(a, res.plan, q.t0)
            else:
                b.add(a, Plan(q.t0, (q.start,)), q.t0)
                still.append(a)
        if len(still) == len(remaining):
            return None
        remaining = still
    return plans


def prioritized(grid: WarehouseGrid, queries: list[Query], *, obstacles: ObstacleWindows,
                board: ReservationBoard | None = None, order: list[int] | None = None,
                max_restarts: int = 10,
                max_expansions: int = DEFAULT_MAX_EXPANSIONS,
                stationary_stage: bool = True) -> PlanningOutcome:
    """Plan every query against ``board`` (all other posted plans are hard).

    Plain restarts first (cheap, and enough almost always); if they fail, the
    stationary-first stage.  ``board`` may contain stale entries for the agents
    being planned; they are removed first.  The input board is not modified.
    """
    by_agent = {q.agent: q for q in queries}
    order = list(order) if order is not None else default_order(grid, queries)
    base = board.copy() if board is not None else ReservationBoard(grid.n_cells)
    for a in by_agent:
        base.remove(a)
    out = PlanningOutcome(None)
    first_order = list(order)
    for attempt in range(max_restarts + 1):
        b = base.copy()
        plans: dict[int, Plan] = {}
        failed = None
        for a in order:
            res = search(grid, by_agent[a], obstacles=obstacles, board=b, hard=None,
                         max_expansions=max_expansions)
            out.calls += 1
            out.expansions += res.expansions
            if not res.ok:
                failed = a
                break
            plans[a] = res.plan
            b.add(a, res.plan, by_agent[a].t0)
        if failed is None:
            out.plans = plans
            out.restarts = attempt
            return out
        out.failed_agent = failed
        if order[0] == failed:
            break          # cannot be planned even with top priority
        order.remove(failed)
        order.insert(0, failed)
    out.restarts = attempt
    if stationary_stage:
        plans = _stationary_first(grid, by_agent, first_order, base, obstacles, out,
                                  min(max_expansions, STATIONARY_TRY_EXPANSIONS))
        if plans is not None:
            out.plans = plans
    return out
