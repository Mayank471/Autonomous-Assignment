"""Initial multiagent plan generation -- Weiss Ch 11 section 4.

The chapter's framing is followed literally, in two phases:

1. **Local planning prior to coordination.**  Every agent plans its own route
   as though it were alone.  The union of those routes is the initial,
   *uncoordinated* multiagent plan -- the analogue of the flawed parallel plan
   in Figure 11.4.
2. **Coordination.**  Agents are totally ordered and replan in turn against the
   accumulated reservations of everyone above them, which is prioritized
   planning (cooperative A*).  This is the concrete realisation of Algorithm
   11.2 step 5: "current superior sends down its plan to the others; other
   agents change their plans to work properly with those of the current
   superior".

Prioritized planning is incomplete -- some orderings admit no solution even
when one exists -- so failures trigger a bounded number of randomised priority
restarts, which is the standard remedy.

This module runs **once per scenario**.  Everything the assignment is actually
about happens afterwards, in :mod:`warehouse.repair`, which never calls back
into here except as the deliberately expensive ``full_replan`` baseline.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .grid import WarehouseGrid
from .obstacles import DynamicObstacles
from .plan import Plan
from .reservation import ReservationTable
from .stastar import INF, HeuristicCache, plan_route, space_time_astar
from .tasks import AgentTasks


@dataclass
class MapfSolution:
    """The outcome of initial plan generation."""

    plans: dict[int, Plan]
    reservations: ReservationTable
    priority_order: tuple[int, ...]
    independent_plans: dict[int, Plan] = field(default_factory=dict)
    restarts: int = 0
    failed_agents: tuple[int, ...] = ()
    planning_time_ms: float = 0.0

    @property
    def success(self) -> bool:
        return not self.failed_agents

    @property
    def sum_of_costs(self) -> int:
        """Total time steps over all agents -- the assignment headline metric."""
        return sum(p.duration for p in self.plans.values())

    @property
    def makespan(self) -> int:
        return max((p.end_time for p in self.plans.values()), default=0)

    @property
    def coordination_overhead(self) -> int:
        """Extra steps coordination costs over the sum of independent optima.

        The independent plans ignore other agents, so their total is a lower
        bound on any collision-free joint plan.  The gap is the price of
        coordination.
        """
        if not self.independent_plans:
            return 0
        ideal = sum(p.duration for p in self.independent_plans.values())
        return self.sum_of_costs - ideal


def find_refuge(
    grid: WarehouseGrid,
    agent: int,
    position: int,
    home: int | None,
    now: int,
    reservations: ReservationTable,
    obstacles: DynamicObstacles,
    heuristics: HeuristicCache,
    *,
    move_filter: Callable[[int, int], bool] | None = None,
    max_candidates: int = 120,
    max_expansions: int = 20_000,
) -> Plan | None:
    """Somewhere ``agent`` can go and stay indefinitely without blocking anyone.

    Needed whenever an agent cannot complete its route -- after a blockage, or
    when it loses a negotiation.  The naive answer, "stop where you are", is
    unsafe: a stationary robot occupies its cell forever, so if any other agent
    was routed through that cell later they collide.  Parking an agent is a
    planning problem in its own right, not a no-op.

    Preference order:

    1. **Its own dock.**  Exclusive to it for all time, so parking there can
       never clash with anyone.
    2. **The nearest cell it can hold forever**, found by trying candidates
       outward from its position with an infinite hold requirement -- which is
       exactly the condition Space-Time A* already checks.

    Returns ``None`` when the agent is walled in and neither is possible.  It
    deliberately does *not* fall back to "stand where you are": an agent frozen
    mid-aisle is a wall nobody was told about, and manufacturing one here was
    the source of collisions that survived every other check.  Callers must
    decide what to do instead -- usually to treat the agent as immovable and
    re-place everyone else around it.
    """
    if home is not None:
        plan = plan_route(
            grid, position, (home,), now, agent,
            reservations, obstacles, heuristics,
            move_filter=move_filter, max_expansions=max_expansions,
        )
        if plan is not None:
            return plan

    nearby = sorted(
        grid.free_cells, key=lambda c: grid.manhattan(position, c)
    )[:max_candidates]
    for cell in nearby:
        plan = space_time_astar(
            grid, position, cell, now, agent,
            reservations, obstacles, heuristics,
            hold=INF, move_filter=move_filter, max_expansions=max_expansions,
        )
        if plan is not None:
            return plan

    return None


def _dedicate_docks(
    reservations: ReservationTable, agent_tasks: Mapping[int, AgentTasks]
) -> None:
    """Mark every agent's home cell as its own dock before any planning starts."""
    for agent, tasks in agent_tasks.items():
        if tasks.home is not None:
            reservations.dedicate(tasks.home, agent)


def plan_independently(
    grid: WarehouseGrid,
    agent_tasks: Mapping[int, AgentTasks],
    obstacles: DynamicObstacles,
    heuristics: HeuristicCache,
    *,
    service_time: int = 1,
    move_filter: Callable[[int, int], bool] | None = None,
) -> dict[int, Plan]:
    """Phase 1: each agent plans alone, ignoring every other agent."""
    empty = ReservationTable()
    out: dict[int, Plan] = {}
    for agent, tasks in agent_tasks.items():
        plan = plan_route(
            grid,
            tasks.start,
            tasks.waypoints(),
            0,
            agent,
            empty,
            obstacles,
            heuristics,
            service_time=service_time,
            move_filter=move_filter,
        )
        if plan is not None:
            out[agent] = plan
    return out


def plan_initial(
    grid: WarehouseGrid,
    agent_tasks: Mapping[int, AgentTasks],
    obstacles: DynamicObstacles,
    heuristics: HeuristicCache,
    *,
    rng: random.Random | None = None,
    service_time: int = 1,
    max_restarts: int = 8,
    move_filter: Callable[[int, int], bool] | None = None,
) -> MapfSolution:
    """Generate an initial collision-free joint plan.

    Args:
        max_restarts: how many randomised priority orderings to try before
            giving up on the agents that could not be placed.
    """
    rng = rng or random.Random(0)
    started = time.perf_counter()

    independent = plan_independently(
        grid,
        agent_tasks,
        obstacles,
        heuristics,
        service_time=service_time,
        move_filter=move_filter,
    )

    # Longest-route-first is the usual prioritisation: the most constrained
    # agent picks its path while the floor is still empty.
    order = sorted(
        agent_tasks,
        key=lambda a: -independent[a].duration if a in independent else 0,
    )

    best: MapfSolution | None = None
    for attempt in range(max_restarts + 1):
        reservations = ReservationTable()
        _dedicate_docks(reservations, agent_tasks)
        plans: dict[int, Plan] = {}
        failed: list[int] = []

        for agent in order:
            tasks = agent_tasks[agent]
            plan = plan_route(
                grid,
                tasks.start,
                tasks.waypoints(),
                0,
                agent,
                reservations,
                obstacles,
                heuristics,
                service_time=service_time,
                move_filter=move_filter,
            )
            if plan is None:
                failed.append(agent)
                # One unplaceable agent must not invalidate the whole fleet, but
                # parking it needs to be safe: see ``find_refuge``.  At t=0 the
                # agent is on its own dock, which is exclusive to it, so the
                # standstill fallback here is safe by construction.
                plan = find_refuge(
                    grid, agent, tasks.start, tasks.home, 0,
                    reservations, obstacles, heuristics, move_filter=move_filter,
                ) or Plan(agent, 0, (tasks.start,))
            plans[agent] = plan
            reservations.add(plan)

        candidate = MapfSolution(
            plans=plans,
            reservations=reservations,
            priority_order=tuple(order),
            independent_plans=independent,
            restarts=attempt,
            failed_agents=tuple(failed),
        )
        if best is None or len(candidate.failed_agents) < len(best.failed_agents):
            best = candidate
        if candidate.success:
            break

        order = order[:]
        rng.shuffle(order)

    assert best is not None
    best.planning_time_ms = (time.perf_counter() - started) * 1000.0
    return best


def replan_all(
    grid: WarehouseGrid,
    agent_tasks: Mapping[int, AgentTasks],
    positions: Mapping[int, int],
    progress: Mapping[int, int],
    now: int,
    obstacles: DynamicObstacles,
    heuristics: HeuristicCache,
    *,
    stages: Mapping[int, int] | None = None,
    broken: Sequence[int] = (),
    rng: random.Random | None = None,
    service_time: int = 1,
    max_restarts: int = 4,
    move_filter: Callable[[int, int], bool] | None = None,
) -> MapfSolution:
    """Re-run prioritized planning for the whole fleet from the current state.

    This is the ``full_replan`` baseline: the thing the assignment says plan
    repair must avoid.  It is included precisely so the cost of avoiding it can
    be measured -- both in computation time and in how many agents end up with
    a different plan.

    Args:
        positions: where each agent currently stands.
        progress: how many tasks each agent has already completed.
        broken: agents that have failed and can no longer move.
    """
    rng = rng or random.Random(0)
    started = time.perf_counter()
    broken_set = set(broken)

    stages = stages or {}
    active = [a for a in agent_tasks if a not in broken_set]
    order = sorted(
        active,
        key=lambda a: -len(
            agent_tasks[a].remaining_from(progress.get(a, 0), stages.get(a, 0))
        ),
    )

    # Agents that cannot be planned *or* parked are immovable: they hold their
    # cell and everyone else must route around them.  Which agents those are is
    # only discoverable by trying, so the pass repeats with the set growing.
    # Placing them last instead -- the obvious order -- lets a mover claim the
    # cell an immovable agent is standing on, which cannot be fixed afterwards
    # and was a source of collisions that survived every downstream check.
    immovable = set(broken_set)

    best: MapfSolution | None = None
    for attempt in range(max_restarts + 1):
        reservations = ReservationTable()
        _dedicate_docks(reservations, agent_tasks)
        plans: dict[int, Plan] = {}
        failed: list[int] = []
        unplaceable: list[int] = []

        for agent in immovable:
            stuck = Plan(agent, now, (positions[agent],))
            plans[agent] = stuck
            reservations.add(stuck)

        for agent in order:
            if agent in immovable:
                continue
            waypoints = agent_tasks[agent].remaining_from(
                progress.get(agent, 0), stages.get(agent, 0)
            )
            plan = plan_route(
                grid,
                positions[agent],
                waypoints,
                now,
                agent,
                reservations,
                obstacles,
                heuristics,
                service_time=service_time,
                move_filter=move_filter,
            )
            if plan is None:
                failed.append(agent)
                plan = find_refuge(
                    grid, agent, positions[agent], agent_tasks[agent].home, now,
                    reservations, obstacles, heuristics, move_filter=move_filter,
                )
            if plan is None:
                unplaceable.append(agent)
                continue
            plans[agent] = plan
            reservations.add(plan)

        if unplaceable:
            # Freeze them and re-place everyone else around them.
            immovable |= set(unplaceable)
            if attempt < max_restarts:
                continue
            for agent in unplaceable:
                stuck = Plan(agent, now, (positions[agent],))
                plans[agent] = stuck

        candidate = MapfSolution(
            plans=plans,
            reservations=reservations,
            priority_order=tuple(order),
            restarts=attempt,
            failed_agents=tuple(failed),
        )
        if best is None or len(candidate.failed_agents) < len(best.failed_agents):
            best = candidate
        if candidate.success:
            break

        order = order[:]
        rng.shuffle(order)

    assert best is not None
    best.planning_time_ms = (time.perf_counter() - started) * 1000.0
    return best
