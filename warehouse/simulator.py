"""Discrete-time execution of a warehouse run, with disruptions and repair.

One time step is: inject whatever disruptions are scheduled for now and repair
the damage, then advance every agent one cell along its current plan, then
update what each agent has finished.

Two properties of this loop matter for the results to mean anything.

*Repair is local in time as well as space.*  Only the agents drawn into an
episode are touched; everyone else keeps executing the plan they already had.
That is PGP mechanism 5 (asynchrony) -- there is no global barrier where the
fleet stops and waits for a new joint plan, which is exactly what the
``full_replan`` baseline does instead.

*The disruption schedule is drawn once, up front, from a seeded RNG.*  Every
strategy therefore faces the identical sequence of events on identical
scenarios, so differences in the numbers are attributable to the repair
strategy rather than to luck.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from .abstractplan import abstract_plan, build_abstracts
from .contractnet import reallocate_tasks
from .disruptions import (
    AgentBreakdown,
    CellBlockage,
    Disruption,
    EmergencyTask,
    summarise,
)
from .emergency import allocate_emergency, inject_emergency_task
from .grid import WarehouseGrid
from .metrics import RunMetrics
from .obstacles import DynamicObstacles
from .organization import MetaLevelOrganization
from .plan import Plan
from .planner import MapfSolution
from .repair import RepairConfig, RepairContext, repair
from .repair.candidates import safe_prefix
from .reservation import ReservationTable
from .stastar import HeuristicCache
from .tasks import AgentTasks


@dataclass
class SimulationResult:
    """Outcome of one run."""

    metrics: RunMetrics
    finish_times: dict[int, int] = field(default_factory=dict)
    history: list[dict[int, int]] = field(default_factory=list)
    disruption_log: list[tuple[int, str, int]] = field(default_factory=list)
    blocked_cells: set[int] = field(default_factory=set)

    @property
    def total_timesteps(self) -> int:
        return self.metrics.total_timesteps


def simulate(
    grid: WarehouseGrid,
    agent_tasks: dict[int, AgentTasks],
    solution: MapfSolution,
    disruptions: Sequence[Disruption],
    *,
    strategy: str,
    config: RepairConfig | None = None,
    obstacles: DynamicObstacles | None = None,
    move_filter: Callable[[int, int], bool] | None = None,
    seed: int = 0,
    max_steps: int | None = None,
    record_history: bool = False,
    verbose: bool = False,
) -> SimulationResult:
    """Execute the fleet's plans, injecting ``disruptions`` and repairing.

    Args:
        solution: the initial joint plan.  Its reservation table is *copied*,
            so one precomputed solution can be replayed under every strategy.
        max_steps: safety bound.  Defaults to a generous multiple of the
            planned makespan.
    """
    config = config or RepairConfig()
    rng = random.Random(seed)

    # Work on copies so the caller's precomputed solution stays reusable.
    reservations = solution.reservations.copy()
    obstacles = obstacles.copy() if obstacles is not None else DynamicObstacles()
    heuristics = HeuristicCache(grid, obstacles, move_filter=move_filter)
    tasks = {a: AgentTasks(a, t.start, list(t.tasks), t.home) for a, t in agent_tasks.items()}

    mlo = MetaLevelOrganization.from_priority(grid, solution.priority_order)
    mlo.assign_zone_coordinators(build_abstracts(grid, solution.plans))

    agents = list(tasks)
    positions = {a: reservations.plan_of(a).at(0) for a in agents if reservations.plan_of(a)}
    progress = {a: 0 for a in agents}
    stages = {a: 0 for a in agents}
    broken: set[int] = set()
    finished: dict[int, int] = {}

    metrics = RunMetrics(
        strategy=strategy,
        n_agents=len(agents),
        seed=seed,
        beta=config.beta,
        tau=config.tau,
        k_max=config.k_max,
        epsilon=config.epsilon,
        social_laws=move_filter is not None,
        tasks_total=sum(len(t.tasks) for t in tasks.values()),
        n_disruptions=len(disruptions),
        disruptions_by_kind=summarise(disruptions),
        initial_plan_ms=solution.planning_time_ms,
        initial_sum_of_costs=solution.sum_of_costs,
    )

    horizon = solution.makespan
    max_steps = max_steps if max_steps is not None else max(200, horizon * 3 + 100)

    ctx = RepairContext(
        grid=grid,
        obstacles=obstacles,
        heuristics=heuristics,
        reservations=reservations,
        agent_tasks=tasks,
        positions=positions,
        progress=progress,
        stages=stages,
        now=0,
        config=config,
        mlo=mlo,
        broken=broken,
        move_filter=move_filter,
    )

    schedule: dict[int, list[Disruption]] = {}
    for disruption in disruptions:
        schedule.setdefault(disruption.time, []).append(disruption)

    result = SimulationResult(metrics=metrics)
    emergency_count = 0
    end_of_run = max_steps

    for now in range(max_steps + 1):
        ctx.now = now

        # ---------------------------------------------------------- disrupt
        for disruption in schedule.get(now, ()):
            applied = _apply(
                disruption, ctx, rng, emergency_index=emergency_count, metrics=metrics
            )
            if applied is None:
                continue
            if isinstance(disruption, EmergencyTask):
                emergency_count += 1
            result.disruption_log.append((now, disruption.kind, applied))

            outcome = repair(strategy, disruption, ctx)
            trimmed = enforce_no_conflicts(ctx)
            metrics.safety_truncations += trimmed
            if outcome is not None:
                metrics.record_repair(outcome)
                if verbose:
                    print(
                        f"  t={now:4d} {disruption.kind:16s} "
                        f"involved={outcome.n_involved} changed={outcome.n_changed} "
                        f"{outcome.wall_ms:.1f}ms"
                    )

        # ------------------------------------------------------------- move
        everyone_done = True
        anyone_moving = False
        for agent in agents:
            if agent in broken or agent in finished:
                continue
            plan = reservations.plan_of(agent)
            if plan is None:
                continue
            positions[agent] = plan.at(now + 1)
            everyone_done = False
            if plan.end_time > now + 1:
                anyone_moving = True

        if record_history:
            result.history.append(dict(positions))

        # --------------------------------------------------------- progress
        for agent in agents:
            if agent in broken or agent in finished:
                continue
            _advance_progress(agent, tasks, positions, progress, stages)
            if _is_finished(agent, tasks, positions, progress):
                finished[agent] = now + 1

        if everyone_done or len(finished) + len(broken) >= len(agents):
            end_of_run = now + 1
            break

        # Deadlock: every unfinished agent has run its plan out and is parked
        # short of its dock, so no further progress is possible.  Stopping here
        # rather than idling to ``max_steps`` keeps the cost of a failed repair
        # honest instead of dominated by the safety bound.
        if not anyone_moving:
            end_of_run = now + 1
            break

    # ------------------------------------------------------------- tally up
    # An agent that never finishes is charged to the end of the run: it did not
    # accomplish its tasks, and pretending its clock stopped early would reward
    # a strategy for stranding agents.
    stranded = [a for a in agents if a not in finished and a not in broken]
    for agent in agents:
        if agent not in finished:
            finished[agent] = end_of_run

    metrics.total_timesteps = sum(finished.values())
    metrics.makespan = max(finished.values(), default=0)
    metrics.agents_finished = len(agents) - len(stranded) - len(broken)
    metrics.agents_stranded = len(stranded)
    metrics.tasks_completed = sum(progress[a] for a in agents)
    metrics.collisions_detected = len(reservations.find_conflicts())

    result.finish_times = finished
    result.blocked_cells = obstacles.all_cells()
    return result


# ---------------------------------------------------------------- internals


def enforce_no_conflicts(ctx: RepairContext, max_passes: int = 12) -> int:
    """Last line of defence: guarantee the joint plan is collision-free.

    Every strategy is meant to produce a safe plan, and each checks its own
    output.  The path that can still fail is parking an agent that cannot
    finish: if its dock is unreachable and every nearby cell is spoken for, it
    ends up standing where it is, and a stationary robot is a wall that other
    agents' committed plans may already run through.

    Conflicts are resolved by rebuilding the joint plan, trimming each agent to
    the longest prefix it can execute and then *hold* given everyone placed
    before it -- a queue forming behind a stalled robot.

    The subtlety is ordering.  An agent that cannot move is an obstacle, not a
    participant, so it must be placed first and the movers routed around it.
    Two earlier attempts at this failed, and both failures are instructive:

    * Ordering purely by authority let a senior *mover* claim the cell a stalled
      agent was standing on -- unfixable on that pass.
    * Classifying "immovable" freshly each pass oscillated.  An agent trimmed to
      a standstill during pass *k* was still classed as a mover during pass
      *k+1* if its plan had meanwhile been rebuilt, so two agents could take
      turns displacing each other indefinitely.

    The fix is a **monotone** frozen set: once an agent is frozen it stays
    frozen, and any agent that cannot be placed joins it.  That makes the loop
    terminate for a countable reason -- every pass either resolves all conflicts
    or grows the frozen set by at least one, and the set is bounded by the fleet
    size.

    Returns the number of agents that had to be trimmed.
    """
    frozen: set[int] = set(ctx.broken)
    trimmed_total = 0

    for _ in range(max_passes):
        if not ctx.reservations.find_conflicts():
            break

        plans = {
            a: p
            for a, p in ((a, ctx.reservations.plan_of(a)) for a in ctx.agent_tasks)
            if p is not None
        }
        if not plans:
            break

        # Anything already reduced to standing still is immovable in fact.
        for agent, plan in plans.items():
            if len(plan.suffix_from(ctx.now).cells) == 1:
                frozen.add(agent)

        rebuilt = ReservationTable()
        for tasks in ctx.agent_tasks.values():
            if tasks.home is not None:
                rebuilt.dedicate(tasks.home, tasks.agent)

        # Frozen agents first: they are obstacles the others must route around.
        for agent in sorted(frozen & plans.keys(), key=ctx.mlo.rank):
            rebuilt.add(Plan(agent, ctx.now, (ctx.positions[agent],)))

        trimmed = 0
        movers = sorted(plans.keys() - frozen, key=ctx.mlo.rank)
        for agent in movers:
            suffix = plans[agent].suffix_from(ctx.now)
            safe = safe_prefix(suffix, ctx, rebuilt)
            if safe is None:
                # Cannot be accommodated at all.  Freeze it permanently; the
                # next pass places it first and the others give way.
                frozen.add(agent)
                safe = Plan(agent, ctx.now, (ctx.positions[agent],))
            if safe.cells != suffix.cells:
                trimmed += 1
            rebuilt.add(safe)

        ctx.reservations.clear()
        for tasks in ctx.agent_tasks.values():
            if tasks.home is not None:
                ctx.reservations.dedicate(tasks.home, tasks.agent)
        for plan in rebuilt.plans():
            ctx.reservations.add(plan)
        ctx.reservations.rebuild_last_use()
        trimmed_total += trimmed

    return trimmed_total


def _apply(
    disruption: Disruption,
    ctx: RepairContext,
    rng: random.Random,
    *,
    emergency_index: int,
    metrics: RunMetrics,
) -> int | None:
    """Make the disruption real.  Returns a describing id, or ``None`` if void.

    A disruption can turn out to be void -- a cell blocked where an agent is
    already standing, or a breakdown of an agent that already failed.  Those
    are skipped rather than forced, and are not counted as repairs.
    """
    if isinstance(disruption, CellBlockage):
        occupied = {p for a, p in ctx.positions.items() if a not in ctx.broken}
        if disruption.cell in occupied:
            return None  # cannot wall in an agent that is standing there
        if ctx.reservations.dock_owner(disruption.cell) is not None:
            return None  # a dock is not a travel cell
        if not ctx.obstacles.block(disruption.cell, ctx.now):
            return None
        return disruption.cell

    if isinstance(disruption, AgentBreakdown):
        agent = disruption.agent
        if agent in ctx.broken:
            return None
        cell = ctx.positions.get(agent)
        if cell is None:
            return None

        ctx.broken.add(agent)
        ctx.obstacles.block(cell, ctx.now)
        # The failed agent holds its cell for good.
        ctx.reservations.remove(agent)
        ctx.reservations.add(Plan(agent, ctx.now, (cell,)))

        # Its unfinished work is re-auctioned (Ch 11 section 3.3 contract net).
        outcome = reallocate_tasks(
            ctx.grid,
            agent,
            ctx.agent_tasks,
            ctx.progress,
            ctx.positions,
            ctx.active_agents(),
        )
        metrics.repair_messages.append(outcome.messages)
        if outcome.unawarded:
            metrics.notes.append(
                f"t={ctx.now}: {len(outcome.unawarded)} tasks of agent {agent} went unclaimed"
            )
        return cell

    if isinstance(disruption, EmergencyTask):
        active = ctx.active_agents()
        next_waypoints = {
            a: _next_waypoint(a, ctx.agent_tasks, ctx.progress, ctx.stages)
            for a in active
        }
        allocation = allocate_emergency(
            disruption,
            ctx.grid,
            ctx.positions,
            next_waypoints,
            active,
            epsilon=ctx.config.epsilon,
        )
        if not allocation.responders:
            return None
        for offset, responder in enumerate(allocation.responders):
            inject_emergency_task(
                disruption,
                responder,
                ctx.agent_tasks,
                ctx.progress,
                emergency_index * 100 + offset,
            )
            # An emergency is extra work the fleet did not originally have, so
            # it joins the denominator rather than inflating the completion rate.
            metrics.tasks_total += 1
        metrics.repair_messages.append(allocation.messages)
        return disruption.cell

    return None


def _next_waypoint(
    agent: int,
    tasks: Mapping[int, AgentTasks],
    progress: Mapping[int, int],
    stages: Mapping[int, int],
) -> int | None:
    remaining = tasks[agent].remaining_from(
        progress.get(agent, 0), stages.get(agent, 0)
    )
    return remaining[0] if remaining else None


def _advance_progress(
    agent: int,
    tasks: Mapping[int, AgentTasks],
    positions: Mapping[int, int],
    progress: dict[int, int],
    stages: dict[int, int],
) -> None:
    """Mark a pickup or delivery as done when the agent stands on it."""
    queue = tasks[agent].tasks
    index = progress[agent]
    if index >= len(queue):
        return

    here = positions[agent]
    task = queue[index]
    if stages[agent] == 0:
        if here == task.pickup:
            stages[agent] = 1
            # Pickup and delivery can be the same cell (an emergency job), in
            # which case arriving completes both.
            if task.delivery == task.pickup:
                progress[agent] = index + 1
                stages[agent] = 0
    elif here == task.delivery:
        progress[agent] = index + 1
        stages[agent] = 0


def _is_finished(
    agent: int,
    tasks: Mapping[int, AgentTasks],
    positions: Mapping[int, int],
    progress: Mapping[int, int],
) -> bool:
    return (
        progress[agent] >= len(tasks[agent].tasks)
        and positions[agent] == tasks[agent].home
    )
