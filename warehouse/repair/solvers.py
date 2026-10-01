"""The four repair strategies, compared head to head.

All of them consume the same :class:`RepairEpisode` -- same neighbourhood, same
candidate domains, same cost model -- so differences in the results are
differences in the negotiation, not in the problem each was handed.

``selfish``
    Only the agents actually hit replan, against everyone else's plans held
    fixed.  Nobody yields.  This is the lower bound on agents changed, and the
    point of including it is to show what that bound costs: an agent with no
    way through advances as far as it can and stops, which strands it and can
    stall others behind it.

``greedy_pgp``
    PGP mechanism 3, as the textbook describes it: each agent forms a partial
    global plan by combining the abstract plans it has received with its own,
    then "search[es] through modifications to the partial global plan to find a
    better joint plan", with the search "done heuristically and in a greedy
    hill-climbing fashion, since finding optimal solutions would take much more
    time and such solutions will likely be made obsolete quickly due to further
    dynamics".  Agents take turns in the MLO's authority order, each adopting
    its best response to what it believes the others will do.

``adopt``
    The same problem posed as a DCOP and solved optimally by ADOPT.

``full_replan``
    Re-run prioritized planning for the entire fleet -- the baseline the
    assignment says to avoid, kept so the cost of avoiding it is measurable.

**Feasibility is never assumed.**  A negotiated assignment can still contain a
collision: when every joint choice in the domains collides, an optimal solver
returns the least-bad one, not a legal one.  Installing that would drive two
robots into the same cell.  So every strategy's result is checked, and anything
that does not check out falls back to :func:`sequential_local_repair` -- still
local, still bounded to the episode's agents, but collision-free by
construction because it plans the participants one at a time in authority
order.  That fallback is the middle rung of the ladder described in Ch 11
section 6.2, and how often it is needed is reported rather than hidden.
"""

from __future__ import annotations

import time
from typing import Mapping

from ..dcop.adopt import solve_adopt
from ..dcop.model import ConstraintNetwork
from ..disruptions import Disruption
from ..plan import Plan
from ..planner import replan_all
from ..stastar import plan_route
from .candidates import Candidate, retreat_plan
from .cbs import solve_cbs, to_candidates
from .episode import (
    RepairContext,
    RepairEpisode,
    RepairOutcome,
    assignment_conflicts,
    batch_triggers,
    build_episode,
    commit,
    conflict_pairs,
    detect_triggers,
    evaluate,
    unary_costs,
)


def _as_candidates(
    episode: RepairEpisode, assignment: Mapping[int, int]
) -> dict[int, Candidate]:
    return {a: episode.candidates[a][i] for a, i in assignment.items()}


def _current_suffix(agent: int, ctx: RepairContext) -> Plan:
    """The agent's remaining plan, or a standstill if it has none."""
    current = ctx.reservations.plan_of(agent)
    if current is not None:
        return current.suffix_from(ctx.now)
    return Plan(agent, ctx.now, (ctx.positions[agent],))


# ------------------------------------------------------- sequential fallback


def sequential_local_repair(
    episode: RepairEpisode, ctx: RepairContext
) -> tuple[dict[int, Candidate], int]:
    """Prioritized replanning restricted to the episode's participants.

    Agents are planned one at a time in the MLO's authority order against a
    table that already holds the uninvolved fleet plus every participant
    settled so far.  That makes the result collision-free by construction, at
    the price of changing nearly every participant -- which is why it is a
    fallback and not the main strategy.

    An agent with no feasible route retreats instead
    (:func:`~warehouse.repair.candidates.retreat_plan`), which finds it somewhere
    it can legally park rather than freezing it mid-aisle.
    """
    table = episode.baseline.copy()
    chosen: dict[int, Candidate] = {}
    searches = 0

    for agent in episode.order:
        searches += 1
        plan = plan_route(
            ctx.grid,
            ctx.positions[agent],
            ctx.agent_tasks[agent].remaining_from(
                ctx.progress.get(agent, 0), ctx.stages.get(agent, 0)
            ),
            ctx.now,
            agent,
            table,
            ctx.obstacles,
            ctx.heuristics,
            service_time=ctx.config.service_time,
            move_filter=ctx.move_filter,
            max_expansions=ctx.config.max_expansions,
        )

        abandons = False
        if plan is None:
            plan, abandons = retreat_plan(agent, ctx, table)

        original = ctx.reservations.plan_of(agent)
        unchanged = (
            original is not None and original.suffix_from(ctx.now).cells == plan.cells
        )
        chosen[agent] = Candidate(
            kind="keep" if unchanged else "sequential",
            plan=plan,
            changed=not unchanged,
            abandons=abandons,
        )
        table.add(plan)

    return chosen, searches


def _finalise(
    episode: RepairEpisode,
    chosen: dict[int, Candidate],
    ctx: RepairContext,
    *,
    solver: str,
    messages: int = 0,
    cycles: int = 0,
    started: float = 0.0,
    searches: int = 0,
    cost: float = 0.0,
    timed_out: bool = False,
) -> RepairOutcome:
    """Check the result is actually collision-free, fall back if not, commit."""
    note = ""
    clashes = assignment_conflicts(episode, chosen)
    if clashes:
        chosen, extra = sequential_local_repair(episode, ctx)
        searches += extra
        note = (
            f"negotiated plan collided on {len(clashes)} pair(s); "
            f"fell back to sequential local repair"
        )

    wall = (time.perf_counter() - started) * 1000.0
    outcome = commit(
        episode,
        chosen,
        ctx,
        solver=solver,
        messages=messages,
        cycles=cycles,
        wall_ms=wall,
        searches=searches,
        cost=cost,
    )
    outcome.timed_out = timed_out
    if note:
        outcome.note = note
    return outcome


# --------------------------------------------------------------------- selfish


def solve_selfish(episode: RepairEpisode, ctx: RepairContext) -> RepairOutcome:
    """Only the disrupted agents replan; every other plan is held fixed.

    Each trigger replans against the *full* reservation table minus itself, so
    no other agent is asked to move.  That is collision-free by construction --
    and when there is no way through, the agent stalls rather than being let
    past, which is precisely the cost of refusing to negotiate.
    """
    started = time.perf_counter()
    table = ctx.reservations.copy()
    chosen: dict[int, Candidate] = {}
    searches = 0

    for agent in episode.order:
        if agent not in episode.triggers:
            chosen[agent] = Candidate(
                kind="keep", plan=_current_suffix(agent, ctx), changed=False
            )
            continue

        table.remove(agent)
        table.rebuild_last_use()
        searches += 1
        plan = plan_route(
            ctx.grid,
            ctx.positions[agent],
            ctx.agent_tasks[agent].remaining_from(
                ctx.progress.get(agent, 0), ctx.stages.get(agent, 0)
            ),
            ctx.now,
            agent,
            table,
            ctx.obstacles,
            ctx.heuristics,
            service_time=ctx.config.service_time,
            move_filter=ctx.move_filter,
            max_expansions=ctx.config.max_expansions,
        )

        abandons = False
        if plan is None:
            plan, abandons = retreat_plan(agent, ctx, table)

        chosen[agent] = Candidate(
            kind="selfish", plan=plan, changed=True, abandons=abandons
        )
        table.add(plan)

    return _finalise(
        episode,
        chosen,
        ctx,
        solver="selfish",
        started=started,
        searches=searches,
    )


# ------------------------------------------------------------------ greedy PGP


def solve_greedy_pgp(episode: RepairEpisode, ctx: RepairContext) -> RepairOutcome:
    """Greedy hill-climbing over the partial global plan (PGP mechanism 3)."""
    started = time.perf_counter()
    config = ctx.config
    unary = unary_costs(episode, config)
    pairs = conflict_pairs(episode)
    clash_cost = config.conflict_cost

    neighbours: dict[int, list[int]] = {a: [] for a in episode.involved}
    for a, b in pairs:
        neighbours[a].append(b)
        neighbours[b].append(a)

    def pair_cost(a: int, ia: int, b: int, ib: int) -> float:
        table = pairs.get((a, b))
        if table is not None:
            return clash_cost if table[(ia, ib)] else 0.0
        table = pairs.get((b, a))
        if table is not None:
            return clash_cost if table[(ib, ia)] else 0.0
        return 0.0

    def local_cost(agent: int, index: int, current: Mapping[int, int]) -> float:
        total = unary[agent][index]
        for other in neighbours[agent]:
            total += pair_cost(agent, index, other, current[other])
        return total

    # Start from "everyone keeps what they have" -- the least disturbing point,
    # and the one the objective already prefers.
    assignment: dict[int, int] = {
        agent: next((i for i, c in enumerate(candidates) if not c.changed), 0)
        for agent, candidates in episode.candidates.items()
    }

    messages = 0
    rounds = 0
    for _ in range(config.greedy_rounds):
        rounds += 1
        improved = False
        # Authority order: superiors settle first and inferiors adapt, which is
        # Algorithm 11.2 step 5 applied to a single episode.
        for agent in episode.order:
            options = range(len(episode.candidates[agent]))
            best = min(options, key=lambda i: (local_cost(agent, i, assignment), i))
            if best != assignment[agent]:
                assignment[agent] = best
                improved = True
            messages += len(neighbours[agent])  # one VALUE per neighbour per turn
        if not improved:
            break

    return _finalise(
        episode,
        _as_candidates(episode, assignment),
        ctx,
        solver="greedy_pgp",
        messages=messages,
        cycles=rounds,
        started=started,
        cost=evaluate(episode, assignment, config, unary=unary, pairs=pairs),
    )


# ----------------------------------------------------------------------- ADOPT


def build_network(episode: RepairEpisode, ctx: RepairContext) -> ConstraintNetwork:
    """Pose the repair episode as the DCOP of Weiss Ch 12.

    Variables are the agents' plans; binary constraints enforce that the chosen
    plans dovetail; unary costs carry the delay and the penalty for changing at
    all.  Collisions are priced at ``conflict_cost`` -- the textbook's recipe
    for encoding a hard constraint as a soft one (p550).
    """
    variables = list(episode.involved)
    domains = {a: tuple(range(len(episode.candidates[a]))) for a in variables}
    network = ConstraintNetwork(variables, domains)

    for agent, table in unary_costs(episode, ctx.config).items():
        network.add_unary(agent, table)

    clash_cost = ctx.config.conflict_cost
    for (a, b), table in conflict_pairs(episode).items():
        network.add_binary(
            a, b, {key: (clash_cost if clash else 0.0) for key, clash in table.items()}
        )

    return network


def solve_adopt_repair(episode: RepairEpisode, ctx: RepairContext) -> RepairOutcome:
    """Solve the repair DCOP optimally with ADOPT."""
    started = time.perf_counter()
    network = build_network(episode, ctx)

    result = solve_adopt(
        network,
        epsilon=ctx.config.epsilon,
        max_cycles=ctx.config.max_cycles,
        validate=False,  # costs are non-negative by construction
    )
    assignment = {a: result.assignment[a] for a in episode.involved}

    return _finalise(
        episode,
        _as_candidates(episode, assignment),
        ctx,
        solver="adopt",
        messages=result.messages,
        cycles=result.cycles,
        started=started,
        cost=result.cost,
        timed_out=not result.terminated,
    )


# ------------------------------------------------------------------------ CBS


def solve_cbs_repair(episode: RepairEpisode, ctx: RepairContext) -> RepairOutcome:
    """Repair by Conflict-Based Search over the neighbourhood.

    Outside the taught syllabus; see :mod:`warehouse.repair.cbs` for why it was
    added and what it fixes.
    """
    started = time.perf_counter()
    result = solve_cbs(episode, ctx)

    if result.solved:
        chosen = to_candidates(result, episode, ctx)
        searches = result.low_level_calls
        note = ""
    else:
        # Out of nodes.  The best node found may still contain a conflict, and
        # the CBS candidates carry no space-time signature for ``_finalise`` to
        # check, so committing it would smuggle a collision past every guard.
        # Fall back to the repair that is safe by construction.
        chosen, extra = sequential_local_repair(episode, ctx)
        searches = result.low_level_calls + extra
        note = "cbs exhausted its node budget; fell back to sequential local repair"

    outcome = _finalise(
        episode,
        chosen,
        ctx,
        solver="cbs",
        started=started,
        searches=searches,
        cycles=result.expansions,
        cost=result.cost,
    )
    outcome.timed_out = not result.solved
    if note:
        outcome.note = note
    return outcome


# ---------------------------------------------------------------- full replan


def solve_full_replan(disruption: Disruption, ctx: RepairContext) -> RepairOutcome:
    """Re-run prioritized planning for the whole fleet -- the baseline."""
    started = time.perf_counter()
    before = {a: ctx.reservations.plan_of(a) for a in ctx.agent_tasks}

    solution = replan_all(
        ctx.grid,
        ctx.agent_tasks,
        ctx.positions,
        ctx.progress,
        ctx.now,
        ctx.obstacles,
        ctx.heuristics,
        stages=ctx.stages,
        broken=tuple(ctx.broken),
        service_time=ctx.config.service_time,
        move_filter=ctx.move_filter,
    )

    changed = [
        agent
        for agent, plan in solution.plans.items()
        if before.get(agent) is None
        or before[agent].suffix_from(ctx.now).cells != plan.cells
    ]

    ctx.reservations.clear()
    for tasks in ctx.agent_tasks.values():
        if tasks.home is not None:
            ctx.reservations.dedicate(tasks.home, tasks.agent)
    for plan in solution.plans.values():
        ctx.reservations.add(plan)

    wall = (time.perf_counter() - started) * 1000.0
    return RepairOutcome(
        plans=dict(solution.plans),
        involved=tuple(solution.plans),
        changed=tuple(changed),
        reported=tuple(changed),
        solver="full_replan",
        success=solution.success,
        wall_ms=wall,
        searches=len(solution.plans),
        note="" if solution.success else "replan left agents unplaced",
        episode_sizes=(len(solution.plans),),
    )


# --------------------------------------------------------------------- facade


SOLVERS: dict[str, str] = {
    "selfish": "disrupted agent replans alone, no negotiation",
    "greedy_pgp": "PGP greedy hill-climbing over the partial global plan",
    "adopt": "repair DCOP solved optimally by ADOPT",
    "cbs": "conflict-based search over the neighbourhood (beyond the syllabus)",
    "full_replan": "prioritized replanning of the whole fleet",
}

#: Strategies that search the joint plan space rather than a fixed candidate
#: set, and so can usefully be retried on a wider neighbourhood when they fail.
ESCALATING = frozenset({"cbs"})


def _merge(outcomes: list[RepairOutcome]) -> RepairOutcome:
    """Fold several negotiation groups for one disruption into a single result."""
    if len(outcomes) == 1:
        return outcomes[0]

    involved: list[int] = []
    changed: list[int] = []
    reported: list[int] = []
    abandoned: list[int] = []
    plans: dict = {}
    for outcome in outcomes:
        plans.update(outcome.plans)
        involved.extend(outcome.involved)
        changed.extend(outcome.changed)
        reported.extend(outcome.reported)
        abandoned.extend(outcome.abandoned)

    notes = [o.note for o in outcomes if o.note]
    return RepairOutcome(
        plans=plans,
        involved=tuple(dict.fromkeys(involved)),
        changed=tuple(dict.fromkeys(changed)),
        reported=tuple(dict.fromkeys(reported)),
        solver=outcomes[0].solver,
        success=all(o.success for o in outcomes),
        cost=sum(o.cost for o in outcomes),
        messages=sum(o.messages for o in outcomes),
        cycles=max(o.cycles for o in outcomes),
        wall_ms=sum(o.wall_ms for o in outcomes),
        hops=max(o.hops for o in outcomes),
        abandoned=tuple(dict.fromkeys(abandoned)),
        searches=sum(o.searches for o in outcomes),
        timed_out=any(o.timed_out for o in outcomes),
        note="; ".join(dict.fromkeys(notes)),
        episode_sizes=tuple(
            size for o in outcomes for size in (o.episode_sizes or (o.n_involved,))
        ),
    )


def repair(
    strategy: str,
    disruption: Disruption,
    ctx: RepairContext,
    *,
    seeds: set[int] | None = None,
) -> RepairOutcome | None:
    """Repair the damage from one disruption under ``strategy``.

    A disruption that invalidates more plans than a single negotiation can
    handle is repaired in successive groups (see
    :func:`~warehouse.repair.episode.batch_triggers`); the results are merged so
    the caller still sees one outcome per disruption.

    Returns ``None`` when the disruption needs no repair at all -- it hit a cell
    nobody was going to use.
    """
    if strategy not in SOLVERS:
        raise KeyError(
            f"unknown repair strategy {strategy!r}; expected one of {sorted(SOLVERS)}"
        )

    if strategy == "full_replan":
        return solve_full_replan(disruption, ctx)

    triggers = set(seeds) if seeds is not None else detect_triggers(disruption, ctx)
    triggers -= ctx.broken
    if not triggers:
        return None

    solve = {
        "selfish": solve_selfish,
        "greedy_pgp": solve_greedy_pgp,
        "adopt": solve_adopt_repair,
        "cbs": solve_cbs_repair,
    }[strategy]

    outcomes: list[RepairOutcome] = []
    for group in batch_triggers(triggers, ctx):
        # Rebuilt per group, so each sees the commits the previous ones made.
        episode = build_episode(disruption, ctx, seeds=group)
        if episode is None:
            continue
        outcome = solve(episode, ctx)

        # Adaptive escalation.  A failure here means the neighbourhood was too
        # narrow to contain a solution, not that none exists -- so widen the
        # ring and try again rather than leaving agents stranded.  Only worth
        # doing for a solver that searches the real plan space; retrying a fixed
        # candidate set on more agents mostly just costs more.
        if (
            strategy in ESCALATING
            and (outcome.timed_out or outcome.abandoned)
            and len(episode.involved) < ctx.config.escalate_below
        ):
            for wider in ctx.config.escalate_k:
                if wider <= ctx.config.k_max:
                    continue
                bigger = build_episode(
                    disruption, ctx, seeds=group, k_max=wider
                )
                if bigger is None:
                    break
                retry = solve(bigger, ctx)
                outcome = retry
                if not retry.timed_out and not retry.abandoned:
                    break

        outcomes.append(outcome)

    return _merge(outcomes) if outcomes else None
