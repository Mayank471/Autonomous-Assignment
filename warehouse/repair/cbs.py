"""Conflict-Based Search over the repair neighbourhood.

This one is deliberately **outside the taught syllabus**.  Weiss and Modi et al.
supply the coordination framework and the distributed solver; CBS (Sharon,
Stern, Felner and Sturtevant, *Conflict-based search for optimal multi-agent
pathfinding*, Artificial Intelligence 219, 2015) supplies a better search, and
it was added to fix a specific measured weakness rather than for its own sake.

**The weakness.** The DCOP formulation gives each agent a handful of
precomputed candidate plans and asks which combination fits.  That is faithful
to Ch 12 -- agents control their own variables and nothing else -- but it means
the negotiation can only ever pick from about five plans per agent.  When no
combination of those is collision-free, agents fall back on abandoning their
tasks and retreating.  Measured over 1152 disrupted runs, that cost roughly nine
abandoned agents per run and left the negotiated strategies stranding *more*
agents than the naive no-negotiation baseline (14.2 against 11.9).  Losing to
the baseline on task completion is not a tuning problem; the candidate set is
simply too small a space to search.

**Why CBS fixes it.** CBS does not enumerate plans up front.  Its high level
searches a tree of *constraints*: find a conflict between two agents, and branch
on it by forbidding one agent from that cell at that time, then replan just that
agent.  The space of plans it can reach is therefore the whole space, not a
sample of it, and CBS is complete -- if a collision-free joint plan exists for
these agents, it finds one.  No agent has to give up because the five options it
happened to generate did not fit together.

**Why it also suits this assignment.** CBS has a property that lines up exactly
with "minimise the number of agents whose plans are altered", and it comes for
free rather than being imposed by a cost term: the root node holds the agents'
*current* plans, and an agent is only ever replanned when a constraint is placed
on it.  An agent that is not party to any conflict is never touched.  So the set
of agents that change is precisely the set the conflicts reach -- which is the
smallest set any sound repair could get away with.  The ``beta`` term still
breaks ties in the high-level cost, but it is no longer doing the heavy lifting.

The price is that CBS is exponential in the number of conflicts, so the search
is bounded and falls back to the sequential repair when it runs out.  The bound
is on *low-level searches* rather than wall-clock time: both bound the cost, but
only the former is deterministic, and a repair layer whose results depend on how
busy the machine was cannot be evaluated reproducibly.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Iterator, Mapping

from ..plan import Plan
from ..planner import find_refuge
from ..stastar import plan_route
from .candidates import Candidate, plan_is_valid, retreat_plan, sign_candidates
from .episode import RepairContext, RepairEpisode, RepairOutcome


@dataclass(frozen=True)
class Constraint:
    """Forbid one agent from a cell at a time, or from one transition.

    A vertex constraint keeps ``agent`` off ``cell`` at ``time``.  An edge
    constraint additionally names ``from_cell``, forbidding the specific move
    ``from_cell -> cell`` beginning at ``time`` -- which is how a head-on swap is
    resolved without also banning the cell outright.
    """

    agent: int
    cell: int
    time: int
    from_cell: int | None = None

    @property
    def is_edge(self) -> bool:
        return self.from_cell is not None


@dataclass(frozen=True)
class Conflict:
    """Two agents colliding, either on a cell or by swapping across an edge."""

    a: int
    b: int
    time: int
    cell_a: int
    cell_b: int
    swap: bool = False

    def constraints(self) -> tuple[Constraint, Constraint]:
        """The two ways to resolve it -- one branch of the CBS tree each."""
        if self.swap:
            return (
                Constraint(self.a, self.cell_b, self.time - 1, from_cell=self.cell_a),
                Constraint(self.b, self.cell_a, self.time - 1, from_cell=self.cell_b),
            )
        return (
            Constraint(self.a, self.cell_a, self.time),
            Constraint(self.b, self.cell_a, self.time),
        )


@dataclass(order=True)
class _Node:
    """One node of the constraint tree, ordered by cost for the priority queue."""

    cost: float
    tie: int
    plans: dict[int, Plan] = field(compare=False)
    constraints: tuple[Constraint, ...] = field(compare=False)
    changed: frozenset[int] = field(compare=False)
    abandoned: frozenset[int] = field(compare=False)


@dataclass
class CBSResult:
    plans: dict[int, Plan]
    changed: frozenset[int]
    abandoned: frozenset[int]
    solved: bool
    expansions: int
    low_level_calls: int
    cost: float


def find_conflict(plans: Mapping[int, Plan], now: int) -> Conflict | None:
    """The earliest collision between any two plans, or ``None``.

    Earliest-first matters: resolving the first conflict often removes later
    ones for free, and it keeps the constraint tree shallow.
    """
    agents = sorted(plans)
    if not agents:
        return None
    horizon = max(p.end_time for p in plans.values())

    previous = {a: plans[a].at(now) for a in agents}
    for t in range(now, horizon + 1):
        occupant: dict[int, int] = {}
        current: dict[int, int] = {}
        for agent in agents:
            cell = plans[agent].at(t)
            current[agent] = cell
            other = occupant.get(cell)
            if other is not None:
                return Conflict(other, agent, t, cell, cell)
            occupant[cell] = agent

        if t > now:
            # A swap means some agent b now stands where a just was, while a
            # stands where b just was.  Looking b up by cell keeps this linear
            # rather than comparing every pair -- which matters because this
            # runs once per node of the constraint tree.
            for a in agents:
                if current[a] == previous[a]:
                    continue
                b = occupant.get(previous[a])
                if b is not None and b != a and previous[b] == current[a]:
                    return Conflict(a, b, t, previous[a], previous[b], swap=True)
        previous = current

    return None


def _constraints_for(
    constraints: tuple[Constraint, ...], agent: int
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int, int]]]:
    vertices = {
        (c.cell, c.time) for c in constraints if c.agent == agent and not c.is_edge
    }
    edges = {
        (c.from_cell, c.cell, c.time)
        for c in constraints
        if c.agent == agent and c.is_edge
    }
    return frozenset(vertices), frozenset(edges)


def _plan_for(
    agent: int,
    ctx: RepairContext,
    episode: RepairEpisode,
    constraints: tuple[Constraint, ...],
) -> tuple[Plan | None, bool]:
    """Low level: replan one agent under its own constraints.

    Returns ``(plan, abandons)``.  Everything other than this agent's
    constraints comes from ``episode.baseline``, which holds the uninvolved
    fleet -- so a CBS solution is automatically compatible with the agents that
    were never drawn into the episode.
    """
    vertices, edges = _constraints_for(constraints, agent)
    plan = plan_route(
        ctx.grid,
        ctx.positions[agent],
        ctx.agent_tasks[agent].remaining_from(
            ctx.progress.get(agent, 0), ctx.stages.get(agent, 0)
        ),
        ctx.now,
        agent,
        episode.baseline,
        ctx.obstacles,
        ctx.heuristics,
        service_time=ctx.config.service_time,
        move_filter=ctx.move_filter,
        max_expansions=ctx.config.max_expansions,
        blocked_vertices=vertices,
        blocked_edges=edges,
    )
    if plan is not None:
        return plan, False

    # No route under these constraints.  The agent can still give up its
    # remaining tasks and park somewhere legal -- but *only* somewhere legal.
    #
    # ``retreat_plan`` is deliberately not used here.  Its last resort hands
    # back the plan the agent already holds, which was safe when committed but
    # may since have been invalidated by other agents' repairs.  Everywhere else
    # that is caught downstream; inside a CBS branch it is not, because the
    # conflict may be with an agent this episode never involved and so never
    # checks.  Feeding those back into the search let unsafe plans reach the
    # commit, where the final safety pass truncated 90 agents in one 80-agent
    # run and stranded most of them.
    #
    # A branch that cannot produce a verifiably safe plan is simply pruned.
    refuge = find_refuge(
        ctx.grid, agent, ctx.positions[agent], ctx.agent_tasks[agent].home, ctx.now,
        episode.baseline, ctx.obstacles, ctx.heuristics,
        move_filter=ctx.move_filter,
        max_expansions=ctx.config.max_expansions,
    )
    if refuge is None:
        return None, True
    if any(refuge.at(t) == cell for cell, t in vertices if t <= refuge.end_time):
        return None, True
    return refuge, True


def solve_cbs(episode: RepairEpisode, ctx: RepairContext) -> CBSResult:
    """Search the constraint tree for a collision-free joint plan."""
    config = ctx.config
    now = ctx.now
    agents = list(episode.involved)

    originals: dict[int, Plan] = {}
    for agent in agents:
        current = ctx.reservations.plan_of(agent)
        originals[agent] = (
            current.suffix_from(now)
            if current is not None
            else Plan(agent, now, (ctx.positions[agent],))
        )

    low_level = 0

    def cost_of(plans: Mapping[int, Plan], changed: frozenset[int],
                abandoned: frozenset[int]) -> float:
        travel = sum(p.end_time - now for p in plans.values())
        return (
            config.alpha * travel
            + config.beta * len(changed)
            + config.hold_cost * len(abandoned)
        )

    # Root: keep what each agent already has wherever that is still executable
    # against the uninvolved fleet, and only plan afresh where it is not.  This
    # is what makes "agents changed" small without having to ask for it.
    root_plans: dict[int, Plan] = {}
    root_changed: set[int] = set()
    root_abandoned: set[int] = set()
    for agent in agents:
        if plan_is_valid(originals[agent], ctx, episode.baseline):
            root_plans[agent] = originals[agent]
            continue
        low_level += 1
        plan, abandons = _plan_for(agent, ctx, episode, ())
        if plan is None:
            plan, _ = retreat_plan(agent, ctx, episode.baseline)
            abandons = True
        root_plans[agent] = plan
        root_changed.add(agent)
        if abandons:
            root_abandoned.add(agent)

    tie = 0
    root = _Node(
        cost=cost_of(root_plans, frozenset(root_changed), frozenset(root_abandoned)),
        tie=tie,
        plans=root_plans,
        constraints=(),
        changed=frozenset(root_changed),
        abandoned=frozenset(root_abandoned),
    )
    queue: list[_Node] = [root]
    expansions = 0

    while queue and expansions < config.cbs_max_nodes:
        if low_level >= config.cbs_max_low_level:
            break
        node = heapq.heappop(queue)
        expansions += 1

        conflict = find_conflict(node.plans, now)
        if conflict is None:
            return CBSResult(
                plans=node.plans,
                changed=node.changed,
                abandoned=node.abandoned,
                solved=True,
                expansions=expansions,
                low_level_calls=low_level,
                cost=node.cost,
            )

        for constraint in conflict.constraints():
            agent = constraint.agent
            constraints = node.constraints + (constraint,)
            low_level += 1
            plan, abandons = _plan_for(agent, ctx, episode, constraints)
            if plan is None:
                continue  # this branch is infeasible; the sibling may not be

            plans = dict(node.plans)
            plans[agent] = plan
            changed = node.changed | {agent}
            abandoned = (
                node.abandoned | {agent} if abandons else node.abandoned - {agent}
            )
            tie += 1
            heapq.heappush(
                queue,
                _Node(
                    cost=cost_of(plans, changed, abandoned),
                    tie=tie,
                    plans=plans,
                    constraints=constraints,
                    changed=changed,
                    abandoned=abandoned,
                ),
            )

    return CBSResult(
        plans=root.plans,
        changed=root.changed,
        abandoned=root.abandoned,
        solved=False,
        expansions=expansions,
        low_level_calls=low_level,
        cost=root.cost,
    )


def to_candidates(
    result: CBSResult, episode: RepairEpisode, ctx: RepairContext
) -> dict[int, Candidate]:
    """Wrap a CBS solution as the ``Candidate`` objects ``commit`` expects.

    The space-time signatures are filled in, which is not cosmetic: ``_finalise``
    checks the chosen plans against each other through
    :meth:`Candidate.conflicts_with`, and that check reads the signatures.
    Handing it unsigned candidates makes it silently vacuous -- CBS solutions
    then bypassed every guard and were caught only by the final safety pass,
    which truncated 90 agents in one 80-agent run and stranded most of them.
    """
    out: dict[int, tuple[Candidate, ...]] = {}
    for agent in episode.involved:
        changed = agent in result.changed
        out[agent] = (
            Candidate(
                kind="cbs" if changed else "keep",
                plan=result.plans[agent],
                changed=changed,
                abandons=agent in result.abandoned,
            ),
        )
    signed = sign_candidates(out)
    return {agent: options[0] for agent, options in signed.items()}
