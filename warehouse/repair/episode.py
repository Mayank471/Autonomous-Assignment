"""A repair episode: detect, gather a neighbourhood, negotiate, commit.

The lifecycle implements Ch 11 section 6 with the six PGP mechanisms of section
6.3 mapped onto it:

1. **Detect** (section 6.1 monitoring) -- whose remaining plan the disruption
   invalidated.
2. **Gather a neighbourhood** -- expand a ring outward over the interaction
   graph derived from *abstract* plans (mechanism 1), bounded by ``k_max``.
   This bound is the direct answer to the chain reaction warned of on p528:
   "repair nonetheless could involve all of the agents ... this could lead to
   others needing to change their plans, in a chain reaction".
3. **Generate options** locally, one small domain per agent.
4. **Negotiate** under the meta-level organization's authority ordering
   (mechanism 2), by hill-climbing over the partial global plan (mechanism 3)
   or by solving the DCOP optimally with ADOPT.  Messages are explicit and
   counted (mechanism 4).
5. **Commit atomically**, leaving uninvolved agents running (mechanism 5).
6. **Report** only changes exceeding the dampening threshold ``tau``
   (mechanism 6).

The objective the negotiation minimises is

    sum over involved agents of  alpha * delay(plan) + beta * [plan != keep]

subject to no pair of chosen plans colliding.  The ``beta`` term is the
assignment's "minimise the number of agents whose original plans are altered"
stated as an objective rather than hoped for: at ``beta = 0`` the system buys
any time saving with any amount of churn, and as ``beta`` grows it will accept
a slower joint plan to leave more agents undisturbed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from ..abstractplan import AbstractPlan, InteractionGraph, abstract_plan
from ..disruptions import Disruption, affected_agents
from ..grid import WarehouseGrid
from ..obstacles import DynamicObstacles
from ..organization import MetaLevelOrganization
from ..plan import Plan
from ..reservation import ReservationTable
from ..stastar import HeuristicCache, plan_route
from ..tasks import AgentTasks
from .candidates import Candidate, generate_candidates, retreat_plan, sign_candidates


@dataclass
class RepairConfig:
    """Knobs of the repair layer.  The swept ones are called out below."""

    #: Hard cap on how many agents a single disruption may involve.  **Swept.**
    #:
    #: This is the single most important parameter for ADOPT's tractability.
    #: The textbook is blunt that a complete solver cannot escape this: "the
    #: number of messages that agents need to exchange is, in the worst case,
    #: exponential in the number of variables ... such exponential elements are
    #: unavoidable in complete approaches and they can severely restrict the
    #: scalability of the approach" (p560).  Repair neighbourhoods here are
    #: dense -- nearly every pair of nearby agents has *some* pair of candidate
    #: plans that collide -- so the constraint graph is close to complete and
    #: the bound has to be small.  6 keeps episodes solvable in milliseconds;
    #: at 8 a noticeable fraction exhaust ``max_cycles`` instead.
    k_max: int = 6
    max_hops: int = 3
    #: Weight on extra travel time in the objective.
    alpha: float = 1.0
    #: Penalty per agent that changes its plan.  **Swept** -- this is the
    #: "minimise agents altered" requirement expressed as a cost.
    beta: float = 4.0
    #: Dampening threshold, PGP mechanism 6.  **Swept.**
    tau: int = 0
    #: How far ahead of the disruption the interaction graph looks.
    window: int = 60
    #: ADOPT error bound.  **Swept.**
    epsilon: float = 0.0
    max_cycles: int = 4000
    #: Node budget for the CBS high-level search.  CBS is exponential in the
    #: number of conflicts, so it gets a bound and falls back when it runs out.
    cbs_max_nodes: int = 100
    #: Budget on CBS *low-level* searches, which is where its cost actually
    #: lives: a node is cheap, the multi-leg replan it triggers is not.
    #:
    #: This deliberately replaced a wall-clock budget.  A time budget bounds
    #: cost correctly but makes the search **non-deterministic** -- whether a
    #: node gets expanded depends on how loaded the machine is -- and two runs
    #: of one configuration then disagreed wildly (54% against 86% task
    #: completion on the same seed).  An evaluation whose numbers are not
    #: reproducible from the seed is not an evaluation, so the bound has to be
    #: something the search itself counts.
    cbs_max_low_level: int = 80
    #: On failure, retry once with a neighbourhood this much wider.  Letting the
    #: ring grow only when the narrow one could not be solved is cheaper than
    #: making every episode large, and it turns ``k_max`` from a fixed guess
    #: into a floor.
    #:
    #: Deliberately a single modest step rather than a ladder.  A ladder of
    #: ``(12, 20)`` tripled the search cost of every failed episode, and since
    #: CBS is exponential in conflicts, retrying on twenty agents is where the
    #: blow-up lives: it took the 2048-run sweep from an hour to a projected
    #: four.  Widening helps when the neighbourhood was *marginally* too small,
    #: which is the common case; when it is badly too small, the sequential
    #: fallback is both cheaper and adequate.
    escalate_k: tuple[int, ...] = (10,)
    #: Do not bother widening an episode that is already this large -- that is
    #: where escalation costs the most and helps the least.
    escalate_below: int = 9
    greedy_rounds: int = 10
    #: Offsets for the "stand aside and let them past" option.  Kept to a single
    #: value: domain size enters ADOPT's cost exponentially, and two wait
    #: lengths rarely both matter.
    wait_options: tuple[int, ...] = (2,)
    detour_penalty: float = 3.0
    #: Cost of abandoning the remaining tasks.  Large but finite, so the solver
    #: takes it only when every real option is infeasible.
    hold_cost: float = 2_000.0
    #: Offer "give up and stand still" only when nothing else is feasible.  As a
    #: routine option it just inflates every domain; as a fallback its presence
    #: in the solution is a clean signal that repair genuinely failed.
    always_offer_hold: bool = False
    #: Price of a space-time collision -- the hard constraint, encoded as a soft
    #: one per Ch 12 p550.  It has to exceed any achievable real cost, but only
    #: just: ADOPT refines its bounds *upward* from zero, so an needlessly huge
    #: penalty makes it grind through orders of magnitude of threshold before it
    #: can prove anything.  Sized at roughly ten times the worst feasible joint
    #: cost, which cut solve times by more than an order of magnitude.
    conflict_cost: float = 50_000.0
    service_time: int = 1
    max_expansions: int = 40_000

    def with_(self, **overrides) -> "RepairConfig":
        from dataclasses import replace

        return replace(self, **overrides)


@dataclass
class RepairContext:
    """Everything a repair needs to know about the world right now."""

    grid: WarehouseGrid
    obstacles: DynamicObstacles
    heuristics: HeuristicCache
    reservations: ReservationTable
    agent_tasks: dict[int, AgentTasks]
    positions: dict[int, int]
    progress: dict[int, int]
    #: Per agent: 0 = still to collect the current object, 1 = already carrying.
    stages: dict[int, int]
    now: int
    config: RepairConfig
    mlo: MetaLevelOrganization
    broken: set[int] = field(default_factory=set)
    move_filter: Callable[[int, int], bool] | None = None

    def active_agents(self) -> list[int]:
        return [a for a in self.agent_tasks if a not in self.broken]


@dataclass
class RepairEpisode:
    """One disruption and the negotiation set it produced."""

    disruption: Disruption
    triggers: set[int]
    involved: list[int]
    candidates: dict[int, tuple[Candidate, ...]]
    baseline: ReservationTable
    hops: int
    order: tuple[int, ...]

    def domain_sizes(self) -> dict[int, int]:
        return {a: len(c) for a, c in self.candidates.items()}


@dataclass
class RepairOutcome:
    """What a repair strategy did, and what it cost to do it."""

    plans: dict[int, Plan] = field(default_factory=dict)
    involved: tuple[int, ...] = ()
    changed: tuple[int, ...] = ()
    reported: tuple[int, ...] = ()
    solver: str = ""
    success: bool = True
    cost: float = 0.0
    messages: int = 0
    cycles: int = 0
    wall_ms: float = 0.0
    hops: int = 0
    abandoned: tuple[int, ...] = ()
    searches: int = 0
    timed_out: bool = False
    note: str = ""
    #: Size of each negotiation group this disruption needed.  One disruption
    #: can be repaired in several groups, so ``involved`` is their union while
    #: this stays per group -- and it is the per-group figure that ``k_max``
    #: bounds and that ADOPT's cost depends on.
    episode_sizes: tuple[int, ...] = ()

    @property
    def n_changed(self) -> int:
        """Agents whose original plan was altered -- the assignment's impact metric."""
        return len(self.changed)

    @property
    def n_involved(self) -> int:
        return len(self.involved)

    @property
    def max_episode_size(self) -> int:
        """Largest single negotiation group -- the quantity ``k_max`` caps."""
        return max(self.episode_sizes, default=len(self.involved))


def detect_triggers(
    disruption: Disruption, ctx: RepairContext
) -> set[int]:
    """Agents whose remaining plan the disruption invalidated (Ch 11 section 6.1)."""
    plans = {
        a: p
        for a, p in ((a, ctx.reservations.plan_of(a)) for a in ctx.agent_tasks)
        if p is not None
    }
    return affected_agents(
        disruption, plans, ctx.now, positions=ctx.positions
    ) - ctx.broken


def batch_triggers(
    triggers: Sequence[int] | set[int], ctx: RepairContext
) -> list[list[int]]:
    """Split a large trigger set into groups small enough to negotiate over.

    One disruption can invalidate many plans at once -- a breakdown in a busy
    aisle routinely hits a dozen agents.  Handing all of them to one episode
    would silently defeat ``k_max``, and a complete DCOP solver cannot cope with
    a dense thirteen-variable problem.

    So they are repaired in successive small groups instead, in authority order,
    each group committing before the next begins.  That keeps every negotiation
    tractable and is the more plausible account anyway: the affected robots sort
    it out in local huddles, not in one fleet-wide meeting.  The reported
    "agents changed" is then the total across the groups, which is the honest
    figure for that disruption.
    """
    ordered = list(ctx.mlo.total_order(triggers))
    size = max(1, ctx.config.k_max // 2)
    return [ordered[i : i + size] for i in range(0, len(ordered), size)]


def build_episode(
    disruption: Disruption,
    ctx: RepairContext,
    *,
    seeds: Sequence[int] | None = None,
    k_max: int | None = None,
) -> RepairEpisode | None:
    """Grow a neighbourhood around ``seeds`` and generate each agent's options.

    Args:
        k_max: override the configured neighbourhood cap.  Used by adaptive
            escalation, which retries a failed repair on a wider ring rather
            than accepting that agents must be stranded.

    Returns ``None`` when nothing needs repairing.
    """
    config = ctx.config
    plans = {a: p for a, p in ((a, ctx.reservations.plan_of(a)) for a in ctx.agent_tasks) if p}

    triggers = set(seeds) if seeds is not None else detect_triggers(disruption, ctx)
    triggers -= ctx.broken
    if not triggers:
        return None

    # Mechanism 1: the neighbourhood is read off abstract plans, restricted to
    # the stretch of time the disruption can actually reach.
    abstracts: list[AbstractPlan] = [
        abstract_plan(ctx.grid, p.suffix_from(ctx.now))
        for a, p in plans.items()
        if a not in ctx.broken
    ]
    graph = InteractionGraph(
        abstracts, window=(ctx.now, ctx.now + config.window)
    )
    involved_set, hops = graph.expand(
        triggers,
        k_max=k_max if k_max is not None else config.k_max,
        max_hops=config.max_hops,
    )
    involved_set -= ctx.broken
    if not involved_set:
        return None

    # Mechanism 2: the MLO's authority ranking orders the participants.
    order = ctx.mlo.total_order(involved_set)

    # Lift every participant's commitments so each plans around the *uninvolved*
    # fleet only.  Collisions among participants are the DCOP's business.
    baseline = ctx.reservations.copy()
    for agent in involved_set:
        baseline.remove(agent)
    baseline.rebuild_last_use()

    candidates = {
        agent: generate_candidates(agent, ctx, baseline) for agent in order
    }
    _add_coordinated_option(candidates, order, ctx, baseline)
    candidates = sign_candidates(candidates)

    return RepairEpisode(
        disruption=disruption,
        triggers=triggers,
        involved=list(order),
        candidates=candidates,
        baseline=baseline,
        hops=hops,
        order=order,
    )


def _add_coordinated_option(
    candidates: dict[int, tuple[Candidate, ...]],
    order: Sequence[int],
    ctx: RepairContext,
    baseline: ReservationTable,
) -> None:
    """Give every agent one option that is guaranteed to fit with the others'.

    Each agent generates its own options independently, against the uninvolved
    fleet only.  That is what the DCOP formulation requires -- no agent may
    enumerate anyone else's choices -- but it has a consequence that is easy to
    miss: several agents routinely plan through the same free corridor, so their
    candidates collide *with each other*, and with only a handful of options
    each there may be no collision-free combination at all.  Measured on a
    16-agent floor, 13 of 16 episodes had no feasible joint assignment, so the
    solver's answer was rejected and the sequential fallback did the real work
    62% of the time.  The negotiation was decorative.

    So one more option is added per agent, planned in the meta-level
    organization's authority order against everything already settled: the
    superior commits, inferiors adapt around it.  That is Algorithm 11.2 step 5
    offered as a *candidate* rather than imposed as an outcome, and it makes
    "everybody takes their coordinated plan" feasible by construction.  The DCOP
    now starts from a solvable problem and searches for something better --
    fewer agents disturbed, less delay -- instead of failing and being
    overruled.
    """
    table = baseline.copy()
    for agent in order:
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

        current = ctx.reservations.plan_of(agent)
        unchanged = (
            current is not None and current.suffix_from(ctx.now).cells == plan.cells
        )
        existing = candidates[agent]
        if not any(c.plan.cells == plan.cells for c in existing):
            candidates[agent] = existing + (
                Candidate(
                    kind="keep" if unchanged else "coordinated",
                    plan=plan,
                    changed=not unchanged,
                    abandons=abandons,
                ),
            )
        table.add(plan)


def unary_costs(
    episode: RepairEpisode, config: RepairConfig
) -> dict[int, dict[int, float]]:
    """``alpha * delay + beta * changed`` per candidate, per agent.

    Delay is measured against the agent's *own best* option, not against its
    current plan.  That keeps every cost non-negative, which ADOPT requires, and
    a constant shift per agent leaves the optimal joint assignment unchanged.
    """
    out: dict[int, dict[int, float]] = {}
    for agent, candidates in episode.candidates.items():
        usable = [c for c in candidates if not c.abandons]
        best = min((c.end_time for c in usable), default=0)
        table: dict[int, float] = {}
        for index, candidate in enumerate(candidates):
            if candidate.abandons:
                table[index] = config.hold_cost
                continue
            delay = max(0, candidate.end_time - best)
            table[index] = config.alpha * delay + (
                config.beta if candidate.changed else 0.0
            )
        out[agent] = table
    return out


def conflict_pairs(
    episode: RepairEpisode,
) -> dict[tuple[int, int], dict[tuple[int, int], bool]]:
    """For each agent pair, which candidate combinations collide.

    Pairs with no colliding combination are left out, so the DCOP's constraint
    graph stays as sparse as the problem really is.
    """
    out: dict[tuple[int, int], dict[tuple[int, int], bool]] = {}
    agents = episode.involved
    for i, a in enumerate(agents):
        for b in agents[i + 1 :]:
            table: dict[tuple[int, int], bool] = {}
            any_conflict = False
            for ia, ca in enumerate(episode.candidates[a]):
                for ib, cb in enumerate(episode.candidates[b]):
                    clash = ca.conflicts_with(cb)
                    table[(ia, ib)] = clash
                    any_conflict |= clash
            if any_conflict:
                out[(a, b)] = table
    return out


def evaluate(
    episode: RepairEpisode,
    assignment: Mapping[int, int],
    config: RepairConfig,
    *,
    unary: Mapping[int, Mapping[int, float]] | None = None,
    pairs: Mapping[tuple[int, int], Mapping[tuple[int, int], bool]] | None = None,
) -> float:
    """Total cost of a joint choice, including any collision penalties."""
    unary = unary if unary is not None else unary_costs(episode, config)
    pairs = pairs if pairs is not None else conflict_pairs(episode)

    total = sum(unary[a][assignment[a]] for a in episode.involved)
    for (a, b), table in pairs.items():
        if table.get((assignment[a], assignment[b]), False):
            total += config.conflict_cost
    return total


def assignment_conflicts(
    episode: RepairEpisode, chosen: Mapping[int, Candidate]
) -> list[tuple[int, int]]:
    """Pairs of participants whose chosen plans collide.

    A negotiated assignment is not guaranteed feasible: if *every* joint choice
    in the domains collides, an optimal solver still returns the least-bad one.
    Installing that would put two robots in the same cell, so callers check this
    and fall back rather than trusting the solver's output.
    """
    out: list[tuple[int, int]] = []
    agents = episode.involved
    for i, a in enumerate(agents):
        ca = chosen.get(a)
        if ca is None:
            continue
        for b in agents[i + 1 :]:
            cb = chosen.get(b)
            if cb is not None and ca.conflicts_with(cb):
                out.append((a, b))
    return out


def commit(
    episode: RepairEpisode,
    chosen: Mapping[int, Candidate],
    ctx: RepairContext,
    *,
    solver: str,
    messages: int = 0,
    cycles: int = 0,
    wall_ms: float = 0.0,
    searches: int = 0,
    cost: float = 0.0,
) -> RepairOutcome:
    """Install the chosen plans atomically and report what actually changed.

    "Changed" means the agent's plan differs at all -- that is the number the
    assignment asks for.  "Reported" applies the dampening threshold on top:
    changes smaller than ``tau`` are absorbed silently and never broadcast, so
    the two numbers together show how much of the churn was worth telling
    anyone about.
    """
    changed: list[int] = []
    reported: list[int] = []
    abandoned: list[int] = []
    installed: dict[int, Plan] = {}

    for agent in episode.involved:
        candidate = chosen[agent]
        installed[agent] = candidate.plan
        if candidate.abandons:
            abandoned.append(agent)
        if not candidate.changed:
            continue

        changed.append(agent)
        old = ctx.reservations.plan_of(agent)
        if old is None:
            reported.append(agent)
            continue
        before = abstract_plan(ctx.grid, old.suffix_from(ctx.now))
        after = abstract_plan(ctx.grid, candidate.plan)
        if before.differs_beyond(after, ctx.config.tau):
            reported.append(agent)

    # Atomic swap: drop all participants, then re-add, so a half-applied
    # repair can never be observed.
    for agent in episode.involved:
        ctx.reservations.remove(agent)
    for agent, plan in installed.items():
        ctx.reservations.add(plan)
    ctx.reservations.rebuild_last_use()

    return RepairOutcome(
        plans=installed,
        involved=tuple(episode.involved),
        changed=tuple(changed),
        reported=tuple(reported),
        solver=solver,
        success=not abandoned,
        cost=cost,
        messages=messages,
        cycles=cycles,
        wall_ms=wall_ms,
        hops=episode.hops,
        abandoned=tuple(abandoned),
        searches=searches,
        episode_sizes=(len(episode.involved),),
    )
