"""Candidate plans -- the domains of the repair DCOP.

Each agent drawn into a repair episode generates its own small set of options,
locally, from its own state.  That is not just convenient: it is what keeps the
formulation faithful to Ch 12's model, where "agents can control only the
variables assigned to them ... and agents are only aware of constraints that
involve variables that they can control".  No agent enumerates anyone else's
options.

The option set is deliberately small and ordered by increasing disturbance:

``keep``
    carry on with the existing plan.  Always domain index 0, so every
    tie-break in the solver falls toward leaving the agent alone.
``wait_k``
    a pure delay insertion -- the route is untouched, every step of it simply
    happens *k* ticks later.  Costs no search at all, and it is the cheapest way
    for a bystander to yield right of way.
``splice``
    re-plan only the leg the agent is currently flying and reattach the rest of
    its route where that leg ends.  This is the repair proper: one search rather
    than one per remaining waypoint.
``detour``
    the same splice, with the corridor it was using penalised, so the search
    produces a genuinely different route rather than the same one again.
``reroute``
    re-plan the whole remaining task sequence.  Offered only when nothing above
    worked, because rebuilding an agent's entire route is barely distinguishable
    from replanning it from scratch.
``hold``
    abandon the remaining tasks and retreat (see :func:`retreat_plan`).  A last
    resort, priced so high the solver takes it only when nothing else is
    feasible, and reported as a repair failure rather than hidden.

Every candidate is validated against the reservation table before it is offered.
That matters most for ``splice`` and ``wait_k``, whose reattached or delayed
tails were collision-free at their *original* timing: sliding them later can
walk them into an agent that is not part of this negotiation and so will not be
replanned to get out of the way.

Conflict checking between two candidates is done on precomputed space-time
*signatures* rather than by walking both paths.  An episode compares every pair
of candidates from every pair of agents, so the naive version dominates the
whole simulation's runtime; set intersection in C does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from ..plan import Plan
from ..stastar import plan_route, space_time_astar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .episode import RepairContext


@dataclass(frozen=True)
class Candidate:
    """One plan an agent could adopt, with what it costs and what it disturbs."""

    kind: str
    plan: Plan
    changed: bool
    abandons: bool = False
    #: Space-time signature, filled by :func:`sign_candidates`.
    occupancy: frozenset[tuple[int, int]] = field(default=frozenset(), repr=False)
    moves: frozenset[tuple[int, int, int]] = field(default=frozenset(), repr=False)
    moves_flipped: frozenset[tuple[int, int, int]] = field(
        default=frozenset(), repr=False
    )

    @property
    def agent(self) -> int:
        return self.plan.agent

    @property
    def end_time(self) -> int:
        return self.plan.end_time

    def conflicts_with(self, other: "Candidate") -> bool:
        """Whether the two plans collide in space-time.

        Both signatures are padded to a common horizon, so an agent parked on
        its goal forever is compared correctly against one that arrives later.
        """
        if self.occupancy & other.occupancy:
            return True
        return bool(self.moves & other.moves_flipped)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"Candidate({self.kind}, agent={self.agent}, ends={self.end_time})"


def plan_is_valid(
    plan: Plan,
    ctx: "RepairContext",
    reservations,
) -> bool:
    """Whether ``plan`` is executable against ``reservations`` and the obstacles.

    Used to test whether an agent can simply *keep* what it has -- the outcome
    that costs nothing and disturbs nobody.
    """
    agent = plan.agent
    obstacles = ctx.obstacles

    for i, cell in enumerate(plan.cells):
        t = plan.start_time + i
        if obstacles.is_blocked(cell, t):
            return False
        if not reservations.is_vertex_free(cell, t, agent):
            return False
    for u, v, t in plan.moves():
        if not reservations.is_edge_free(u, v, t, agent):
            return False

    # The agent parks on its final cell, so that cell must stay usable forever.
    if obstacles.blocked_from(plan.goal) != float("inf"):
        return False
    perm = reservations.permanent_owner(plan.goal)
    if perm is not None and perm[1] != agent:
        return False
    return plan.end_time >= reservations.last_use(plan.goal)


def generate_candidates(
    agent: int,
    ctx: "RepairContext",
    reservations,
) -> tuple[Candidate, ...]:
    """Build ``agent``'s option set, planned against ``reservations``.

    ``reservations`` must already have every *involved* agent's plan lifted, so
    each agent plans around the uninvolved fleet only; collisions among the
    involved are what the DCOP's binary constraints are for.
    """
    config = ctx.config
    now = ctx.now
    position = ctx.positions[agent]
    waypoints = ctx.agent_tasks[agent].remaining_from(
        ctx.progress.get(agent, 0), ctx.stages.get(agent, 0)
    )
    current = ctx.reservations.plan_of(agent)

    suffix = current.suffix_from(now) if current is not None else None
    out: list[Candidate] = []
    seen: set[tuple[int, ...]] = set()

    def offer(kind: str, plan: Plan | None, *, changed: bool, abandons: bool = False):
        if plan is None:
            return
        key = (plan.start_time, *plan.cells)
        if key in seen:
            return
        seen.add(key)
        out.append(Candidate(kind=kind, plan=plan, changed=changed, abandons=abandons))

    # --- keep -------------------------------------------------------------
    if suffix is not None and plan_is_valid(suffix, ctx, reservations):
        offer("keep", suffix, changed=False)

    common = dict(
        service_time=config.service_time,
        move_filter=ctx.move_filter,
        max_expansions=config.max_expansions,
    )

    # --- wait k, then carry on unchanged ----------------------------------
    # A pure delay insertion: the route is untouched, every step of it simply
    # happens k ticks later.  No search at all, and it is the cheapest way for a
    # bystander to yield right of way.
    if suffix is not None:
        for k in config.wait_options:
            if not _can_stand(position, now, k, ctx, reservations, agent):
                continue
            delayed = Plan(agent, now, tuple([position] * k) + suffix.cells)
            if plan_is_valid(delayed, ctx, reservations):
                offer(f"wait{k}", delayed, changed=True)

    # --- splice: repair the damaged leg, keep the rest of the route --------
    # The assignment's actual requirement -- repair, not replan.  Only the leg
    # the agent is currently flying is re-planned; the remainder of its route is
    # reattached where that leg ends, shifted in time to meet it.  One search
    # instead of one per remaining waypoint, which is where nearly all the cost
    # of this system used to go.
    if suffix is not None and waypoints:
        offer("splice", _splice(agent, ctx, reservations, suffix, waypoints[0], common),
              changed=True)
        offer(
            "detour",
            _splice(
                agent, ctx, reservations, suffix, waypoints[0], common,
                avoid=frozenset(suffix.cells), avoid_penalty=config.detour_penalty,
            ),
            changed=True,
        )

    # --- reroute: replan the whole remaining task sequence -----------------
    # The expensive option, offered only when splicing could not produce a
    # usable plan, because a repair that rebuilds the entire route is barely
    # distinguishable from replanning that agent from scratch.
    if not any(c.changed and not c.abandons for c in out):
        offer(
            "reroute",
            plan_route(
                ctx.grid, position, waypoints, now, agent,
                reservations, ctx.obstacles, ctx.heuristics, **common,
            ),
            changed=True,
        )

    # --- retreat: give up on the tasks and go park ------------------------
    # Never a bare standstill.  Freezing an agent mid-aisle turns it into a wall
    # that the rest of the fleet was never told about, and agents whose plans
    # already ran through that cell would drive straight into it.  Retreating to
    # its own dock is safe by construction.
    if not out or config.always_offer_hold:
        fallback, _ = retreat_plan(agent, ctx, reservations)
        offer("hold", fallback, changed=True, abandons=True)

    return tuple(out)


def safe_prefix(
    plan: Plan,
    ctx: "RepairContext",
    reservations,
) -> Plan | None:
    """The longest leading portion of ``plan`` the agent can execute and then stop on.

    This is what physically happens when a robot's way is blocked and nobody
    yields: it advances as far as it safely can and stops there.  Stopping is
    only safe if the cell it stops on stays free forever after, so the prefix is
    trimmed back until that holds -- otherwise the "stalled" robot would be
    standing where someone else is due to drive through.

    Returns ``None`` if the agent cannot even hold its current cell.
    """
    agent = plan.agent
    obstacles = ctx.obstacles

    # How far it can get before hitting a blocked cell or another agent.
    reach = 0
    for i, cell in enumerate(plan.cells):
        t = plan.start_time + i
        if obstacles.is_blocked(cell, t) or not reservations.is_vertex_free(cell, t, agent):
            break
        if i > 0:
            u, v = plan.cells[i - 1], cell
            if not reservations.is_edge_free(u, v, plan.start_time + i - 1, agent):
                break
        reach = i

    # Trim back until the stopping cell can be held indefinitely.
    for i in range(reach, -1, -1):
        cell = plan.cells[i]
        stop = plan.start_time + i
        if obstacles.blocked_from(cell) != float("inf"):
            continue
        perm = reservations.permanent_owner(cell)
        if perm is not None and perm[1] != agent:
            continue
        if stop < reservations.last_use(cell):
            continue
        return Plan(agent, plan.start_time, plan.cells[: i + 1])

    return None


def retreat_plan(
    agent: int,
    ctx: "RepairContext",
    reservations,
) -> tuple[Plan, bool]:
    """What an agent does when it cannot complete its tasks: go home and park.

    Returns ``(plan, abandons)``.  Ordered by preference:

    1. **Route to its own dock.**  Safe by construction -- a dock belongs to one
       agent for all time, so nobody else can ever be there and parking on it
       can never clash with anyone.
    2. **Advance as far as it safely can and stop** (:func:`safe_prefix`).
    3. **Stand still**, but only if standing still is itself safe.
    4. **Keep the plan it already had**, as a last resort.

    Fabricating an unchecked standstill instead of this is what produced
    permanent-reservation collisions: a robot frozen mid-aisle is a wall, and
    other agents were still routed through it.
    """
    from ..planner import find_refuge  # local import: avoids a cycle

    position = ctx.positions[agent]
    current = ctx.reservations.plan_of(agent)
    base = (
        current.suffix_from(ctx.now)
        if current is not None
        else Plan(agent, ctx.now, (position,))
    )

    # Stopping part-way along the route it was already following is less
    # disruptive than driving all the way home, so try that first.
    trimmed = safe_prefix(base, ctx, reservations)
    if trimmed is not None and len(trimmed.cells) > 1:
        return trimmed, True

    refuge = find_refuge(
        ctx.grid, agent, position, ctx.agent_tasks[agent].home, ctx.now,
        reservations, ctx.obstacles, ctx.heuristics,
        move_filter=ctx.move_filter,
        max_expansions=ctx.config.max_expansions,
    )
    if refuge is not None:
        return refuge, True

    # Walled in with nowhere legal to park.  Hand back the plan it already
    # holds: that one was collision-free when it was committed, so it cannot
    # introduce a *new* conflict, and the caller checks the result anyway.
    return base, True


def _splice(
    agent: int,
    ctx: "RepairContext",
    reservations,
    suffix: Plan,
    waypoint: int,
    common: dict,
    *,
    avoid: frozenset[int] | None = None,
    avoid_penalty: float = 0.0,
) -> Plan | None:
    """Re-plan only the leg to ``waypoint`` and reattach the rest of the route.

    The agent keeps everything it had planned beyond ``waypoint``; that portion
    just starts whenever the new leg actually gets there.  If the new leg is
    slower the whole tail slides later, which is a delay the negotiation can
    price; if it is the same length the result is identical to ``keep``.

    Returns ``None`` when the original route never reaches ``waypoint`` (so
    there is nothing to reattach), the leg cannot be planned, or the reattached
    tail does not survive the shift.  That last check is essential: the tail was
    collision-free at its *original* timing, and sliding it later can walk it
    straight into an agent that is not part of this negotiation and will not be
    replanned.
    """
    try:
        index = suffix.cells.index(waypoint)
    except ValueError:
        return None

    leg = space_time_astar(
        ctx.grid, suffix.cells[0], waypoint, suffix.start_time, agent,
        reservations, ctx.obstacles, ctx.heuristics,
        hold=common["service_time"],
        avoid=avoid, avoid_penalty=avoid_penalty,
        move_filter=common["move_filter"],
        max_expansions=common["max_expansions"],
    )
    if leg is None:
        return None

    tail = suffix.cells[index + 1 :]
    spliced = Plan(agent, suffix.start_time, leg.cells + tail)
    return spliced if plan_is_valid(spliced, ctx, reservations) else None


def _can_stand(
    cell: int, now: int, k: int, ctx: "RepairContext", reservations, agent: int
) -> bool:
    """Whether ``agent`` may remain on ``cell`` for the next ``k`` steps."""
    for t in range(now, now + k + 1):
        if ctx.obstacles.is_blocked(cell, t):
            return False
        if not reservations.is_vertex_free(cell, t, agent):
            return False
    return True


def sign_candidates(
    candidates_by_agent: dict[int, tuple[Candidate, ...]]
) -> dict[int, tuple[Candidate, ...]]:
    """Attach space-time signatures, padded to a horizon shared by all candidates.

    Padding matters: a plan ends by parking on its goal forever, so without a
    common horizon a short plan would appear not to collide with a longer one
    that walks over its parking spot later.
    """
    horizon = 0
    earliest = None
    for candidates in candidates_by_agent.values():
        for candidate in candidates:
            horizon = max(horizon, candidate.plan.end_time)
            start = candidate.plan.start_time
            earliest = start if earliest is None else min(earliest, start)
    if earliest is None:
        return candidates_by_agent

    out: dict[int, tuple[Candidate, ...]] = {}
    for agent, candidates in candidates_by_agent.items():
        signed: list[Candidate] = []
        for candidate in candidates:
            plan = candidate.plan
            occupancy = {(t, plan.at(t)) for t in range(earliest, horizon + 1)}
            moves = {
                (t, u, v) for u, v, t in plan.moves() if u != v
            }
            signed.append(
                Candidate(
                    kind=candidate.kind,
                    plan=plan,
                    changed=candidate.changed,
                    abandons=candidate.abandons,
                    occupancy=frozenset(occupancy),
                    moves=frozenset(moves),
                    moves_flipped=frozenset((t, v, u) for t, u, v in moves),
                )
            )
        out[agent] = tuple(signed)
    return out
