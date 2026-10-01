"""Abstract plans -- PGP mechanisms 1 and 6 (Weiss Ch 11 section 6.3, p530).

**Mechanism 1, abstraction.**  The textbook's description of partial global
planning is explicit that agents do not exchange their detailed plans:

    "even though each agent builds a detailed plan for processing signals and
    constructing larger interpretations, the agents communicate with each other
    only about what partial interpretations they plan to construct and when.
    Thus, even though internally an agent might frequently make changes to its
    detailed plan, other agents' perceptions of its plans generally evolve much
    more slowly."

The warehouse analogue of "what, and when" is *which zone the agent will be in,
and over what time window*.  So an :class:`AbstractPlan` is a sequence of
``(zone, enter, exit)`` intervals, which is one to two orders of magnitude
smaller than the cell-by-cell path and is what agents actually broadcast.

**Mechanism 6, dampened responsiveness.**  Not every change is worth telling
anyone about.  PGP used "a simple thresholding strategy: if an abstract plan
step will occur earlier or later than expected by more than a parameterized
number of time steps, then the agent should alert others.  A larger threshold
effectively introduced slack into the system, cutting down on coordination
overhead".  :meth:`AbstractPlan.differs_beyond` is that test, and ``tau`` is
that parameter -- swept in the experiments because it is the most direct lever
on how many agents a single disruption drags into a negotiation.

The same abstraction also answers "who should I even talk to?".  Agents whose
zone-time intervals never overlap cannot collide, so
:class:`InteractionGraph` derives the negotiation neighbourhood from the
abstract plans alone, without ever comparing two detailed paths.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .grid import WarehouseGrid
from .plan import Plan


@dataclass(frozen=True)
class ZoneInterval:
    """An agent's occupancy of one zone over a contiguous time window."""

    zone: int
    enter: int
    exit: int

    def overlaps(self, other: "ZoneInterval", slack: int = 0) -> bool:
        if self.zone != other.zone:
            return False
        return self.enter - slack <= other.exit and other.enter - slack <= self.exit


@dataclass(frozen=True)
class AbstractPlan:
    """The zone-level summary of a detailed plan -- what an agent broadcasts."""

    agent: int
    intervals: tuple[ZoneInterval, ...]

    @property
    def start_time(self) -> int:
        return self.intervals[0].enter if self.intervals else 0

    @property
    def end_time(self) -> int:
        return self.intervals[-1].exit if self.intervals else 0

    def zones(self) -> frozenset[int]:
        return frozenset(iv.zone for iv in self.intervals)

    def differs_beyond(self, other: "AbstractPlan", tau: int) -> bool:
        """Whether this plan differs from ``other`` by more than the slack ``tau``.

        PGP mechanism 6: below the threshold the change is absorbed silently and
        no one is told, which is what keeps a small local adjustment from
        cascading into a fleet-wide renegotiation.
        """
        if len(self.intervals) != len(other.intervals):
            return True
        for a, b in zip(self.intervals, other.intervals):
            if a.zone != b.zone:
                return True
            if abs(a.enter - b.enter) > tau or abs(a.exit - b.exit) > tau:
                return True
        return False

    def compression_ratio(self, plan: Plan) -> float:
        """Abstract entries per detailed cell -- how much abstraction actually saves."""
        return len(self.intervals) / max(1, len(plan.cells))

    def __len__(self) -> int:
        return len(self.intervals)


def abstract_plan(grid: WarehouseGrid, plan: Plan) -> AbstractPlan:
    """Summarise ``plan`` as the sequence of zones it passes through."""
    intervals: list[ZoneInterval] = []
    current_zone: int | None = None
    enter = plan.start_time

    for i, cell in enumerate(plan.cells):
        t = plan.start_time + i
        zone = grid.zone_of(cell)
        if zone != current_zone:
            if current_zone is not None:
                intervals.append(ZoneInterval(current_zone, enter, t - 1))
            current_zone = zone
            enter = t

    if current_zone is not None:
        intervals.append(ZoneInterval(current_zone, enter, plan.end_time))

    return AbstractPlan(plan.agent, tuple(intervals))


class InteractionGraph:
    """Which agents could possibly interfere with which, from abstract plans alone.

    Two agents are linked when their zone-time intervals overlap: they are in
    the same part of the floor at overlapping times, so one's change could
    matter to the other.  Agents that never share a zone-time cannot collide,
    whatever either of them replans.

    This is what makes the repair *local*.  Building it costs one pass over the
    abstract intervals rather than an all-pairs comparison of detailed paths,
    and it defines exactly the neighbourhood Ch 12 means by "only neighboring
    agents can directly communicate with each other".
    """

    __slots__ = ("_adj", "_by_zone", "_weight", "slack")

    def __init__(
        self,
        abstracts: Iterable[AbstractPlan],
        *,
        slack: int = 2,
        window: tuple[int, int] | None = None,
    ) -> None:
        """
        Args:
            slack: treat intervals this close in time as overlapping.
            window: only consider the part of each plan inside ``(from, to)``.
                A disruption only matters over the stretch of time around it,
                and restricting to that stretch keeps the neighbourhood from
                swelling to include every agent that shares a zone at any point
                in the whole run.
        """
        self.slack = slack
        self._adj: dict[int, set[int]] = defaultdict(set)
        self._by_zone: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        # Edge weight: how long two agents are co-located in a zone.  Used to
        # order the expanding ring by interaction strength.
        self._weight: dict[tuple[int, int], int] = defaultdict(int)

        lo, hi = window if window is not None else (None, None)
        for ap in abstracts:
            self._adj.setdefault(ap.agent, set())
            for iv in ap.intervals:
                enter, exit_ = iv.enter, iv.exit
                if lo is not None:
                    if exit_ < lo or enter > hi:
                        continue
                    enter, exit_ = max(enter, lo), min(exit_, hi)
                self._by_zone[iv.zone].append((enter, exit_, ap.agent))

        for entries in self._by_zone.values():
            entries.sort()
            for i, (enter_a, exit_a, agent_a) in enumerate(entries):
                for enter_b, exit_b, agent_b in entries[i + 1 :]:
                    if enter_b - slack > exit_a:
                        break  # sorted by entry: nothing later can overlap either
                    if agent_a == agent_b:
                        continue
                    self._adj[agent_a].add(agent_b)
                    self._adj[agent_b].add(agent_a)
                    overlap = min(exit_a, exit_b) - max(enter_a, enter_b) + 1
                    key = (agent_a, agent_b) if agent_a < agent_b else (agent_b, agent_a)
                    self._weight[key] += max(1, overlap)

    def neighbors(self, agent: int) -> set[int]:
        return self._adj.get(agent, set())

    def weight(self, a: int, b: int) -> int:
        """How strongly two agents interact -- total co-located zone time."""
        key = (a, b) if a < b else (b, a)
        return self._weight.get(key, 0)

    def agents(self) -> tuple[int, ...]:
        return tuple(self._adj)

    def expand(
        self,
        seeds: Iterable[int],
        *,
        k_max: int,
        max_hops: int = 3,
    ) -> tuple[set[int], int]:
        """Grow an expanding ring outward from ``seeds``.

        Returns the involved set and how many hops it took.  Growth stops at
        ``k_max`` agents, which is the hard bound on how far a single
        disruption is allowed to propagate -- the countermeasure to the "chain
        reaction of planning and coordination efforts" warned about on p528.

        Agents are admitted nearest hop first and, within a hop, strongest
        interaction first, so that truncating at ``k_max`` keeps the agents
        that matter most rather than the ones with the lowest ids.
        """
        involved = set(seeds)
        frontier = set(involved)
        hops = 0

        while frontier and len(involved) < k_max and hops < max_hops:
            hops += 1
            nxt: set[int] = set()
            for agent in frontier:
                nxt |= self.neighbors(agent)
            nxt -= involved
            if not nxt:
                break

            def relevance(candidate: int) -> tuple[int, int]:
                strength = max(self.weight(candidate, a) for a in frontier)
                return (-strength, candidate)

            added: set[int] = set()
            for agent in sorted(nxt, key=relevance):
                if len(involved) >= k_max:
                    break
                involved.add(agent)
                added.add(agent)
            frontier = added

        return involved, hops

    def degree(self, agent: int) -> int:
        return len(self._adj.get(agent, ()))

    def __len__(self) -> int:
        return len(self._adj)


def build_abstracts(
    grid: WarehouseGrid, plans: Mapping[int, Plan]
) -> dict[int, AbstractPlan]:
    """Abstract every plan in one pass."""
    return {agent: abstract_plan(grid, plan) for agent, plan in plans.items()}
