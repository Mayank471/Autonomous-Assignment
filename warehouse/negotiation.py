"""Negotiation protocol between agents during a repair session.

A disruption starts a *session* led by one disrupted agent.  Agents exchange
four kinds of message:

``REPLAN_REQUEST``  leader -> j: "propose a new route that respects these
                    constraints and treats these agents' plans as fixed"
``PROPOSAL``        j -> leader: a path and its cost
``CANNOT_COMPLY``   j -> leader: no path exists under those constraints
``COMMIT``          leader -> j: adopt this path (sent to every agent whose plan
                    changes)
``RELEASE``         leader -> j: you were consulted but keep your plan

Every agent plans only its own route, using its own goal list, so a task list
never leaves its owner.  Reading the shared reservation board is not a
message.  All low-level searches made during a repair -- by any strategy -- go
through :meth:`Session.replan`, so message counts and search effort are
measured the same way for every strategy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from .grid import WarehouseGrid
from .obstacles import ObstacleWindows
from .plan import Plan
from .reservation import ReservationBoard
from .stastar import NO_CONSTRAINTS, Constraints, Query, SearchResult, search

REPLAN_REQUEST, PROPOSAL, CANNOT_COMPLY, COMMIT, RELEASE = (
    "REPLAN_REQUEST", "PROPOSAL", "CANNOT_COMPLY", "COMMIT", "RELEASE")


@dataclass
class Session:
    grid: WarehouseGrid
    obstacles: ObstacleWindows
    board: ReservationBoard
    query_of: Callable[[int], Query]          # an agent's private view of its own task
    leader: int
    max_expansions_per_call: int = 400_000
    messages: Counter = field(default_factory=Counter)
    contacted: set[int] = field(default_factory=set)
    ll_calls: int = 0
    ll_expansions: int = 0
    ll_budget_hits: int = 0
    relax_deadlines: bool = False     # set when an emergency deadline became impossible

    def _send(self, kind: str, to: int) -> None:
        self.messages[kind] += 1
        if to != self.leader:
            self.contacted.add(to)

    def replan(self, agent: int, *, ignore: frozenset[int] | set[int] = frozenset(),
               hard: frozenset[int] | set[int] | None = None,
               constraints: Constraints = NO_CONSTRAINTS,
               board: ReservationBoard | None = None,
               cat: ReservationBoard | None = None) -> SearchResult:
        """Ask ``agent`` for its best route under the given conditions."""
        remote = agent != self.leader
        if remote:
            self._send(REPLAN_REQUEST, agent)
        q = self.query_of(agent)
        if self.relax_deadlines and q.deadline is not None:
            q = Query(q.agent, q.start, q.t0, q.goals, q.dock, q.hold_until, None)
        res = search(self.grid, q, obstacles=self.obstacles,
                     board=self.board if board is None else board,
                     ignore=ignore, hard=hard, constraints=constraints, cat=cat,
                     max_expansions=self.max_expansions_per_call)
        self.ll_calls += 1
        self.ll_expansions += res.expansions
        if res.status == "budget":
            self.ll_budget_hits += 1
        if remote:
            self.messages[PROPOSAL if res.ok else CANNOT_COMPLY] += 1
        return res

    def close(self, old: dict[int, Plan], new: dict[int, Plan], t: int) -> set[int]:
        """Send COMMIT/RELEASE and return the agents whose plans changed."""
        from .plan import plans_differ
        changed = {a for a, p in new.items() if plans_differ(old[a], p, t)}
        for a in sorted(changed):
            if a != self.leader:
                self._send(COMMIT, a)
        for a in sorted(self.contacted - changed):
            self._send(RELEASE, a)
        return changed

    @property
    def n_messages(self) -> int:
        return sum(self.messages.values())
