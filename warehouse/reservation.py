"""Space-time reservation table -- the shared structure agents coordinate through.

This is the concrete representation of the "constraints enforce that the plans
dovetail together suitably" idea from Weiss Ch 11 p497.  Every committed plan
contributes three kinds of reservation:

* **vertex** -- agent *a* occupies cell *v* at time *t*;
* **edge** -- agent *a* traverses *u -> v* between *t* and *t+1*, which forbids
  another agent traversing *v -> u* over the same interval (a head-on swap);
* **permanent** -- agent *a* parks on its final cell from its arrival time
  onward, so that cell is blocked for everyone else forever after.

Reservations are per-agent revocable, because plan repair needs to lift one
agent's commitments, replan it, and put them back.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from .plan import Plan

INF = float("inf")


class ReservationTable:
    """Vertex, edge and permanent reservations, indexed for fast lookup."""

    __slots__ = (
        "_vertex", "_edge", "_permanent", "_by_agent", "_last_use", "_horizon", "_docks",
    )

    def __init__(self) -> None:
        self._vertex: dict[tuple[int, int], int] = {}
        self._edge: dict[tuple[int, int, int], int] = {}
        self._permanent: dict[int, tuple[int, int]] = {}  # cell -> (from_time, agent)
        # Dedicated docks: cell -> owning agent, for all time.  Unlike the other
        # indices these survive plan removal, because a dock belongs to its agent
        # whatever that agent is currently doing.
        self._docks: dict[int, int] = {}
        self._by_agent: dict[int, Plan] = {}
        # cell -> latest time any agent has a *vertex* reservation there.  Used to
        # decide whether an agent may settle on a cell permanently.
        self._last_use: dict[int, int] = {}
        self._horizon: int = 0

    # ------------------------------------------------------------------ basics

    def __len__(self) -> int:
        return len(self._by_agent)

    def __contains__(self, agent: int) -> bool:
        return agent in self._by_agent

    def agents(self) -> Iterator[int]:
        return iter(self._by_agent)

    def plans(self) -> Iterator[Plan]:
        return iter(self._by_agent.values())

    def plan_of(self, agent: int) -> Plan | None:
        return self._by_agent.get(agent)

    @property
    def horizon(self) -> int:
        """Latest time any reservation refers to."""
        return self._horizon

    def copy(self) -> "ReservationTable":
        """A deep-enough copy: plans are immutable, so only the indices are cloned."""
        other = ReservationTable()
        other._vertex = dict(self._vertex)
        other._edge = dict(self._edge)
        other._permanent = dict(self._permanent)
        other._docks = dict(self._docks)
        other._by_agent = dict(self._by_agent)
        other._last_use = dict(self._last_use)
        other._horizon = self._horizon
        return other

    # ------------------------------------------------------------- maintenance

    def add(self, plan: Plan) -> None:
        """Commit ``plan``, replacing any plan already held for that agent."""
        if plan.agent in self._by_agent:
            self.remove(plan.agent)

        agent = plan.agent
        t0 = plan.start_time
        for i, cell in enumerate(plan.cells):
            t = t0 + i
            self._vertex[(cell, t)] = agent
            if t > self._last_use.get(cell, -1):
                self._last_use[cell] = t
        for u, v, t in plan.moves():
            self._edge[(u, v, t)] = agent

        self._permanent[plan.goal] = (plan.end_time, agent)
        self._by_agent[agent] = plan
        if plan.end_time > self._horizon:
            self._horizon = plan.end_time

    def remove(self, agent: int) -> Plan | None:
        """Lift ``agent``'s reservations and return the plan that held them."""
        plan = self._by_agent.pop(agent, None)
        if plan is None:
            return None

        t0 = plan.start_time
        for i, cell in enumerate(plan.cells):
            key = (cell, t0 + i)
            if self._vertex.get(key) == agent:
                del self._vertex[key]
        for u, v, t in plan.moves():
            key = (u, v, t)
            if self._edge.get(key) == agent:
                del self._edge[key]
        perm = self._permanent.get(plan.goal)
        if perm is not None and perm[1] == agent:
            del self._permanent[plan.goal]

        # _last_use and _horizon are upper bounds used only to decide whether a
        # cell may be settled on permanently; leaving them stale is conservative
        # (it can only make the planner wait longer, never produce a collision).
        return plan

    def dedicate(self, cell: int, agent: int) -> None:
        """Reserve ``cell`` as ``agent``'s dock, for all time.

        Agents finish their route parked on their dock, which makes it a
        permanent reservation.  Dedicating it up front stops other agents
        routing through a cell that will be occupied forever anyway -- without
        this they plan through it early on and then have to be displaced, and
        the owner cannot settle until the last of that traffic has cleared.
        """
        self._docks[cell] = agent

    def dock_owner(self, cell: int) -> int | None:
        return self._docks.get(cell)

    def add_all(self, plans: Iterable[Plan]) -> None:
        for plan in plans:
            self.add(plan)

    def clear(self) -> None:
        self._vertex.clear()
        self._edge.clear()
        self._permanent.clear()
        self._by_agent.clear()
        self._last_use.clear()
        self._horizon = 0

    def rebuild_last_use(self) -> None:
        """Recompute the ``_last_use`` index exactly, discarding stale upper bounds.

        Removal leaves ``_last_use`` conservative rather than exact; call this
        after a batch of removals if the slack starts costing path quality.
        """
        self._last_use.clear()
        self._horizon = 0
        for plan in self._by_agent.values():
            for i, cell in enumerate(plan.cells):
                t = plan.start_time + i
                if t > self._last_use.get(cell, -1):
                    self._last_use[cell] = t
            self._horizon = max(self._horizon, plan.end_time)

    # ---------------------------------------------------------------- querying

    def vertex_owner(self, cell: int, t: int) -> int | None:
        """Agent occupying ``cell`` at ``t``, whether transiently or permanently."""
        dock = self._docks.get(cell)
        if dock is not None:
            return dock
        owner = self._vertex.get((cell, t))
        if owner is not None:
            return owner
        perm = self._permanent.get(cell)
        if perm is not None and t >= perm[0]:
            return perm[1]
        return None

    def is_vertex_free(self, cell: int, t: int, agent: int) -> bool:
        """Whether ``agent`` may occupy ``cell`` at time ``t``."""
        dock = self._docks.get(cell)
        if dock is not None and dock != agent:
            return False
        owner = self._vertex.get((cell, t))
        if owner is not None and owner != agent:
            return False
        perm = self._permanent.get(cell)
        if perm is not None and perm[1] != agent and t >= perm[0]:
            return False
        return True

    def is_edge_free(self, u: int, v: int, t: int, agent: int) -> bool:
        """Whether ``agent`` may traverse ``u -> v`` starting at ``t`` (no swap)."""
        owner = self._edge.get((v, u, t))
        return owner is None or owner == agent

    def last_use(self, cell: int) -> int:
        """Latest time any agent has a transient reservation on ``cell``."""
        return self._last_use.get(cell, -1)

    def permanent_owner(self, cell: int) -> tuple[int, int] | None:
        """``(from_time, agent)`` if some agent parks on ``cell``, else ``None``."""
        return self._permanent.get(cell)

    def blocking_agents(self, cell: int, t: int, agent: int) -> set[int]:
        """Agents whose reservations prevent ``agent`` occupying ``cell`` at ``t``.

        Used by the repair layer to work out *who* to negotiate with, rather
        than merely that the search failed.
        """
        out: set[int] = set()
        dock = self._docks.get(cell)
        if dock is not None and dock != agent:
            out.add(dock)
        owner = self._vertex.get((cell, t))
        if owner is not None and owner != agent:
            out.add(owner)
        perm = self._permanent.get(cell)
        if perm is not None and perm[1] != agent and t >= perm[0]:
            out.add(perm[1])
        return out

    # ------------------------------------------------------------- consistency

    def find_conflicts(self) -> list[tuple[str, int, int, int]]:
        """Return every pairwise conflict in the committed plans.

        Each entry is ``(kind, agent_a, agent_b, time)`` with ``kind`` one of
        ``"vertex"``, ``"edge"`` or ``"permanent"``.  A correct joint plan has
        none; the repair tests assert exactly that after every commit.
        """
        conflicts: list[tuple[str, int, int, int]] = []
        plans = list(self._by_agent.values())
        if not plans:
            return conflicts

        horizon = max(p.end_time for p in plans)
        for t in range(min(p.start_time for p in plans), horizon + 1):
            seen: dict[int, int] = {}
            for plan in plans:
                if t < plan.start_time:
                    continue
                cell = plan.at(t)
                other = seen.get(cell)
                if other is not None:
                    kind = "permanent" if t > plan.end_time or t > _end(plans, other) else "vertex"
                    conflicts.append((kind, other, plan.agent, t))
                else:
                    seen[cell] = plan.agent

        # Edge (swap) conflicts.
        for plan in plans:
            for u, v, t in plan.moves():
                owner = self._edge.get((v, u, t))
                if owner is not None and owner != plan.agent:
                    if plan.agent < owner:  # report each pair once
                        conflicts.append(("edge", plan.agent, owner, t))
        return conflicts

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"ReservationTable(agents={len(self._by_agent)}, "
            f"vertices={len(self._vertex)}, edges={len(self._edge)}, "
            f"horizon={self._horizon})"
        )


def _end(plans: list[Plan], agent: int) -> int:
    for p in plans:
        if p.agent == agent:
            return p.end_time
    return -1
