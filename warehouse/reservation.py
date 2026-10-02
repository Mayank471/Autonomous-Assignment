"""The shared space-time reservation board.

Every committed plan is posted here; any agent can read it (like a warehouse
traffic-management system).  Committed plans are mutually collision-free, so
each (time, cell) and each directed move has at most one owner.

Keys are packed integers for speed:

* vertex ``(t, cell)``           -> ``t * NC + cell``
* move ``a -> b`` during t..t+1  -> ``(t * NC + a) * NC + b``

After its plan ends an agent is *parked* at its last cell forever.
"""

from __future__ import annotations

from .plan import Plan


class ReservationBoard:
    def __init__(self, n_cells: int) -> None:
        self.nc = n_cells
        self.vertex: dict[int, int] = {}
        self.edge: dict[int, int] = {}
        self.parked: dict[int, tuple[int, int]] = {}   # cell -> (agent, from_t)
        self._keys: dict[int, tuple[list[int], list[int], int]] = {}
        self.max_t = 0

    # ----------------------------------------------------------------- update
    def add(self, agent: int, plan: Plan, t_from: int) -> None:
        if agent in self._keys:
            self.remove(agent)
        nc = self.nc
        vkeys, ekeys = [], []
        start = max(t_from, plan.t0)
        end = plan.end
        prev = plan.at(start)
        for t in range(start, end + 1):
            cell = plan.at(t)
            if t > start and cell != prev:
                k = ((t - 1) * nc + prev) * nc + cell
                self.edge[k] = agent
                ekeys.append(k)
            k = t * nc + cell
            self.vertex[k] = agent
            vkeys.append(k)
            prev = cell
        last = plan.cells[-1]
        self.parked[last] = (agent, max(end, start))
        self._keys[agent] = (vkeys, ekeys, last)
        if end > self.max_t:
            self.max_t = end

    def remove(self, agent: int) -> None:
        entry = self._keys.pop(agent, None)
        if entry is None:
            return
        vkeys, ekeys, last = entry
        for k in vkeys:
            if self.vertex.get(k) == agent:
                del self.vertex[k]
        for k in ekeys:
            if self.edge.get(k) == agent:
                del self.edge[k]
        if self.parked.get(last, (None,))[0] == agent:
            del self.parked[last]

    @classmethod
    def from_plans(cls, n_cells: int, plans: dict[int, Plan], t_from: int) -> "ReservationBoard":
        b = cls(n_cells)
        for a in sorted(plans):
            b.add(a, plans[a], t_from)
        return b

    def copy(self) -> "ReservationBoard":
        b = ReservationBoard(self.nc)
        b.vertex = dict(self.vertex)
        b.edge = dict(self.edge)
        b.parked = dict(self.parked)
        b._keys = dict(self._keys)
        b.max_t = self.max_t
        return b

    # ------------------------------------------------------------------ query
    def occupant(self, t: int, cell: int) -> int | None:
        a = self.vertex.get(t * self.nc + cell)
        if a is not None:
            return a
        p = self.parked.get(cell)
        if p is not None and t >= p[1]:
            return p[0]
        return None

    def mover(self, t: int, a: int, b: int) -> int | None:
        """Agent moving ``a -> b`` between ``t`` and ``t+1``, if any."""
        return self.edge.get((t * self.nc + a) * self.nc + b)

    @property
    def agents(self) -> list[int]:
        return sorted(self._keys)
