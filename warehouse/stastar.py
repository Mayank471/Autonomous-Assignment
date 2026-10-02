"""Multi-goal Space-Time A* -- the single low-level search used everywhere.

State ``(cell, t, k)``: the agent is in ``cell`` at time ``t`` and ``k`` is the
index of the next goal it must visit.  Actions are the four moves plus wait;
every action costs one time step, so the cost of reaching a state is fixed by
its time step (``g = t - t0``).

What a search must respect:

hard
    static walls, other agents' docks, obstacle windows (blockages and frozen
    agents), CBS constraints, an optional *hold* (a broken agent may not move
    before it is repaired), an optional *deadline* on one goal, and the
    reservation-board entries of every agent classified as hard.
soft
    board entries of agents classified as soft are passable, but each collision
    with one counts as a *soft conflict*.

The search returns the cheapest path and, among the cheapest, the one with the
fewest soft conflicts.  This lexicographic order is exact: all paths to a state
share the same ``g``, so the priority ``(f, soft)`` is consistent.

Termination: after ``T_last`` (the latest time any dynamic constraint exists)
the world is static, so a state popped at ``t >= T_last`` is finished with the
static shortest route through its remaining goals, whose cost the heuristic
already gives exactly.  The search is bounded by an expansion count, never by
wall-clock time.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

from .grid import UNREACHABLE, WarehouseGrid
from .obstacles import ObstacleWindows
from .plan import Plan
from .reservation import ReservationBoard

OK, INFEASIBLE, BUDGET = "ok", "infeasible", "budget"
DEFAULT_MAX_EXPANSIONS = 400_000


@dataclass
class Constraints:
    """CBS constraints for one agent (packed keys, see reservation.py)."""

    vertex: frozenset[int] = frozenset()
    edge: frozenset[int] = frozenset()
    max_t: int = 0

    def with_vertex(self, key: int, t: int) -> "Constraints":
        return Constraints(self.vertex | {key}, self.edge, max(self.max_t, t))

    def with_edge(self, key: int, t: int) -> "Constraints":
        return Constraints(self.vertex, self.edge | {key}, max(self.max_t, t + 1))


NO_CONSTRAINTS = Constraints()


@dataclass(frozen=True)
class Query:
    """What one agent needs to plan: where it is, and what it still has to do."""

    agent: int
    start: int
    t0: int
    goals: tuple[int, ...]          # remaining goals; goals[-1] is the dock
    dock: int
    hold_until: int = -1            # must stay at ``start`` for all t <= hold_until
    deadline: tuple[int, int] | None = None   # (index into goals, latest arrival time)


@dataclass
class SearchResult:
    status: str
    plan: Plan | None = None
    soft: int = 0
    expansions: int = 0

    @property
    def ok(self) -> bool:
        return self.status == OK


def advance(goals: tuple[int, ...] | list[int], k: int, cell: int) -> int:
    """Goal index after being in ``cell``: goals are visited strictly in order."""
    n = len(goals)
    while k < n and goals[k] == cell:
        k += 1
    return k


@dataclass
class _Ctx:
    nc: int
    goals: tuple[int, ...]
    dists: list[list[int]]
    rest: list[int]
    prefix: list[int]


def _context(grid: WarehouseGrid, goals: tuple[int, ...]) -> _Ctx:
    dists = [grid.dist_from(g) for g in goals]
    legs = [dists[i + 1][goals[i]] for i in range(len(goals) - 1)]
    rest = [0] * (len(goals) + 1)
    for i in range(len(goals) - 2, -1, -1):
        rest[i] = rest[i + 1] + legs[i]
    prefix = [0] * (len(goals) + 1)
    for i in range(1, len(goals)):
        prefix[i] = prefix[i - 1] + legs[i - 1]
    return _Ctx(grid.n_cells, goals, dists, rest, prefix)


def static_route(grid: WarehouseGrid, cell: int, goals: tuple[int, ...], k: int) -> list[int]:
    """Cells visited after ``cell`` on a static shortest route through
    ``goals[k:]`` (deterministic: lowest distance, then lowest cell id)."""
    out = []
    nbrs = grid.neighbors
    while k < len(goals):
        d = grid.dist_from(goals[k])
        while cell != goals[k]:
            cur = d[cell]
            nxt = min((v for v in nbrs[cell] if d[v] == cur - 1), default=None)
            if nxt is None:
                raise RuntimeError("static route: goal unreachable")
            cell = nxt
            out.append(cell)
        k = advance(goals, k, cell)
    return out


def search(grid: WarehouseGrid, q: Query, *, obstacles: ObstacleWindows,
           board: ReservationBoard | None = None, ignore: frozenset[int] | set[int] = frozenset(),
           hard: frozenset[int] | set[int] | None = None,
           constraints: Constraints = NO_CONSTRAINTS,
           cat: ReservationBoard | None = None,
           max_expansions: int = DEFAULT_MAX_EXPANSIONS) -> SearchResult:
    """Plan ``q.agent`` from ``q.start`` at ``q.t0`` through ``q.goals``.

    Board occupants are classified as: ``q.agent`` or in ``ignore`` -> ignored;
    ``hard is None`` or in ``hard`` -> hard; otherwise soft.

    ``cat`` is an optional conflict-avoidance table (the current paths of the
    other agents being replanned with this one): every collision with it also
    counts as a soft conflict, which steers ties toward paths that leave the
    others alone without ever changing the cost.
    """
    agent, goals, t0 = q.agent, q.goals, q.t0
    G = len(goals)
    if G == 0:
        return SearchResult(OK, Plan(t0, (q.start,)), 0, 0)
    ctx = _context(grid, goals)
    nc_total, dists, rest, prefix = ctx.nc, ctx.dists, ctx.rest, ctx.prefix
    last_d = dists[G - 1]
    if dists[0][q.start] >= UNREACHABLE:
        return SearchResult(INFEASIBLE)

    def h(cell: int, k: int) -> int:
        return dists[k][cell] + rest[k] if k < G else last_d[cell]

    dock_set, own_dock = grid.dock_set, q.dock
    nbrs = grid.neighbors
    vcons, econs = constraints.vertex, constraints.edge
    has_cons = bool(vcons or econs)
    vertex = board.vertex if board is not None else {}
    edge = board.edge if board is not None else {}
    parked = board.parked if board is not None else {}
    hold = q.hold_until
    dl_idx, dl_t = q.deadline if q.deadline is not None else (-1, 0)

    t_last = max(t0, obstacles.last_time(), constraints.max_t, hold,
                 dl_t if q.deadline is not None else 0,
                 board.max_t if board is not None else 0,
                 cat.max_t if cat is not None else 0)
    cat_v = cat.vertex if cat is not None else None
    cat_e = cat.edge if cat is not None else None

    def occupant(t: int, cell: int) -> int | None:
        a = vertex.get(t * nc_total + cell)
        if a is not None:
            return a
        p = parked.get(cell)
        if p is not None and t >= p[1]:
            return p[0]
        return None

    def final_free(cell: int, t: int) -> bool:
        if obstacles.blocked_after(cell, t, agent):
            return False
        if has_cons:
            for tt in range(t + 1, constraints.max_t + 1):
                if tt * nc_total + cell in vcons:
                    return False
        if board is not None:
            for tt in range(t + 1, board.max_t + 1):
                o = occupant(tt, cell)
                if o is not None and o != agent and o not in ignore:
                    return False
        return True

    # Hard agents parked forever outside a dock are permanent obstacles, so the
    # static completion must route around them.  (Never happens in the
    # warehouse, where plans always end in private docks.)
    permanent = frozenset(
        c for c, (a, _) in parked.items()
        if c not in dock_set and a != agent and a not in ignore and (hard is None or a in hard)
    )
    soft_parked = frozenset(
        c for c, (a, _) in parked.items()
        if c not in dock_set and a != agent and a not in ignore and hard is not None and a not in hard
    )

    G1 = G + 1
    k0 = advance(goals, 0, q.start)
    if q.deadline is not None and k0 <= dl_idx and t0 + dists[k0][q.start] + prefix[dl_idx] - prefix[k0] > dl_t:
        return SearchResult(INFEASIBLE)
    start_key = (t0 * nc_total + q.start) * G1 + k0
    parent: dict[int, int] = {start_key: -1}
    best_soft: dict[int, int] = {start_key: 0}
    closed: set[int] = set()
    heap = [(t0 + h(q.start, k0), 0, -t0, start_key)]
    expansions = 0

    heappush, heappop = heapq.heappush, heapq.heappop
    owin = obstacles.windows
    # parked entries that can matter: outside docks (never in the warehouse)
    # or another agent parked in this agent's own dock (impossible, but cheap)
    pk = {c: v for c, v in parked.items()
          if (c not in dock_set or c == own_dock) and v[0] != agent}
    while heap:
        f, s, negt, key = heappop(heap)
        if key in closed:
            continue
        closed.add(key)
        k = key % G1
        rem = key // G1
        cell = rem % nc_total
        t = rem // nc_total

        if k == G and (t >= t_last or final_free(cell, t)):
            return SearchResult(OK, _build(parent, key, G1, nc_total, t0, []), s, expansions)
        if t >= t_last:
            if permanent:
                tail = _route_avoiding(grid, cell, goals, k, permanent)
                if tail is None:
                    continue        # this branch can never finish; try others
            else:
                tail = static_route(grid, cell, goals, k)
            if soft_parked:
                s += sum(1 for c in tail if c in soft_parked)
            return SearchResult(OK, _build(parent, key, G1, nc_total, t0, tail), s, expansions)

        expansions += 1
        if expansions > max_expansions:
            return SearchResult(BUDGET, None, 0, expansions)

        nt = t + 1
        moves = (cell,) if t < hold else (cell,) + nbrs[cell]
        for ncell in moves:
            if ncell != cell and ncell in dock_set and ncell != own_dock:
                continue
            ws = owin.get(ncell)
            if ws is not None:
                hit = False
                for ws_s, ws_e, ws_o in ws:
                    if ws_s <= nt < ws_e and ws_o != agent:
                        hit = True
                        break
                if hit:
                    continue
            if has_cons:
                if nt * nc_total + ncell in vcons:
                    continue
                if ncell != cell and (t * nc_total + cell) * nc_total + ncell in econs:
                    continue
            ns = s
            o = vertex.get(nt * nc_total + ncell)
            if o is None and pk:
                p = pk.get(ncell)
                if p is not None and nt >= p[1]:
                    o = p[0]
            if o is not None and o != agent and o not in ignore:
                if hard is None or o in hard:
                    continue
                ns += 1
            if ncell != cell:
                o = edge.get((t * nc_total + ncell) * nc_total + cell)
                if o is not None and o != agent and o not in ignore:
                    if hard is None or o in hard:
                        continue
                    ns += 1
            if cat_v is not None:
                o = cat_v.get(nt * nc_total + ncell)
                if o is not None and o != agent:
                    ns += 1
                if ncell != cell:
                    o = cat_e.get((t * nc_total + ncell) * nc_total + cell)
                    if o is not None and o != agent:
                        ns += 1
            nk = k
            if k < G and goals[k] == ncell:
                nk = advance(goals, k, ncell)
            if dl_idx >= 0 and nk <= dl_idx:
                if nt + dists[nk][ncell] + prefix[dl_idx] - prefix[nk] > dl_t:
                    continue
            nkey = (nt * nc_total + ncell) * G1 + nk
            if nkey in closed:
                continue
            prev = best_soft.get(nkey)
            if prev is not None and prev <= ns:
                continue
            best_soft[nkey] = ns
            parent[nkey] = key
            hv = dists[nk][ncell] + rest[nk] if nk < G else last_d[ncell]
            heappush(heap, (nt + hv, ns, -nt, nkey))

    return SearchResult(INFEASIBLE, None, 0, expansions)


def _route_avoiding(grid: WarehouseGrid, cell: int, goals: tuple[int, ...], k: int,
                    avoid: frozenset[int]) -> list[int] | None:
    """Like :func:`static_route` but never entering ``avoid``; ``None`` if impossible."""
    out = []
    while k < len(goals):
        target = goals[k]
        if target in avoid:
            return None
        prev = {cell: -1}
        q = deque([cell])
        while q and target not in prev:
            u = q.popleft()
            for v in grid.neighbors[u]:
                if v not in prev and v not in avoid:
                    prev[v] = u
                    q.append(v)
        if target not in prev:
            return None
        leg = []
        v = target
        while v != cell:
            leg.append(v)
            v = prev[v]
        out += reversed(leg)
        cell = target
        k = advance(goals, k, cell)
    return out


def _build(parent: dict[int, int], key: int, G1: int, nc: int, t0: int, tail: list[int]) -> Plan:
    cells = []
    while key != -1:
        cells.append((key // G1) % nc)
        key = parent[key]
    cells.reverse()
    return Plan(t0, tuple(cells + tail))
