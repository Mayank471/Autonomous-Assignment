"""Space-Time A* -- the single-agent low-level search.

Search states are ``(cell, time)`` and the action set is the four grid moves
plus *wait*.  Other agents enter only through the reservation table, so the
same routine serves three roles unchanged:

* the independent local plan of Ch 11 section 4 (empty reservation table);
* one agent's turn in prioritized planning (higher-priority agents reserved);
* a candidate repair plan (everyone else reserved, blocked cells added).

The heuristic is the true distance to the goal over the static floor, found by
one backward BFS per goal and cached, which keeps it admissible and consistent.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Callable, Iterable, Sequence

from .grid import WarehouseGrid
from .obstacles import DynamicObstacles
from .plan import Plan
from .reservation import ReservationTable

INF = float("inf")

#: Hard cap on node expansions per search.  A bound is needed because a
#: space-time search over a congested floor can otherwise wander for a very
#: long time before proving failure; hitting it is reported as "no path".
DEFAULT_MAX_EXPANSIONS = 150_000


class HeuristicCache:
    """Backward-BFS distance-to-goal, cached per goal cell.

    Distances ignore *other agents* (they are time-varying) but do respect
    permanently blocked cells, which keeps the estimate admissible while making
    it far more informed once the floor starts filling with obstacles.

    A ``move_filter`` (social laws) makes the movement graph directed, so the
    backward BFS traverses each edge in reverse: cell ``u`` is a predecessor of
    ``v`` only when ``u -> v`` is actually permitted.  That yields exact
    directed distances rather than an optimistic undirected estimate.
    """

    __slots__ = ("_grid", "_obstacles", "_cache", "_version", "_move_filter")

    def __init__(
        self,
        grid: WarehouseGrid,
        obstacles: DynamicObstacles | None = None,
        move_filter: Callable[[int, int], bool] | None = None,
    ):
        self._grid = grid
        self._obstacles = obstacles if obstacles is not None else DynamicObstacles()
        self._cache: dict[int, list[int]] = {}
        self._version = self._obstacles.version
        self._move_filter = move_filter

    def _blocked(self) -> set[int]:
        return self._obstacles.all_cells()

    def _invalidate_if_stale(self) -> None:
        if self._obstacles.version != self._version:
            self._cache.clear()
            self._version = self._obstacles.version

    def distances(self, goal: int) -> list[int]:
        """Distance from every cell to ``goal``; ``-1`` where unreachable."""
        self._invalidate_if_stale()
        dist = self._cache.get(goal)
        if dist is not None:
            return dist

        grid = self._grid
        blocked = self._blocked()
        dist = [-1] * grid.n_cells
        if goal in blocked or not grid.is_free(goal):
            # Goal itself unusable: leave everything unreachable.
            self._cache[goal] = dist
            return dist

        dist[goal] = 0
        queue = deque([goal])
        allows = self._move_filter
        while queue:
            cell = queue.popleft()
            d = dist[cell] + 1
            for nxt in grid.neighbors(cell):
                if dist[nxt] != -1 or nxt in blocked:
                    continue
                # Backward search: keep ``nxt`` only if ``nxt -> cell`` is legal.
                if allows is not None and not allows(nxt, cell):
                    continue
                dist[nxt] = d
                queue.append(nxt)
        self._cache[goal] = dist
        return dist

    def distance(self, goal: int, cell: int) -> float:
        d = self.distances(goal)[cell]
        return INF if d < 0 else d

    def clear(self) -> None:
        self._cache.clear()


def space_time_astar(
    grid: WarehouseGrid,
    start: int,
    goal: int,
    start_time: int,
    agent: int,
    reservations: ReservationTable,
    obstacles: DynamicObstacles,
    heuristics: HeuristicCache,
    *,
    hold: float = 0,
    avoid: frozenset[int] | set[int] | None = None,
    avoid_penalty: float = 0.0,
    max_time: int | None = None,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    move_filter: Callable[[int, int], bool] | None = None,
    blocked_vertices: frozenset[tuple[int, int]] | None = None,
    blocked_edges: frozenset[tuple[int, int, int]] | None = None,
) -> Plan | None:
    """Find a minimum-cost space-time path from ``start`` to ``goal``.

    Args:
        hold: how long the agent must be able to *stay* on the goal after
            arriving.  ``0`` for a waypoint it passes through, a positive
            integer for one it services, ``inf`` for its final destination
            (which makes the cell a permanent reservation).
        avoid: cells that cost ``avoid_penalty`` extra per step.  Used to coax
            the search into producing a genuinely different route when
            generating alternative repair candidates; the path stays valid and
            the heuristic stays admissible because the penalty is non-negative.
        max_time: latest time step the search may reach.  Defaults to a bound
            that lets an agent outwait every other committed plan.
        move_filter: optional ``(u, v) -> bool`` convention, e.g. the one-way
            aisle social law.  Must match the one the heuristic was built with.
        blocked_vertices: extra ``(cell, t)`` pairs this agent may not occupy.
        blocked_edges: extra ``(u, v, t)`` transitions this agent may not make.

    ``blocked_vertices`` and ``blocked_edges`` carry constraints that belong to
    *this agent alone* rather than to the shared reservation table, which is what
    Conflict-Based Search needs: its high level imposes a constraint on one agent
    and replans only that agent, leaving everyone else's plan untouched.

    Returns:
        A :class:`Plan` starting at ``start_time``, or ``None`` if no path
        exists within the bounds.
    """
    dist = heuristics.distances(goal)
    h0 = dist[start]
    if h0 < 0:
        return None  # goal unreachable on the static floor

    if max_time is None:
        slack = 2 * (grid.n_rows + grid.n_cols)
        max_time = max(start_time, reservations.horizon) + h0 + slack

    avoid = avoid or frozenset()
    finite_hold = hold if hold != INF else None

    # Priority queue of (f, h, counter, cell, time, g).  The counter breaks ties
    # deterministically so runs are reproducible from the seed alone.
    counter = 0
    open_heap: list[tuple[float, int, int, int, int, float]] = [
        (float(h0), h0, counter, start, start_time, 0.0)
    ]
    best_g: dict[tuple[int, int], float] = {(start, start_time): 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    expansions = 0

    while open_heap:
        f, _h, _c, cell, t, g = heapq.heappop(open_heap)
        if g > best_g.get((cell, t), INF):
            continue  # stale heap entry

        expansions += 1
        if expansions > max_expansions:
            return None

        if cell == goal and _can_hold(
            goal, t, finite_hold, reservations, obstacles, agent,
            blocked_vertices=blocked_vertices,
        ):
            return _reconstruct(agent, start_time, parent, (cell, t))

        nt = t + 1
        if nt > max_time:
            continue

        for nxt in (cell, *grid.neighbors(cell)):
            if nxt != cell and move_filter is not None and not move_filter(cell, nxt):
                continue
            if obstacles.is_blocked(nxt, nt):
                continue
            if not reservations.is_vertex_free(nxt, nt, agent):
                continue
            if nxt != cell and not reservations.is_edge_free(cell, nxt, t, agent):
                continue
            if blocked_vertices is not None and (nxt, nt) in blocked_vertices:
                continue
            if blocked_edges is not None and (cell, nxt, t) in blocked_edges:
                continue
            hn = dist[nxt]
            if hn < 0:
                continue

            ng = g + 1.0 + (avoid_penalty if nxt in avoid else 0.0)
            key = (nxt, nt)
            if ng >= best_g.get(key, INF):
                continue
            best_g[key] = ng
            parent[key] = (cell, t)
            counter += 1
            heapq.heappush(open_heap, (ng + hn, hn, counter, nxt, nt, ng))

    return None


def _can_hold(
    goal: int,
    t: int,
    hold: int | None,
    reservations: ReservationTable,
    obstacles: DynamicObstacles,
    agent: int,
    *,
    blocked_vertices: frozenset[tuple[int, int]] | None = None,
) -> bool:
    """Whether ``agent`` may occupy ``goal`` from ``t`` for the required duration.

    ``hold is None`` means *forever*, which additionally requires that no other
    agent ever passes through the cell afterwards.
    """
    if hold is None:
        if obstacles.blocked_from(goal) != INF:
            return False
        if blocked_vertices:
            # Parking here forever must not violate a constraint at any later
            # time, so no constraint on this cell may sit at or after arrival.
            if any(c == goal and ct >= t for c, ct in blocked_vertices):
                return False
        perm = reservations.permanent_owner(goal)
        if perm is not None and perm[1] != agent:
            return False
        return t >= reservations.last_use(goal)

    for k in range(hold + 1):
        if obstacles.is_blocked(goal, t + k):
            return False
        if not reservations.is_vertex_free(goal, t + k, agent):
            return False
        if blocked_vertices is not None and (goal, t + k) in blocked_vertices:
            return False
    return True


def _reconstruct(
    agent: int,
    start_time: int,
    parent: dict[tuple[int, int], tuple[int, int]],
    end: tuple[int, int],
) -> Plan:
    cells: list[int] = []
    node = end
    while node in parent:
        cells.append(node[0])
        node = parent[node]
    cells.append(node[0])
    cells.reverse()
    return Plan(agent, start_time, tuple(cells))


def plan_route(
    grid: WarehouseGrid,
    start: int,
    waypoints: Sequence[int],
    start_time: int,
    agent: int,
    reservations: ReservationTable,
    obstacles: DynamicObstacles,
    heuristics: HeuristicCache,
    *,
    service_time: int = 1,
    avoid: frozenset[int] | set[int] | None = None,
    avoid_penalty: float = 0.0,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
    move_filter: Callable[[int, int], bool] | None = None,
    blocked_vertices: frozenset[tuple[int, int]] | None = None,
    blocked_edges: frozenset[tuple[int, int, int]] | None = None,
) -> Plan | None:
    """Plan a multi-leg route through ``waypoints``, servicing each in turn.

    Each waypoint is held for ``service_time`` steps (the pick or the drop);
    the final waypoint is held forever, so the agent parks there.  Legs are
    planned in sequence with time carried forward -- the standard way to handle
    a task sequence with Space-Time A* without blowing up the state space.
    """
    if not waypoints:
        return Plan(agent, start_time, (start,))

    cells: list[int] = [start]
    current = start
    t = start_time
    last = len(waypoints) - 1

    for i, waypoint in enumerate(waypoints):
        hold: float = INF if i == last else service_time
        leg = space_time_astar(
            grid,
            current,
            waypoint,
            t,
            agent,
            reservations,
            obstacles,
            heuristics,
            hold=hold,
            avoid=avoid,
            avoid_penalty=avoid_penalty,
            max_expansions=max_expansions,
            move_filter=move_filter,
            blocked_vertices=blocked_vertices,
            blocked_edges=blocked_edges,
        )
        if leg is None:
            return None

        cells.extend(leg.cells[1:])
        t = leg.end_time
        current = waypoint

        if i != last:
            # Service the waypoint: stand still while picking or dropping.
            for _ in range(service_time):
                cells.append(waypoint)
                t += 1

    return Plan(agent, start_time, tuple(cells))


def path_cost(plan: Plan) -> int:
    """Time steps the agent spends executing ``plan``."""
    return plan.duration


def plans_conflict(a: Plan, b: Plan) -> bool:
    """Whether two plans collide in space-time (vertex or edge/swap).

    Both are treated as parking on their final cell forever, matching how the
    reservation table interprets them.
    """
    if a.agent == b.agent:
        return False

    lo = min(a.start_time, b.start_time)
    hi = max(a.end_time, b.end_time)

    prev_a = a.at(lo)
    prev_b = b.at(lo)
    if prev_a == prev_b:
        return True
    for t in range(lo + 1, hi + 1):
        cur_a = a.at(t)
        cur_b = b.at(t)
        if cur_a == cur_b:
            return True
        if cur_a == prev_b and cur_b == prev_a:
            return True  # head-on swap
        prev_a, prev_b = cur_a, cur_b
    return False
