"""Independent feasibility checker for a set of committed plans.

Deliberately shares no code with the reservation board or the search: it just
replays every agent's position step by step and looks for anything physically
wrong.  The simulator runs it after every repair commit, so a planner bug
surfaces at the repair that caused it rather than many steps later.
"""

from __future__ import annotations

from .grid import WarehouseGrid
from .obstacles import ObstacleWindows
from .plan import Plan


class PlanViolation(RuntimeError):
    pass


def check_plans(grid: WarehouseGrid, plans: dict[int, Plan], t_from: int, *,
                docks: dict[int, int], obstacles: ObstacleWindows | None = None,
                limit: int = 20) -> list[str]:
    """Return a list of human-readable violations (empty if the plans are sound)."""
    errs: list[str] = []
    agents = sorted(plans)
    if not agents:
        return errs
    for a in agents:
        if plans[a].t0 > t_from:
            errs.append(f"agent {a}: plan starts at {plans[a].t0} > {t_from}")
    if errs:
        return errs
    t_end = max(plans[a].end for a in agents) + 1
    windows = list(obstacles.items()) if obstacles is not None else []
    all_docks = grid.dock_set

    prev = {a: plans[a].at(t_from) for a in agents}
    for t in range(t_from, t_end + 1):
        pos = {a: plans[a].at(t) for a in agents}
        seen: dict[int, int] = {}
        for a in agents:
            c = pos[a]
            if not grid.free[c]:
                errs.append(f"t={t} agent {a} in wall cell {grid.rc(c)}")
            if c in all_docks and c != docks.get(a):
                errs.append(f"t={t} agent {a} in someone else's dock {grid.rc(c)}")
            if c in seen:
                errs.append(f"t={t} vertex conflict agents {seen[c]},{a} at {grid.rc(c)}")
            seen[c] = a
            if t > t_from:
                p = prev[a]
                if p != c and c not in grid.neighbors[p]:
                    errs.append(f"t={t} agent {a} made an illegal move {grid.rc(p)}->{grid.rc(c)}")
        if t > t_from:
            moved = {(prev[a], pos[a]): a for a in agents if prev[a] != pos[a]}
            for (u, v), a in moved.items():
                b = moved.get((v, u))
                if b is not None and a < b:
                    errs.append(f"t={t} swap conflict agents {a},{b} on {grid.rc(u)}<->{grid.rc(v)}")
        for cell, s, e, owner in windows:
            if s <= t < e:
                if owner >= 0 and owner in pos and pos[owner] != cell:
                    errs.append(f"t={t} frozen agent {owner} left {grid.rc(cell)}")
                occ = seen.get(cell)
                if occ is not None and occ != owner:
                    errs.append(f"t={t} agent {occ} in blocked cell {grid.rc(cell)}")
        if len(errs) >= limit:
            break
        prev = pos
    return errs


def assert_plans(grid: WarehouseGrid, plans: dict[int, Plan], t_from: int, *,
                 docks: dict[int, int], obstacles: ObstacleWindows | None = None,
                 context: str = "") -> None:
    errs = check_plans(grid, plans, t_from, docks=docks, obstacles=obstacles)
    if errs:
        raise PlanViolation(f"{context}: " + "; ".join(errs[:5]))
