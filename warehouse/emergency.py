"""High-priority emergency tasks.

An emergency gives agent ``e`` an urgent pickup -> delivery that goes to the
front of its goal list.  Its deadline is the earliest time ``e`` could possibly
deliver -- planning only around obstacles, as if every other agent made way --
plus a slack ``delta`` (0 = a true emergency).  The deadline stays attached to
the agent, and binds every later repair, until the delivery is made.
"""

from __future__ import annotations

from .grid import WarehouseGrid
from .obstacles import ObstacleWindows
from .stastar import Query, search
from .tasks import E_DELIVERY, E_PICKUP, Mission


def earliest_delivery(grid: WarehouseGrid, agent: int, start: int, t: int, pickup: int,
                      station: int, dock: int, obstacles: ObstacleWindows,
                      hold_until: int = -1) -> int | None:
    q = Query(agent, start, t, (pickup, station), dock, hold_until)
    res = search(grid, q, obstacles=obstacles)
    return res.plan.end if res.ok else None


def insert_emergency(mission: Mission, progress: int, pickup: int, station: int) -> int:
    """Insert the urgent task before the agent's next goal.  Returns the
    absolute goal index of the emergency delivery."""
    mission.goals[progress:progress] = [pickup, station]
    mission.kinds[progress:progress] = [E_PICKUP, E_DELIVERY]
    return progress + 1
