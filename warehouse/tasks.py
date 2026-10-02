"""Pickup-and-delivery tasks and each agent's mission.

An agent's mission is a fixed sequence of goal cells visited in order::

    dock -> p1 -> d1 -> p2 -> d2 -> ... -> pm -> dm -> dock

Capacity is one object, so pickups and deliveries alternate.  The agent chooses
the order of its own tasks by brute force over all m! orders, using static
distances (m is small, 3 by default).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .grid import WarehouseGrid

PICKUP, DELIVERY, DOCK = "pickup", "delivery", "dock"
E_PICKUP, E_DELIVERY = "emergency_pickup", "emergency_delivery"


@dataclass(frozen=True)
class Task:
    tid: int
    pickup: int
    delivery: int


@dataclass
class Mission:
    """The goal sequence of one agent.  ``goals[-1]`` is always its dock."""

    agent: int
    dock: int
    goals: list[int]
    kinds: list[str]

    def copy(self) -> "Mission":
        return Mission(self.agent, self.dock, list(self.goals), list(self.kinds))


def best_order(grid: WarehouseGrid, dock: int, tasks: list[Task]) -> list[Task]:
    """The task order minimising the static tour length (ties: lexicographic)."""
    best, best_len = None, None
    for perm in itertools.permutations(tasks):
        cur, length = dock, 0
        for t in perm:
            length += grid.dist(cur, t.pickup) + grid.dist(t.pickup, t.delivery)
            cur = t.delivery
        length += grid.dist(cur, dock)
        if best_len is None or length < best_len:
            best, best_len = list(perm), length
    return best or []


def mission_from_tasks(grid: WarehouseGrid, agent: int, dock: int, tasks: list[Task]) -> Mission:
    goals, kinds = [], []
    for t in best_order(grid, dock, tasks):
        goals += [t.pickup, t.delivery]
        kinds += [PICKUP, DELIVERY]
    goals.append(dock)
    kinds.append(DOCK)
    return Mission(agent, dock, goals, kinds)
