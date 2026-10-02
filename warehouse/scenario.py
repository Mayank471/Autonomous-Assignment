"""Instance generation: one reproducible warehouse problem per seed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import WarehouseGrid, make_warehouse
from .tasks import Mission, Task, mission_from_tasks


@dataclass
class Instance:
    grid: WarehouseGrid
    missions: list[Mission]
    seed: int

    @property
    def n_agents(self) -> int:
        return len(self.missions)

    @property
    def docks(self) -> list[int]:
        return [m.dock for m in self.missions]


def rng_for(seed: int, *stream: int) -> np.random.Generator:
    """Independent random stream per (seed, component), so adding randomness to
    one component never shifts another."""
    return np.random.default_rng(np.random.SeedSequence([seed, *stream]))


STREAM_TASKS, STREAM_EVENTS, STREAM_SINGLE = 1, 2, 3


def make_instance(n_agents: int, seed: int, *, preset: str = "main", n_tasks: int = 3,
                  grid: WarehouseGrid | None = None) -> Instance:
    grid = grid or make_warehouse(preset)
    rng = rng_for(seed, STREAM_TASKS)
    if n_agents > len(grid.docks):
        raise ValueError(f"{n_agents} agents but only {len(grid.docks)} docks")
    if n_agents * n_tasks > len(grid.pickups):
        raise ValueError("not enough distinct pickup points")
    docks = sorted(int(d) for d in rng.choice(grid.docks, size=n_agents, replace=False))
    pickups = [int(p) for p in rng.choice(grid.pickups, size=n_agents * n_tasks, replace=False)]
    deliveries = [int(s) for s in rng.choice(grid.stations, size=n_agents * n_tasks, replace=True)]
    missions = []
    for a in range(n_agents):
        tasks = [Task(a * n_tasks + k, pickups[a * n_tasks + k], deliveries[a * n_tasks + k])
                 for k in range(n_tasks)]
        missions.append(mission_from_tasks(grid, a, docks[a], tasks))
    return Instance(grid, missions, seed)
