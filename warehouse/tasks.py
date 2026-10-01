"""Pickup-and-delivery tasks and their assignment to agents.

Each agent is "assigned a set of objects to be picked up and carried to their
respective delivery locations", so an agent's route is the waypoint sequence
``pickup_1, delivery_1, pickup_2, delivery_2, ...``.

Assignment here is a greedy nearest-insertion heuristic.  It is deliberately
simple because task allocation is not what the assignment is measuring -- but
the same marginal-cost pricing reappears in :mod:`warehouse.contractnet`, where
a broken agent's remaining tasks are re-auctioned during recovery.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .grid import WarehouseGrid


@dataclass(frozen=True)
class Task:
    """One object to collect and deliver."""

    task_id: int
    pickup: int
    delivery: int

    def waypoints(self) -> tuple[int, int]:
        return (self.pickup, self.delivery)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"Task({self.task_id}: {self.pickup}->{self.delivery})"


@dataclass
class AgentTasks:
    """An agent's ordered task queue and the waypoints it implies.

    The route always ends at the agent's own ``home`` dock rather than at its
    last delivery station.  This is not cosmetic: an agent that finishes a route
    stays on its final cell for the rest of the run, so that cell becomes a
    *permanent* reservation.  There are far fewer delivery stations than agents,
    so parking on one would make it unreachable for every agent routed there
    afterwards -- they could never complete.  Docks are distinct by
    construction, which removes the contention entirely and matches how real
    fleets return to a charging bay.
    """

    agent: int
    start: int
    tasks: list[Task] = field(default_factory=list)
    home: int | None = None

    def __post_init__(self) -> None:
        if self.home is None:
            self.home = self.start

    def waypoints(self) -> tuple[int, ...]:
        out: list[int] = []
        for task in self.tasks:
            out.append(task.pickup)
            out.append(task.delivery)
        out.append(self.home)
        return tuple(out)

    def remaining_from(self, task_index: int, stage: int = 0) -> tuple[int, ...]:
        """Waypoints still outstanding once ``task_index`` tasks are complete.

        Args:
            stage: ``0`` if the agent still has to collect the current task's
                object, ``1`` if it is already carrying it.  Getting this wrong
                would send a loaded agent back to the shelf it just visited, so
                a repair mid-delivery must pass the agent's real stage.
        """
        out: list[int] = []
        for offset, task in enumerate(self.tasks[task_index:]):
            if not (offset == 0 and stage == 1):
                out.append(task.pickup)
            out.append(task.delivery)
        out.append(self.home)
        return tuple(out)

    @property
    def n_waypoints(self) -> int:
        return 2 * len(self.tasks) + 1

    def __len__(self) -> int:
        return len(self.tasks)


def generate_tasks(
    grid: WarehouseGrid, n_tasks: int, rng: random.Random
) -> list[Task]:
    """Sample ``n_tasks`` pickup/delivery pairs from the floor's stations."""
    if not grid.pickup_points or not grid.delivery_points:
        raise ValueError("grid has no pickup or delivery points")

    return [
        Task(
            task_id=i,
            pickup=rng.choice(grid.pickup_points),
            delivery=rng.choice(grid.delivery_points),
        )
        for i in range(n_tasks)
    ]


def _floor_stays_connected(grid: WarehouseGrid, docks: Sequence[int]) -> bool:
    """Whether the floor minus ``docks`` is still connected and reaches every dock.

    Docks are dedicated to their owner for all time, so they are walls to
    everyone else.  Punching enough of them into a warehouse of single-width
    aisles can sever it, which would make the instance unsolvable for reasons
    that have nothing to do with plan repair.
    """
    blocked = set(docks)
    open_cells = [c for c in grid.free_cells if c not in blocked]
    if not open_cells:
        return False

    seen = {open_cells[0]}
    queue = deque(seen)
    while queue:
        cell = queue.popleft()
        for nxt in grid.neighbors(cell):
            if nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    if len(seen) != len(open_cells):
        return False

    # Every agent must be able to leave and re-enter its own dock.
    return all(any(n in seen for n in grid.neighbors(d)) for d in docks)


def sample_start_positions(
    grid: WarehouseGrid,
    n_agents: int,
    rng: random.Random,
    *,
    max_attempts: int = 40,
) -> list[int]:
    """Distinct dock cells that leave the floor connected.

    Agents dock on open floor rather than in the aisles they will later need to
    reach, which keeps the instance solvable at high agent counts.  Because a
    dock is impassable to every other agent, the sample is rejected and redrawn
    if it would cut the floor in two.
    """
    pool = list(grid.dock_points)
    if len(pool) < n_agents:
        pickups = set(grid.pickup_points)
        pool = [c for c in grid.free_cells if c not in pickups]
    if len(pool) < n_agents:
        raise ValueError(f"only {len(pool)} candidate cells for {n_agents} agents")

    chosen: list[int] = []
    for _ in range(max_attempts):
        chosen = rng.sample(pool, n_agents)
        if _floor_stays_connected(grid, chosen):
            return chosen
    raise ValueError(
        f"could not place {n_agents} docks without disconnecting the floor "
        f"after {max_attempts} attempts"
    )


def assign_tasks(
    grid: WarehouseGrid,
    starts: Sequence[int],
    tasks: Iterable[Task],
    *,
    rng: random.Random | None = None,
) -> dict[int, AgentTasks]:
    """Distribute ``tasks`` over agents by greedy nearest insertion.

    Agents take turns claiming the outstanding task whose pickup is closest to
    where the previous one left them, measured by Manhattan distance.  Taking
    turns keeps the workload balanced, which matters because the headline metric
    sums completion times over every agent.
    """
    pending = list(tasks)
    assignment = {
        agent: AgentTasks(agent=agent, start=start) for agent, start in enumerate(starts)
    }
    cursor = {agent: start for agent, start in enumerate(starts)}

    order = list(assignment)
    while pending:
        for agent in order:
            if not pending:
                break
            here = cursor[agent]
            best = min(pending, key=lambda t: grid.manhattan(here, t.pickup))
            pending.remove(best)
            assignment[agent].tasks.append(best)
            cursor[agent] = best.delivery

    return assignment


def build_scenario(
    grid: WarehouseGrid,
    n_agents: int,
    tasks_per_agent: int,
    seed: int,
) -> tuple[list[int], dict[int, AgentTasks]]:
    """Generate a reproducible scenario: start cells plus an assigned task queue."""
    rng = random.Random(seed)
    starts = sample_start_positions(grid, n_agents, rng)
    tasks = generate_tasks(grid, n_agents * tasks_per_agent, rng)
    return starts, assign_tasks(grid, starts, tasks, rng=rng)
