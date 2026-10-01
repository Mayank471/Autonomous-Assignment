"""Emergency response as a target-tracking DCOP (Weiss Ch 12 section 3.1.2).

An emergency task is the warehouse's version of a target appearing in a sensor
field.  The textbook's formulation carries over term for term:

===========================  ==========================================
Target tracking              Warehouse emergency
===========================  ==========================================
sensor                       robot near the emergency
sensing modality             respond, or carry on (``idle``)
overlapping sensing range    both robots can reach the emergency in time
constraint per target        the emergency needs a *pair* to service it
tracking accuracy            emergency serviced
energy cost of aiming        delay to that robot's own deliveries
===========================  ==========================================

so :func:`build_emergency_network` is literally
:func:`~warehouse.dcop.examples.target_tracking_network` with the diversion
costs added as unary terms.  The standalone demo and the live warehouse run
through the same builder and the same ADOPT solver.

Why a *pair*: the textbook's example is "two cameras ... to both focus on a
potentially dangerous target, thus providing a more accurate estimate", and its
binary constraints express exactly "these two together cover it".  An emergency
needing two robots (a spill, a dropped load, a blocked aisle) keeps that
structure intact.  ``required_responders`` is honoured by pricing the
constraint; it is not a cardinality constraint, which a binary network cannot
express directly.

Once the responders are chosen, the emergency is spliced to the front of each
one's task queue and an ordinary repair episode re-plans them -- so an
emergency ends up exercising the same negotiation machinery as any other
disruption, seeded with a different set of agents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .dcop.adopt import solve_adopt
from .dcop.examples import target_tracking_network
from .dcop.model import ConstraintNetwork
from .disruptions import EmergencyTask
from .grid import WarehouseGrid
from .tasks import Task

#: The single "target" every emergency network is built around.
EMERGENCY_TARGET = "emergency"

#: Base id for synthesised emergency tasks, kept clear of generated task ids.
EMERGENCY_TASK_ID_BASE = 1_000_000


@dataclass
class EmergencyAllocation:
    """Who was asked, who agreed, and what the negotiation cost."""

    responders: tuple[int, ...] = ()
    considered: tuple[int, ...] = ()
    diversion_cost: float = 0.0
    messages: int = 0
    cycles: int = 0
    wall_ms: float = 0.0
    solved: bool = True

    @property
    def n_responders(self) -> int:
        return len(self.responders)


def diversion_cost(
    grid: WarehouseGrid,
    position: int,
    emergency_cell: int,
    next_waypoint: int | None,
) -> float:
    """Extra travel an agent takes on by routing via the emergency.

    A Manhattan estimate, deliberately: this prices a *bid*, and the textbook's
    bidders reason from an abstraction rather than a planned route.  The real
    cost is settled later when the agent actually replans.
    """
    if next_waypoint is None:
        return float(grid.manhattan(position, emergency_cell))
    direct = grid.manhattan(position, next_waypoint)
    via = grid.manhattan(position, emergency_cell) + grid.manhattan(
        emergency_cell, next_waypoint
    )
    return float(max(0, via - direct))


def candidate_responders(
    emergency: EmergencyTask,
    grid: WarehouseGrid,
    positions: Mapping[int, int],
    active: Sequence[int],
    *,
    radius: int,
    limit: int = 6,
) -> dict[int, int]:
    """Agents within ``radius`` of the emergency -- the "sensors" that cover it.

    Capped at ``limit`` because a complete DCOP solver is exponential in the
    worst case and an emergency does not need the whole fleet to deliberate.
    """
    in_range = [
        (grid.manhattan(positions[a], emergency.cell), a)
        for a in active
        if a in positions and grid.manhattan(positions[a], emergency.cell) <= radius
    ]
    in_range.sort()
    return {agent: distance for distance, agent in in_range[:limit]}


def build_emergency_network(
    costs: Mapping[int, float], *, reward: float
) -> ConstraintNetwork:
    """The target-tracking DCOP for one emergency.

    ``costs`` maps each candidate responder to its diversion cost.  Every
    candidate covers the single target, so the network is complete over them --
    which is right: any pair of them could be the pair that responds.
    """
    coverage = {agent: (EMERGENCY_TARGET,) for agent in costs}
    network = target_tracking_network(coverage, reward=reward, energy=0.0)
    for agent, cost in costs.items():
        network.add_unary(agent, {"idle": 0.0, EMERGENCY_TARGET: float(cost)})
    return network


def allocate_emergency(
    emergency: EmergencyTask,
    grid: WarehouseGrid,
    positions: Mapping[int, int],
    next_waypoints: Mapping[int, int | None],
    active: Sequence[int],
    *,
    radius: int = 25,
    limit: int = 6,
    reward: float | None = None,
    epsilon: float = 0.0,
) -> EmergencyAllocation:
    """Decide which agents divert, by solving the target-tracking DCOP.

    ``reward`` defaults to comfortably more than the worst diversion on offer,
    so responding is always worth it -- an emergency that nobody answers
    because it was too inconvenient would not be much of an emergency.
    """
    started = time.perf_counter()
    candidates = candidate_responders(
        emergency, grid, positions, active, radius=radius, limit=limit
    )
    if len(candidates) < 2:
        # Nobody, or only one agent, in range: no pair to negotiate over.
        return EmergencyAllocation(
            responders=tuple(candidates),
            considered=tuple(candidates),
            diversion_cost=float(sum(candidates.values())),
            wall_ms=(time.perf_counter() - started) * 1000.0,
            solved=False,
        )

    costs = {
        agent: diversion_cost(
            grid, positions[agent], emergency.cell, next_waypoints.get(agent)
        )
        for agent in candidates
    }
    if reward is None:
        reward = max(costs.values(), default=0.0) * 2.0 + 10.0

    network = build_emergency_network(costs, reward=reward)
    result = solve_adopt(network, epsilon=epsilon, validate=False)

    responders = tuple(
        sorted(a for a, v in result.assignment.items() if v == EMERGENCY_TARGET)
    )
    return EmergencyAllocation(
        responders=responders,
        considered=tuple(sorted(candidates)),
        diversion_cost=float(sum(costs[a] for a in responders)),
        messages=result.messages,
        cycles=result.cycles,
        wall_ms=(time.perf_counter() - started) * 1000.0,
        solved=result.terminated,
    )


def inject_emergency_task(
    emergency: EmergencyTask,
    agent: int,
    agent_tasks,
    progress: Mapping[int, int],
    sequence: int,
) -> Task:
    """Splice the emergency to the front of ``agent``'s outstanding queue.

    Pickup and delivery are both the emergency cell: the agent goes there and
    services it rather than carrying anything away.  Inserting at the front of
    the *remaining* queue is what makes it high priority -- it is done before
    whatever the agent had lined up next.
    """
    task = Task(
        task_id=EMERGENCY_TASK_ID_BASE + sequence,
        pickup=emergency.cell,
        delivery=emergency.cell,
    )
    tasks = agent_tasks[agent]
    insert_at = min(progress.get(agent, 0), len(tasks.tasks))
    tasks.tasks.insert(insert_at, task)
    return task


def is_emergency_task(task: Task) -> bool:
    return task.task_id >= EMERGENCY_TASK_ID_BASE
