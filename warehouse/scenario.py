"""Scenario construction -- one place that assembles a reproducible run.

Two jobs:

* build a scenario deterministically from a seed, so a configuration can be
  replayed exactly;
* cache the *initial* joint plan per ``(layout, n_agents, seed, social_laws)``.

The cache is what makes the full sweep tractable.  Initial plan generation is
by far the most expensive step, and it does not depend on the obstacle density
or the repair strategy -- those only affect what happens *after* execution
starts.  So one plan is computed and then replayed across every
density x strategy combination, which is also the reason those combinations are
directly comparable: they are literally the same starting point.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .disruptions import Disruption, generate_disruptions
from .grid import WarehouseGrid
from .obstacles import DynamicObstacles
from .planner import MapfSolution, plan_initial
from .sociallaws import SocialLaws
from .stastar import HeuristicCache
from .tasks import AgentTasks, build_scenario


@dataclass
class Scenario:
    """A fully specified, reproducible run before any disruption is applied."""

    grid: WarehouseGrid
    agent_tasks: dict[int, AgentTasks]
    solution: MapfSolution
    laws: SocialLaws
    seed: int
    n_agents: int
    tasks_per_agent: int

    @property
    def move_filter(self) -> Callable[[int, int], bool] | None:
        return self.laws.as_filter()

    def disruptions(self, density: float, seed: int | None = None) -> list[Disruption]:
        """Draw a disruption schedule at the given obstacle density."""
        return generate_disruptions(
            self.grid,
            self.solution.plans,
            density=density,
            horizon=self.solution.makespan,
            rng=random.Random((seed if seed is not None else self.seed) * 7919 + 13),
        )

    def fresh_tasks(self) -> dict[int, AgentTasks]:
        """A private copy of the task queues, since repair mutates them."""
        return {
            a: AgentTasks(a, t.start, list(t.tasks), t.home)
            for a, t in self.agent_tasks.items()
        }


_CACHE: dict[tuple, Scenario] = {}


def make_grid(
    n_rows: int = 50, n_cols: int = 60, **kwargs
) -> WarehouseGrid:
    return WarehouseGrid.kiva(n_rows=n_rows, n_cols=n_cols, **kwargs)


def build(
    grid: WarehouseGrid,
    n_agents: int,
    seed: int,
    *,
    tasks_per_agent: int = 3,
    social_laws: bool = False,
    use_cache: bool = True,
) -> Scenario:
    """Build (or reuse) the scenario and its initial joint plan."""
    key = (id(grid), n_agents, seed, tasks_per_agent, social_laws)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    _, agent_tasks = build_scenario(grid, n_agents, tasks_per_agent, seed)
    laws = SocialLaws(grid, enabled=social_laws)
    move_filter = laws.as_filter()

    obstacles = DynamicObstacles()
    heuristics = HeuristicCache(grid, obstacles, move_filter=move_filter)
    solution = plan_initial(
        grid,
        agent_tasks,
        obstacles,
        heuristics,
        rng=random.Random(seed),
        move_filter=move_filter,
    )

    scenario = Scenario(
        grid=grid,
        agent_tasks=agent_tasks,
        solution=solution,
        laws=laws,
        seed=seed,
        n_agents=n_agents,
        tasks_per_agent=tasks_per_agent,
    )
    if use_cache:
        _CACHE[key] = scenario
    return scenario


def clear_cache() -> None:
    _CACHE.clear()
