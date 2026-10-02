"""Independent checker and prioritized (Cooperative A*) initial planning."""

import pytest

from warehouse.grid import grid_from_ascii
from warehouse.obstacles import ObstacleWindows
from warehouse.plan import Plan
from warehouse.planner import prioritized
from warehouse.scenario import make_instance
from warehouse.stastar import Query, advance
from warehouse.validate import check_plans

LANE = """
#######
#D...D#
#######
"""


def _lane():
    g = grid_from_ascii(LANE)
    return g, g.cell(1, 1), g.cell(1, 5)


def test_checker_accepts_sound_plans():
    g, a, b = _lane()
    plans = {0: Plan(0, (a, a + 1, a + 2)), 1: Plan(0, (b,))}
    assert check_plans(g, plans, 0, docks={0: a, 1: b}) == []


@pytest.mark.parametrize("kind", ["vertex", "swap", "jump", "dock", "blocked", "frozen"])
def test_checker_catches_injected_violations(kind):
    g, a, b = _lane()
    c1, c2, c3 = a + 1, a + 2, a + 3
    docks = {0: a, 1: b}
    obs = None
    if kind == "vertex":
        plans = {0: Plan(0, (c1, c2)), 1: Plan(0, (c3, c2))}
    elif kind == "swap":
        plans = {0: Plan(0, (c1, c2)), 1: Plan(0, (c2, c1))}
    elif kind == "jump":
        plans = {0: Plan(0, (c1, c3)), 1: Plan(0, (b,))}
    elif kind == "dock":
        plans = {0: Plan(0, (c3, b)), 1: Plan(0, (c1,))}
    elif kind == "blocked":
        obs = ObstacleWindows()
        obs.add(c2, 1, 5)
        plans = {0: Plan(0, (c1, c2)), 1: Plan(0, (b,))}
    else:  # frozen agent moves during its window
        obs = ObstacleWindows()
        obs.add(c1, 0, 4, owner=0)
        plans = {0: Plan(0, (c1, c2)), 1: Plan(0, (b,))}
    assert check_plans(g, plans, 0, docks=docks, obstacles=obs) != []


@pytest.mark.parametrize("n_agents,seed", [(10, 0), (30, 1), (50, 2)])
def test_initial_plan_is_collision_free_and_complete(n_agents, seed):
    inst = make_instance(n_agents, seed)
    g = inst.grid
    queries = [Query(m.agent, m.dock, 0, tuple(m.goals), m.dock) for m in inst.missions]
    out = prioritized(g, queries, obstacles=ObstacleWindows())
    assert out.ok
    docks = {m.agent: m.dock for m in inst.missions}
    assert check_plans(g, out.plans, 0, docks=docks) == []
    for m in inst.missions:
        p = out.plans[m.agent]
        k = 0
        for c in p.cells:
            k = advance(m.goals, k, c)
        assert k == len(m.goals)
        assert p.cells[-1] == m.dock
