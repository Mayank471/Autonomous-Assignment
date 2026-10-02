"""Grid generator and multi-goal Space-Time A*."""

from collections import deque

import pytest

from warehouse.grid import grid_from_ascii, make_warehouse
from warehouse.obstacles import ObstacleWindows
from warehouse.plan import Plan
from warehouse.reservation import ReservationBoard
from warehouse.scenario import make_instance
from warehouse.stastar import BUDGET, INFEASIBLE, Constraints, Query, advance, search
from warehouse.validate import check_plans


def goals_done(plan: Plan, goals) -> int:
    k = 0
    for c in plan.cells:
        k = advance(goals, k, c)
    return k


# --------------------------------------------------------------------- grid
@pytest.mark.parametrize("preset,shape", [("small", (11, 15)), ("main", (21, 33))])
def test_presets_shape_and_connectivity(preset, shape):
    g = make_warehouse(preset)
    assert (g.height, g.width) == shape
    free = [c for c in range(g.n_cells) if g.free[c]]
    seen = {free[0]}
    q = deque([free[0]])
    while q:
        u = q.popleft()
        for v in g.neighbors[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    assert len(seen) == len(free), "every free cell must be reachable"
    # docks and stations are dead-end pockets
    for c in g.docks + g.stations:
        assert len(g.neighbors[c]) == 1
    assert len(g.docks) >= 50 if preset == "main" else len(g.docks) >= 8
    assert not set(g.blockable) & (g.dock_set | g.station_set)


# ------------------------------------------------------------- st-astar core
def test_cost_equals_static_tour_on_empty_world():
    inst = make_instance(20, seed=3)
    g = inst.grid
    for m in inst.missions:
        q = Query(m.agent, m.dock, 0, tuple(m.goals), m.dock)
        res = search(g, q, obstacles=ObstacleWindows())
        assert res.ok
        assert res.plan.end == g.tour_length(m.dock, m.goals)
        assert goals_done(res.plan, m.goals) == len(m.goals)
        assert res.plan.cells[-1] == m.dock


CORRIDOR = """
#######
#D...S#
###.###
###D###
#######
"""


def _corridor():
    g = grid_from_ascii(CORRIDOR)
    d_left = g.cell(1, 1)
    d_down = g.cell(3, 3)
    station = g.cell(1, 5)
    return g, d_left, d_down, station


def test_vertex_reservation_forces_wait():
    g, d_left, d_down, station = _corridor()
    q = Query(0, d_left, 0, (station, d_left), d_left)
    free = search(g, q, obstacles=ObstacleWindows())
    assert free.plan.end == 8
    board = ReservationBoard(g.n_cells)
    # agent 1 sits at the corridor junction (1,3) from t=1..4, then goes home
    other = Plan(0, (g.cell(2, 3), g.cell(1, 3), g.cell(1, 3), g.cell(1, 3), g.cell(1, 3),
                     g.cell(2, 3), d_down))
    board.add(1, other, 0)
    res = search(g, q, obstacles=ObstacleWindows(), board=board)
    assert res.ok and res.plan.end > 8
    errs = check_plans(g, {0: res.plan, 1: other}, 0, docks={0: d_left, 1: d_down})
    assert errs == []


def test_swap_conflict_is_avoided():
    g, d_left, d_down, station = _corridor()
    board = ReservationBoard(g.n_cells)
    # agent 1 travels station -> left along the corridor
    other = Plan(0, (g.cell(1, 4), g.cell(1, 3), g.cell(2, 3), d_down))
    board.add(1, other, 0)
    q = Query(0, g.cell(1, 2), 0, (station, d_left), d_left)
    res = search(g, q, obstacles=ObstacleWindows(), board=board)
    assert res.ok
    errs = check_plans(g, {0: res.plan, 1: other}, 0, docks={0: d_left, 1: d_down})
    assert errs == []


def test_obstacle_window_wait_and_hold_and_deadline():
    g, d_left, d_down, station = _corridor()
    q = Query(0, d_left, 0, (station, d_left), d_left)
    obs = ObstacleWindows()
    obs.add(g.cell(1, 3), 0, 6)          # junction blocked for t < 6
    res = search(g, q, obstacles=obs)
    assert res.ok
    assert all(res.plan.at(t) != g.cell(1, 3) for t in range(6))
    assert res.plan.end == 8 + (6 - 2)   # reaches the junction at t=6 instead of t=2

    held = search(g, Query(0, d_left, 0, (station, d_left), d_left, hold_until=5),
                  obstacles=ObstacleWindows())
    assert all(held.plan.at(t) == d_left for t in range(6))
    assert held.plan.end == 13

    tight = Query(0, d_left, 0, (station, d_left), d_left, deadline=(0, 3))
    assert search(g, tight, obstacles=ObstacleWindows()).status == INFEASIBLE
    exact = Query(0, d_left, 0, (station, d_left), d_left, deadline=(0, 4))
    assert search(g, exact, obstacles=ObstacleWindows()).ok


def test_soft_agents_are_passable_but_minimised():
    # two equally short routes around a pillar
    g = grid_from_ascii("""
#######
##...##
#D.#.S#
##...##
#######
""")
    dock, station = g.cell(2, 1), g.cell(2, 5)
    board = ReservationBoard(g.n_cells)
    board.add(7, Plan(0, (g.cell(3, 3),)), 0)      # soft agent parked on the lower route
    q = Query(0, dock, 0, (station,), station)
    res = search(g, q, obstacles=ObstacleWindows(), board=board, hard=frozenset())
    assert res.ok and res.plan.end == 6 and res.soft == 0
    assert g.cell(3, 3) not in res.plan.cells
    # cost comes first: with the upper route walled off, the soft agent is crossed
    g2 = grid_from_ascii("""
#######
##.#.##
#D.#.S#
##...##
#######
""")
    b2 = ReservationBoard(g2.n_cells)
    b2.add(7, Plan(0, (g2.cell(3, 3),)), 0)
    q2 = Query(0, g2.cell(2, 1), 0, (g2.cell(2, 5),), g2.cell(2, 5))
    soft = search(g2, q2, obstacles=ObstacleWindows(), board=b2, hard=frozenset())
    assert soft.ok and soft.soft == 1 and soft.plan.end == 6
    assert search(g2, q2, obstacles=ObstacleWindows(), board=b2, hard=None).status == INFEASIBLE


def test_cbs_constraints_and_budget():
    g, d_left, d_down, station = _corridor()
    q = Query(0, d_left, 0, (station, d_left), d_left)
    nc = g.n_cells
    c = Constraints().with_vertex(2 * nc + g.cell(1, 3), 2)
    res = search(g, q, obstacles=ObstacleWindows(), constraints=c)
    assert res.ok and res.plan.at(2) != g.cell(1, 3) and res.plan.end == 9
    assert search(g, q, obstacles=ObstacleWindows(), constraints=c,
                  max_expansions=1).status == BUDGET
