"""Space-Time A*, the reservation table, and the social-law convention."""

from __future__ import annotations

import pytest

from warehouse.grid import WarehouseGrid
from warehouse.obstacles import DynamicObstacles
from warehouse.plan import Plan
from warehouse.reservation import ReservationTable
from warehouse.sociallaws import SocialLaws, check_connectivity
from warehouse.stastar import (
    HeuristicCache,
    plan_route,
    plans_conflict,
    space_time_astar,
)

# A small open floor with one shelf block, used where a hand-drawn layout is
# clearer than a generated one.
SMALL = """
.......
.#####.
.......
.#####.
.......
"""


@pytest.fixture
def grid():
    return WarehouseGrid.from_ascii(SMALL, zone_shape=(2, 3))


@pytest.fixture
def parts(grid):
    obstacles = DynamicObstacles()
    return grid, ReservationTable(), obstacles, HeuristicCache(grid, obstacles)


# ------------------------------------------------------------------ basics


def test_finds_the_shortest_path_on_an_empty_floor(parts):
    grid, table, obstacles, heur = parts
    start, goal = grid.cell(0, 0), grid.cell(0, 6)

    plan = space_time_astar(grid, start, goal, 0, 0, table, obstacles, heur, hold=0)

    assert plan is not None
    assert plan.cells[0] == start and plan.goal == goal
    assert plan.duration == 6  # straight along the top row
    assert len(plan.cells) == 7


def test_path_steps_are_adjacent_or_waits(parts):
    grid, table, obstacles, heur = parts
    plan = space_time_astar(
        grid, grid.cell(0, 0), grid.cell(4, 6), 0, 0, table, obstacles, heur
    )
    assert plan is not None
    for a, b in zip(plan.cells, plan.cells[1:]):
        assert a == b or b in grid.neighbors(a)


def test_routes_around_a_shelf(parts):
    grid, table, obstacles, heur = parts
    # (2,3) sits between two shelf rows, reachable only along row 2.
    plan = space_time_astar(
        grid, grid.cell(0, 0), grid.cell(2, 3), 0, 0, table, obstacles, heur
    )
    assert plan is not None
    assert all(grid.is_free(c) for c in plan.cells)


def test_unreachable_goal_returns_none(grid):
    obstacles = DynamicObstacles()
    heur = HeuristicCache(grid, obstacles)
    # Seal off the top-left corner.
    for cell in (grid.cell(0, 1), grid.cell(1, 0)):
        obstacles.block(cell, 0)

    plan = space_time_astar(
        grid, grid.cell(2, 2), grid.cell(0, 0), 0, 0,
        ReservationTable(), obstacles, heur,
    )
    assert plan is None


# ------------------------------------------------- constraints from others


def test_waits_for_a_reserved_cell(parts):
    grid, table, obstacles, heur = parts
    # Agent 1 sits in the only corridor cell for a while.
    blocker = grid.cell(2, 3)
    table.add(Plan(1, 0, tuple([blocker] * 6)))

    plan = space_time_astar(
        grid, grid.cell(2, 0), grid.cell(2, 6), 0, 0, table, obstacles, heur
    )
    assert plan is not None
    # It cannot pass through the blocker while that agent is there.
    for i, cell in enumerate(plan.cells):
        if cell == blocker:
            assert i > 5


def test_head_on_swap_is_forbidden(parts):
    grid, table, obstacles, heur = parts
    left, right = grid.cell(0, 2), grid.cell(0, 3)
    table.add(Plan(1, 0, (right, left)))  # agent 1 moves right -> left at t=0

    plan = space_time_astar(grid, left, right, 0, 0, table, obstacles, heur, hold=0)

    assert plan is not None
    # Agent 0 must not take left -> right over the same interval.
    assert (plan.cells[0], plan.cells[1]) != (left, right)


def test_permanent_goal_reservation_blocks_others(parts):
    grid, table, obstacles, heur = parts
    parked = grid.cell(0, 6)
    table.add(Plan(1, 0, (parked,)))  # agent 1 parks there forever

    assert not table.is_vertex_free(parked, 50, agent=0)
    plan = space_time_astar(grid, grid.cell(0, 0), parked, 0, 0, table, obstacles, heur)
    assert plan is None


def test_dedicated_dock_is_off_limits_to_everyone_else(parts):
    grid, table, obstacles, heur = parts
    dock = grid.cell(0, 3)
    table.dedicate(dock, agent=1)

    assert table.is_vertex_free(dock, 0, agent=1)
    assert not table.is_vertex_free(dock, 0, agent=0)
    assert not table.is_vertex_free(dock, 999, agent=0)
    # Survives plan removal, unlike ordinary reservations.
    table.remove(1)
    assert not table.is_vertex_free(dock, 0, agent=0)


def test_dynamic_obstacle_blocks_only_from_its_start_time(parts):
    grid, table, obstacles, heur = parts
    cell = grid.cell(2, 3)
    obstacles.block(cell, 10)

    assert not obstacles.is_blocked(cell, 9)
    assert obstacles.is_blocked(cell, 10)
    assert obstacles.is_blocked(cell, 500)


# --------------------------------------------------------------- multi-leg


def test_plan_route_visits_every_waypoint_in_order(parts):
    grid, table, obstacles, heur = parts
    waypoints = (grid.cell(0, 6), grid.cell(4, 6), grid.cell(4, 0))

    plan = plan_route(
        grid, grid.cell(0, 0), waypoints, 0, 0,
        table, obstacles, heur, service_time=1,
    )

    assert plan is not None
    positions = {}
    for i, cell in enumerate(plan.cells):
        positions.setdefault(cell, i)
    assert positions[waypoints[0]] < positions[waypoints[1]] < positions[waypoints[2]]
    assert plan.goal == waypoints[-1]


def test_service_time_is_spent_at_each_waypoint(parts):
    grid, table, obstacles, heur = parts
    stop = grid.cell(0, 3)
    plan = plan_route(
        grid, grid.cell(0, 0), (stop, grid.cell(0, 6)), 0, 0,
        table, obstacles, heur, service_time=2,
    )
    assert plan is not None
    assert plan.cells.count(stop) >= 3  # arrival plus two service steps


# ------------------------------------------------------ conflict detection


def test_plans_conflict_detects_shared_cell():
    a = Plan(0, 0, (1, 2, 3))
    b = Plan(1, 0, (4, 2, 5))  # both on cell 2 at t=1
    assert plans_conflict(a, b)


def test_plans_conflict_detects_swap():
    a = Plan(0, 0, (1, 2))
    b = Plan(1, 0, (2, 1))
    assert plans_conflict(a, b)


def test_plans_conflict_accounts_for_parking():
    a = Plan(0, 0, (1, 2))  # parks on cell 2 from t=1 onward
    b = Plan(1, 0, (5, 6, 7, 2))  # arrives on cell 2 at t=3
    assert plans_conflict(a, b)


def test_disjoint_plans_do_not_conflict():
    assert not plans_conflict(Plan(0, 0, (1, 2, 3)), Plan(1, 0, (10, 11, 12)))


def test_reservation_table_reports_no_conflicts_for_a_clean_plan_set():
    table = ReservationTable()
    table.add(Plan(0, 0, (1, 2, 3)))
    table.add(Plan(1, 0, (10, 11, 12)))
    assert table.find_conflicts() == []


# ------------------------------------------------------------ social laws


def test_social_laws_make_alternate_aisles_one_way():
    grid = WarehouseGrid.kiva(n_rows=30, n_cols=30)
    laws = SocialLaws(grid, enabled=True)

    assert laws.restricted_rows, "no through-aisles were classified"
    eastbound = laws.restricted_rows[0]
    interior = grid.n_cols // 2
    here = grid.cell(eastbound, interior)
    assert laws.allows(here, here + 1) != laws.allows(here, here - 1)


def test_social_laws_never_restrict_vertical_moves_or_waiting():
    grid = WarehouseGrid.kiva(n_rows=30, n_cols=30)
    laws = SocialLaws(grid, enabled=True)
    row = laws.restricted_rows[0]
    here = grid.cell(row, grid.n_cols // 2)

    assert laws.allows(here, here)  # waiting
    assert laws.allows(here, here + grid.n_cols)  # south
    assert laws.allows(here, here - grid.n_cols)  # north


def test_social_laws_keep_the_floor_strongly_connected():
    """A convention that trades conflicts for infeasibility would be no use."""
    grid = WarehouseGrid.kiva(n_rows=30, n_cols=30)
    assert check_connectivity(grid, SocialLaws(grid, enabled=True))


def test_disabled_social_laws_allow_everything():
    grid = WarehouseGrid.kiva(n_rows=20, n_cols=20)
    laws = SocialLaws(grid, enabled=False)
    assert laws.as_filter() is None
    assert laws.allows(0, 1)


def test_paths_respect_the_move_filter(grid):
    obstacles = DynamicObstacles()
    # Forbid every eastward move; the search must go around.
    def westbound_only(u: int, v: int) -> bool:
        return v != u + 1

    heur = HeuristicCache(grid, obstacles, move_filter=westbound_only)
    plan = space_time_astar(
        grid, grid.cell(0, 0), grid.cell(0, 6), 0, 0,
        ReservationTable(), obstacles, heur, move_filter=westbound_only,
    )
    assert plan is None  # nothing can move east at all, so the goal is unreachable
