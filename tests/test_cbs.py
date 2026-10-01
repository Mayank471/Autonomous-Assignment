"""Conflict-Based Search: the repair solver from outside the syllabus.

CBS is the one component here not taken from the taught material, so it carries
the burden of justifying itself. These tests pin down the three properties it
was added for: it resolves conflicts the candidate-set formulation could not, it
only replans agents a conflict actually reaches, and it is deterministic.

That last one is not a nicety. An earlier version bounded the search by
wall-clock time, which bounds cost correctly but makes the result depend on how
busy the machine is: two runs of one configuration disagreed by 34 percentage
points of task completion on the same seed. The bound is now on low-level
searches, which the search counts itself.
"""

from __future__ import annotations

import pytest

from warehouse import scenario
from warehouse.plan import Plan
from warehouse.repair import RepairConfig
from warehouse.repair.cbs import Conflict, Constraint, find_conflict
from warehouse.simulator import simulate


@pytest.fixture(scope="module")
def grid():
    return scenario.make_grid(n_rows=30, n_cols=36)


def run(grid, strategy, *, n_agents=16, seed=5, density=0.03, config=None, **kw):
    sc = scenario.build(grid, n_agents, seed=seed)
    return simulate(
        grid,
        sc.fresh_tasks(),
        sc.solution,
        sc.disruptions(density=density),
        strategy=strategy,
        config=config or RepairConfig(),
        seed=seed,
        move_filter=sc.move_filter,
        **kw,
    )


# ------------------------------------------------------- conflict detection


def test_detects_a_vertex_conflict():
    plans = {0: Plan(0, 0, (1, 2, 3)), 1: Plan(1, 0, (9, 2, 8))}
    conflict = find_conflict(plans, 0)

    assert conflict is not None
    assert not conflict.swap
    assert conflict.time == 1
    assert conflict.cell_a == 2
    assert {conflict.a, conflict.b} == {0, 1}


def test_detects_a_head_on_swap():
    plans = {0: Plan(0, 0, (1, 2)), 1: Plan(1, 0, (2, 1))}
    conflict = find_conflict(plans, 0)

    assert conflict is not None
    assert conflict.swap
    assert {conflict.a, conflict.b} == {0, 1}


def test_detects_a_conflict_against_a_parked_agent():
    """A finished agent occupies its final cell forever, not just once."""
    plans = {0: Plan(0, 0, (5,)), 1: Plan(1, 0, (1, 2, 3, 4, 5))}
    conflict = find_conflict(plans, 0)

    assert conflict is not None
    assert conflict.cell_a == 5


def test_reports_the_earliest_conflict_first():
    plans = {0: Plan(0, 0, (1, 2, 7, 7)), 1: Plan(1, 0, (9, 2, 9, 7))}
    conflict = find_conflict(plans, 0)

    assert conflict is not None
    assert conflict.time == 1, "a later conflict was reported before an earlier one"


def test_no_conflict_between_disjoint_plans():
    plans = {0: Plan(0, 0, (1, 2, 3)), 1: Plan(1, 0, (10, 11, 12))}
    assert find_conflict(plans, 0) is None


def test_conflicts_before_now_are_ignored():
    """Only the remaining plan can be repaired; the past is not in scope."""
    plans = {0: Plan(0, 0, (1, 2, 3, 4)), 1: Plan(1, 0, (9, 2, 8, 7))}
    assert find_conflict(plans, 0) is not None
    assert find_conflict(plans, 2) is None


# ------------------------------------------------------------- branching


def test_a_vertex_conflict_branches_into_one_constraint_per_agent():
    conflict = Conflict(a=3, b=7, time=4, cell_a=11, cell_b=11)
    left, right = conflict.constraints()

    assert {left.agent, right.agent} == {3, 7}
    assert left.cell == right.cell == 11
    assert left.time == right.time == 4
    assert not left.is_edge and not right.is_edge


def test_a_swap_branches_into_edge_constraints():
    """A swap must forbid the transition, not the cell.

    Banning the cell outright would stop the agent ever using it, which is a far
    stronger constraint than the conflict warrants and prunes valid solutions.
    """
    conflict = Conflict(a=1, b=2, time=5, cell_a=10, cell_b=20, swap=True)
    left, right = conflict.constraints()

    assert left.is_edge and right.is_edge
    assert (left.agent, left.from_cell, left.cell) == (1, 10, 20)
    assert (right.agent, right.from_cell, right.cell) == (2, 20, 10)


# --------------------------------------------------------------- behaviour


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_cbs_leaves_no_collisions(grid, seed):
    assert run(grid, "cbs", seed=seed).metrics.collisions_detected == 0


@pytest.mark.parametrize("density", [0.0, 0.03, 0.08])
def test_cbs_leaves_no_collisions_across_densities(grid, density):
    assert run(grid, "cbs", density=density).metrics.collisions_detected == 0


def test_cbs_is_deterministic(grid):
    """Identical inputs must give identical results, run to run.

    The reason this is asserted rather than assumed: bounding the search by
    wall-clock time silently broke it, and the failure was invisible in any
    single run.
    """
    runs = [run(grid, "cbs", seed=7).metrics for _ in range(3)]
    signatures = {
        (m.total_timesteps, round(m.mean_agents_changed, 6), m.tasks_completed)
        for m in runs
    }
    assert len(signatures) == 1, f"CBS gave {len(signatures)} different results: {signatures}"


def test_cbs_changes_fewer_plans_than_the_candidate_set_formulation(grid):
    """The reason CBS was added: it only replans agents a conflict reaches."""
    cbs = run(grid, "cbs", seed=6).metrics
    dcop = run(grid, "adopt", seed=6).metrics

    assert cbs.agents_changed and dcop.agents_changed
    assert cbs.mean_agents_changed <= dcop.mean_agents_changed


def test_cbs_abandons_fewer_agents_than_the_candidate_set_formulation(grid):
    """Searching the plan space rather than five samples of it means fewer
    agents have to give up on their tasks."""
    cbs = run(grid, "cbs", seed=6, density=0.06).metrics
    dcop = run(grid, "adopt", seed=6, density=0.06).metrics

    assert cbs.abandoned_agents <= dcop.abandoned_agents


def test_cbs_respects_the_neighbourhood_cap(grid):
    result = run(grid, "cbs", config=RepairConfig(k_max=4, escalate_k=()))
    assert result.metrics.episode_sizes
    assert max(result.metrics.episode_sizes) <= 4


def test_cbs_never_calls_global_plan_generation(grid, monkeypatch):
    """It is still a *local* repair, whatever search it uses inside."""
    import warehouse.repair.solvers as solvers

    def forbidden(*args, **kwargs):
        raise AssertionError("CBS must not re-invoke global planning")

    monkeypatch.setattr(solvers, "replan_all", forbidden)
    run(grid, "cbs", seed=8)


def test_a_tighter_budget_falls_back_rather_than_failing(grid):
    """Running out of search budget must degrade, not break."""
    result = run(grid, "cbs", config=RepairConfig(cbs_max_nodes=1, cbs_max_low_level=1))

    assert result.metrics.collisions_detected == 0
    assert result.metrics.total_timesteps > 0


def test_escalation_reduces_churn(grid):
    """Widening a neighbourhood that could not be solved finds better repairs
    than giving up and letting the sequential fallback change everyone."""
    off = run(grid, "cbs", n_agents=20, seed=9, config=RepairConfig(escalate_k=())).metrics
    on = run(grid, "cbs", n_agents=20, seed=9, config=RepairConfig()).metrics

    assert on.mean_agents_changed <= off.mean_agents_changed
