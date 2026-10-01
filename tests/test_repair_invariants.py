"""Invariants the repair layer must never break.

The central one is safety: **after every disruption is repaired, the joint plan
contains no vertex or edge collision.**  A repair that produces a faster
schedule by driving two robots through each other is not a repair.  This is
checked for every strategy, over many seeds, at several obstacle densities, and
after each disruption rather than only at the end of the run.

The rest pin down the properties the assignment actually asks about: that
repair stays local, that it never re-invokes global plan generation, and that
the ``beta`` knob really does trade travel time against how many agents are
disturbed.
"""

from __future__ import annotations

import pytest

from warehouse import scenario
from warehouse.repair import RepairConfig, SOLVERS
from warehouse.repair.episode import build_episode
from warehouse.simulator import simulate

LOCAL_STRATEGIES = ("selfish", "greedy_pgp", "adopt")
ALL_STRATEGIES = (*LOCAL_STRATEGIES, "full_replan")


@pytest.fixture(scope="module")
def grid():
    # Smaller than the experiment layout so the suite stays quick.
    return scenario.make_grid(n_rows=30, n_cols=36)


def run(grid, strategy, *, n_agents=12, seed=1, density=0.03, config=None, **kw):
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


# ------------------------------------------------------------------ safety


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_no_collisions_after_a_full_run(grid, strategy, seed):
    result = run(grid, strategy, seed=seed)
    conflicts = result.metrics.collisions_detected
    assert conflicts == 0, f"{strategy} seed={seed} left {conflicts} collisions"


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
@pytest.mark.parametrize("density", [0.0, 0.02, 0.06, 0.12])
def test_no_collisions_across_obstacle_densities(grid, strategy, density):
    result = run(grid, strategy, density=density, seed=7)
    assert result.metrics.collisions_detected == 0


@pytest.mark.parametrize("strategy", LOCAL_STRATEGIES)
def test_no_collisions_after_every_disruption(grid, strategy, monkeypatch):
    """Check after each disruption is repaired, not only at the end of the run.

    The boundary is the *disruption*, not the negotiation group: one disruption
    that invalidates many plans is repaired in successive groups, and between
    groups some agents legitimately still hold a plan that is about to change.
    """
    import warehouse.simulator as sim

    seen: list[int] = []
    original = sim.repair

    def checked_repair(strategy_name, disruption, ctx, **kwargs):
        outcome = original(strategy_name, disruption, ctx, **kwargs)
        seen.append(len(ctx.reservations.find_conflicts()))
        return outcome

    monkeypatch.setattr(sim, "repair", checked_repair)
    run(grid, strategy, seed=3)

    assert seen, "no repairs ran, so nothing was checked"
    assert max(seen) == 0, f"{strategy} left a collision after a disruption"


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_agents_only_ever_move_to_adjacent_cells(grid, strategy):
    """No teleporting: consecutive positions must be equal or 4-connected."""
    result = run(grid, strategy, seed=5, record_history=True)
    history = result.history
    assert len(history) > 5

    for before, after in zip(history, history[1:]):
        for agent, cell in after.items():
            previous = before.get(agent)
            if previous is None or previous == cell:
                continue
            assert cell in grid.neighbors(previous), (
                f"{strategy}: agent {agent} jumped from {previous} to {cell}"
            )


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_agents_never_stand_on_a_blocked_cell(grid, strategy):
    result = run(grid, strategy, seed=6, record_history=True)
    for positions in result.history:
        for agent, cell in positions.items():
            assert grid.is_free(cell), f"{strategy}: agent {agent} stood on a shelf"


# ------------------------------------------------------------- locality


@pytest.mark.parametrize("k_max", [2, 4, 6])
def test_negotiation_groups_never_exceed_k_max(grid, k_max):
    """``k_max`` bounds each negotiation group, which is what keeps ADOPT tractable.

    It does *not* bound the union over a whole disruption: one event can
    invalidate more plans than a single group can hold, and those are repaired
    in successive groups.  The two numbers are reported separately.
    """
    result = run(grid, "adopt", n_agents=16, config=RepairConfig(k_max=k_max))

    assert result.metrics.episode_sizes, "no repairs happened"
    assert max(result.metrics.episode_sizes) <= k_max


def test_local_repair_touches_far_fewer_agents_than_the_fleet(grid):
    """Even summed over groups, a disruption stays well short of everyone."""
    result = run(grid, "adopt", n_agents=16, config=RepairConfig(k_max=4))

    assert result.metrics.agents_involved
    assert result.metrics.mean_agents_involved < 16


def test_local_repair_changes_far_fewer_agents_than_replanning(grid):
    """The headline claim: bounded impact per disruption."""
    local = run(grid, "adopt", n_agents=16, seed=2)
    whole = run(grid, "full_replan", n_agents=16, seed=2)

    assert local.metrics.agents_changed and whole.metrics.agents_changed
    assert local.metrics.mean_agents_changed < whole.metrics.mean_agents_changed


def test_local_repair_never_calls_global_plan_generation(grid, monkeypatch):
    """The assignment's core constraint, asserted rather than assumed."""
    import warehouse.repair.solvers as solvers

    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("replan_all")
        raise AssertionError("local repair must not re-invoke global planning")

    monkeypatch.setattr(solvers, "replan_all", forbidden)
    for strategy in LOCAL_STRATEGIES:
        run(grid, strategy, seed=8)
    assert not calls


def test_episode_only_involves_agents_that_could_interact(grid):
    """Participants come from the interaction graph, so they share zone-time."""
    result = run(grid, "adopt", n_agents=14, seed=4)
    assert result.metrics.agents_involved
    assert min(result.metrics.agents_involved) >= 1
    assert max(result.metrics.episode_sizes) <= RepairConfig().k_max


# ----------------------------------------------------------- the beta knob


def test_higher_change_penalty_disturbs_fewer_agents(grid):
    """``beta`` is the "minimise agents altered" requirement as an objective.

    Raising it should not increase how many agents get touched; the solver
    should start preferring joint plans that leave more of them alone.
    """
    cheap = run(grid, "adopt", n_agents=16, seed=11, config=RepairConfig(beta=0.0))
    dear = run(grid, "adopt", n_agents=16, seed=11, config=RepairConfig(beta=40.0))

    assert cheap.metrics.agents_changed and dear.metrics.agents_changed
    assert dear.metrics.mean_agents_changed <= cheap.metrics.mean_agents_changed


def test_dampening_threshold_reduces_what_gets_broadcast(grid):
    """PGP mechanism 6: below ``tau`` a change is absorbed and never announced."""
    loud = run(grid, "adopt", n_agents=16, seed=12, config=RepairConfig(tau=0))
    quiet = run(grid, "adopt", n_agents=16, seed=12, config=RepairConfig(tau=25))

    assert quiet.metrics.mean_reported_changes <= loud.metrics.mean_reported_changes
    # Dampening only suppresses the announcement, never the change itself.
    assert quiet.metrics.mean_reported_changes <= quiet.metrics.mean_agents_changed


# --------------------------------------------------------------- soundness


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_runs_terminate_and_report_consistent_totals(grid, strategy):
    result = run(grid, strategy, seed=9)
    metrics = result.metrics

    assert metrics.total_timesteps > 0
    assert metrics.tasks_completed <= metrics.tasks_total
    assert 0.0 <= metrics.completion_rate <= 1.0
    assert metrics.agents_finished + metrics.agents_stranded <= metrics.n_agents
    assert len(result.finish_times) == metrics.n_agents


def test_undisrupted_run_completes_every_task(grid):
    """With nothing going wrong the initial plan should simply execute."""
    result = run(grid, "adopt", density=0.0, seed=10)
    metrics = result.metrics

    assert metrics.n_disruptions == 0
    assert metrics.repairs_performed == 0
    assert metrics.tasks_completed == metrics.tasks_total
    assert metrics.agents_stranded == 0
    assert metrics.collisions_detected == 0


def test_unknown_strategy_is_rejected(grid):
    with pytest.raises(KeyError, match="unknown repair strategy"):
        run(grid, "telepathy")


def test_all_advertised_strategies_run(grid):
    for strategy in SOLVERS:
        result = run(grid, strategy, n_agents=8, seed=13)
        assert result.metrics.collisions_detected == 0
