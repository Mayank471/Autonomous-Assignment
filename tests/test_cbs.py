"""Keep/Release CBS: hand-built scenarios with known answers, and exactness
against the brute-force oracle on random small instances."""

import pytest

from warehouse.disruptions import KINDS, sample_single
from warehouse.grid import grid_from_ascii
from warehouse.obstacles import ObstacleWindows
from warehouse.plan import Plan, plans_differ
from warehouse.repair import RepairContext, repair
from warehouse.repair.cbs import brute_force_min_change
from warehouse.scenario import STREAM_SINGLE, make_instance, rng_for
from warehouse.simulator import SimConfig, World, apply_event, initial_plans, make_context
from warehouse.stastar import Query
from warehouse.validate import check_plans


def _ctx(g, t, plans, queries, forced, obstacles=None):
    obstacles = obstacles or ObstacleWindows()
    return RepairContext(g, t, plans, obstacles, tuple(forced), frozenset(plans),
                         lambda a: queries[a], None)


def _changed(ctx, res):
    return {a for a, p in res.plans.items() if plans_differ(ctx.plans[a], p, ctx.t)}


def _sound(ctx, res, docks):
    merged = dict(ctx.plans)
    merged.update(res.plans)
    return check_plans(ctx.grid, merged, ctx.t, docks=docks, obstacles=ctx.obstacles) == []


def test_blockage_that_can_be_waited_out_changes_one_agent():
    g = grid_from_ascii("""
########
#D....S#
###.####
###D####
########
""")
    a_dock, c_dock, station = g.cell(1, 1), g.cell(3, 3), g.cell(1, 6)
    a_plan = Plan(0, tuple(g.cell(1, c) for c in range(1, 7)) +
                  tuple(g.cell(1, c) for c in range(5, 0, -1)))
    plans = {0: a_plan, 1: Plan(0, (c_dock,))}
    obs = ObstacleWindows()
    obs.add(g.cell(1, 4), 0, 8)                    # A's path crosses (1,4) at t=3
    queries = {0: Query(0, a_dock, 0, (station, a_dock), a_dock),
               1: Query(1, c_dock, 0, (c_dock,), c_dock)}
    ctx = _ctx(g, 0, plans, queries, [0], obs)
    res = repair("krcbs", ctx)
    assert res.ok and res.level == "fast" and res.exact
    assert _changed(ctx, res) == {0}
    assert _sound(ctx, res, {0: a_dock, 1: c_dock})


STEP_ASIDE = """
#########
#D.....S#
####.####
####D####
#########
"""


def test_emergency_makes_one_blocker_step_aside():
    g = grid_from_ascii(STEP_ASIDE)
    a_dock, b_dock, station = g.cell(1, 1), g.cell(3, 4), g.cell(1, 7)
    # B idles in the lane until t=30, then goes home through the side branch
    b_plan = Plan(0, (g.cell(1, 5),) * 31 + (g.cell(1, 4), g.cell(2, 4), b_dock))
    plans = {0: Plan(0, (g.cell(1, 2),)), 1: b_plan}
    queries = {0: Query(0, g.cell(1, 2), 0, (station, a_dock), a_dock, deadline=(0, 5)),
               1: Query(1, g.cell(1, 5), 0, (b_dock,), b_dock)}
    ctx = _ctx(g, 0, plans, queries, [0])
    res = repair("krcbs", ctx)
    assert res.ok and res.level == "krcbs" and res.exact
    assert _changed(ctx, res) == {0, 1}
    assert res.plans[0].at(5) == station           # deadline met
    assert _sound(ctx, res, {0: a_dock, 1: b_dock})
    assert brute_force_min_change(ctx)[0] == 2


def test_release_is_transitive_when_the_refuge_is_occupied():
    g = grid_from_ascii("""
##########
#D......S#
##D#.#####
####.#####
####D#####
##########
""")
    a_dock, b_dock, c_dock, station = g.cell(1, 1), g.cell(2, 2), g.cell(4, 4), g.cell(1, 8)
    # B idles in the lane and C idles in the only side branch (B's only refuge)
    b_plan = Plan(0, (g.cell(1, 5),) * 41 + (g.cell(1, 4), g.cell(1, 3), g.cell(1, 2), b_dock))
    c_plan = Plan(0, (g.cell(2, 4),) * 41 + (g.cell(3, 4), c_dock))
    plans = {0: Plan(0, (g.cell(1, 2),)), 1: b_plan, 2: c_plan}
    queries = {0: Query(0, g.cell(1, 2), 0, (station, a_dock), a_dock, deadline=(0, 6)),
               1: Query(1, g.cell(1, 5), 0, (b_dock,), b_dock),
               2: Query(2, g.cell(2, 4), 0, (c_dock,), c_dock)}
    ctx = _ctx(g, 0, plans, queries, [0])
    res = repair("krcbs", ctx)
    assert res.ok and res.exact
    assert _changed(ctx, res) == {0, 1, 2}
    assert _sound(ctx, res, {0: a_dock, 1: b_dock, 2: c_dock})
    assert brute_force_min_change(ctx)[0] == 3


def _single_disruption_cases(n_cases: int, n_agents: int = 6):
    cfg = SimConfig()
    for seed in range(n_cases * 3):
        inst = make_instance(n_agents, seed, preset="small")
        plans = initial_plans(inst)
        world = World(inst, plans)
        rng = rng_for(seed, STREAM_SINGLE)
        makespan = max(p.end for p in plans.values())
        t_d = int(rng.integers(1, max(2, int(0.6 * makespan))))
        while world.t < t_d:
            world.step()
        kind = KINDS[seed % 3]
        ev = sample_single(world, kind, rng)
        if ev is None:
            continue
        status, forced, cell = apply_event(world, ev, cfg)
        if status != "applied" or not forced:
            continue
        yield seed, kind, make_context(world, forced, cell, cfg)
        n_cases -= 1
        if n_cases == 0:
            return


@pytest.mark.parametrize("n_agents", [4, 6])
def test_krcbs_matches_brute_force_and_beats_every_other_strategy(n_agents):
    checked = 0
    for seed, kind, ctx in _single_disruption_cases(12, n_agents):
        res = repair("krcbs", ctx)
        assert res.ok
        docks = {a: ctx.query_of(a).dock for a in ctx.active}
        docks.update({a: ctx.plans[a].cells[-1] for a in ctx.plans if a not in docks})
        assert _sound(ctx, res, docks), (seed, kind)
        n_changed = len(_changed(ctx, res))
        soc = ctx.base_soc + sum(p.end - ctx.plans[a].end
                                 for a, p in res.plans.items()
                                 if plans_differ(ctx.plans[a], p, ctx.t))
        if not res.exact:
            continue
        best = brute_force_min_change(ctx)
        assert best is not None
        assert n_changed == best[0], (seed, kind)
        assert soc == best[1], (seed, kind)
        for other in ("local", "cascade", "global"):
            o = repair(other, ctx)
            assert o.ok
            assert len(_changed(ctx, o)) >= n_changed, (seed, kind, other)
        checked += 1
    assert checked >= 8
