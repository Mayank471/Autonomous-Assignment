"""End-to-end invariants of disrupted episodes, for every strategy.

Every commit inside ``run_episode`` is already checked by the independent
checker (``SimConfig.validate``), so a collision anywhere raises.  On top of
that these tests require: every episode completes, the result is fully
deterministic, and a failure is reported rather than papered over.
"""

import pytest

from warehouse.disruptions import EventRates, make_schedule
from warehouse.repair import STRATEGIES
from warehouse.scenario import make_instance
from warehouse.simulator import SimConfig, initial_plans, run_episode


def _episode(strategy, n_agents=12, seed=0, rho=0.03, preset="main"):
    inst = make_instance(n_agents, seed, preset=preset)
    plans = initial_plans(inst)
    horizon = max(p.end for p in plans.values())
    events = make_schedule(inst, horizon, EventRates(rho=rho, emergencies=1), seed)
    return run_episode(inst, plans, events, SimConfig(strategy=strategy))


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("seed", [0, 1])
def test_every_strategy_completes_collision_free(strategy, seed):
    r = _episode(strategy, seed=seed)
    assert r.completed, f"{strategy} failed: {r.levels}"
    assert r.agents_done == 1.0 and r.deliveries_done == 1.0
    assert all(rec.get("failed") is False for rec in r.repairs)
    assert all(rec["changed"] >= 1 for rec in r.repairs)


def _fingerprint(r):
    keep = ("eid", "kind", "t", "status", "forced", "changed", "collateral", "level",
            "hl_nodes", "ll_calls", "ll_expansions", "messages", "plan_distance")
    return (r.soc, r.makespan, [tuple(rec.get(k) for k in keep) for rec in r.records])


def test_runs_are_deterministic():
    assert _fingerprint(_episode("krcbs", seed=3)) == _fingerprint(_episode("krcbs", seed=3))


def test_no_disruptions_means_no_repairs_and_original_cost():
    inst = make_instance(10, 5)
    plans = initial_plans(inst)
    r = run_episode(inst, plans, [], SimConfig(strategy="krcbs"))
    assert r.completed and r.records == [] and r.soc == r.soc0


def test_failed_repair_is_reported_not_hidden(monkeypatch):
    import warehouse.simulator as sim
    from warehouse.repair.episode import LEVEL_FAILED, RepairResult

    def always_fail(strategy, ctx):
        return RepairResult(None, LEVEL_FAILED, ctx.session())

    monkeypatch.setattr(sim, "repair", always_fail)
    r = _episode("krcbs", seed=0)
    assert not r.completed and r.soc is None
    assert r.records[-1]["failed"] is True
    assert r.agents_done < 1.0
