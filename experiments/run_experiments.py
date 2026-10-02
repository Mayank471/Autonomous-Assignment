"""Experiments E1-E4 (see report/report.md for what each one answers).

    python experiments/run_experiments.py all --quick     # smoke test, minutes
    python experiments/run_experiments.py all             # full study
    python experiments/run_experiments.py single agents density slack exactness
    python experiments/run_experiments.py all --jobs 8

E1 ``single``     one disruption injected into a running plan; every strategy
                  repairs the identical state.  -> agents changed per disruption
E2a ``agents``    full episodes, number of agents varied.  -> total time, scaling
E2b ``density``   full episodes, obstacle density varied.   -> total time, scaling
E3 ``slack``      emergency deadline slack varied.          -> changes vs lateness
E4 ``exactness``  KR-CBS against a brute-force oracle on small instances.

Every job is deterministic in its seed and is written as one JSON line as soon
as it finishes, so an interrupted run resumes where it stopped.  All strategies
see the same instance, initial plan and disruptions.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warehouse.disruptions import KINDS, EventRates, make_schedule, sample_single  # noqa: E402
from warehouse.plan import plan_distance, plans_differ  # noqa: E402
from warehouse.repair import STRATEGIES, repair  # noqa: E402
from warehouse.repair.cbs import brute_force_min_change  # noqa: E402
from warehouse.scenario import STREAM_SINGLE, make_instance, rng_for  # noqa: E402
from warehouse.simulator import (SimConfig, World, apply_event, initial_plans,  # noqa: E402
                                 make_context, run_episode)
from warehouse.stastar import advance  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

FULL = dict(
    agents=[10, 20, 30, 40, 50], densities=[0.0, 0.01, 0.02, 0.04, 0.08],
    base_n=30, base_rho=0.02, seeds=20, single_samples=100,
    slacks=[0, 2, 5, 10, None], slack_samples=100, slack_n=30,
    exact_agents=[4, 6, 8], exact_samples=70,
)
QUICK = dict(
    agents=[10, 30], densities=[0.0, 0.04], base_n=20, base_rho=0.02, seeds=2,
    single_samples=4, slacks=[0, None], slack_samples=4, slack_n=20,
    exact_agents=[4], exact_samples=6,
)


# ------------------------------------------------------------------ helpers
def _world_at(n_agents: int, seed: int, preset: str = "main"):
    """Instance + initial plan, advanced (undisrupted) to a random time."""
    inst = make_instance(n_agents, seed, preset=preset)
    plans = initial_plans(inst)
    world = World(inst, plans)
    rng = rng_for(seed, STREAM_SINGLE)
    makespan = max(p.end for p in plans.values())
    t_d = int(rng.integers(max(1, int(0.1 * makespan)), max(2, int(0.7 * makespan))))
    while world.t < t_d:
        world.step()
    return world, rng


def _delivery_time(world: World, a: int, plan, goal_idx: int) -> int | None:
    st = world.agents[a]
    k = st.progress
    for i, c in enumerate(plan.cells):
        k = advance(st.mission.goals, k, c)
        if k > goal_idx:
            return plan.t0 + i
    return None


def _repair_once(world: World, ev, strategy: str, cfg: SimConfig) -> dict | None:
    """Apply ``ev`` to a copy of ``world`` and repair it with ``strategy``."""
    w = world.copy()
    status, forced, cell = apply_event(w, ev, cfg)
    if status != "applied" or not forced:
        return None
    ctx = make_context(w, forced, cell, cfg)
    t0 = time.perf_counter()
    res = repair(strategy, ctx)
    runtime = time.perf_counter() - t0
    out = {"forced": len(forced), "level": res.level, "exact": res.exact,
           "runtime_s": runtime, "ok": res.ok, "hl_nodes": res.hl_nodes,
           "ll_calls": res.session.ll_calls, "ll_expansions": res.session.ll_expansions,
           "deadline_relaxed": res.deadline_relaxed}
    if not res.ok:
        return out
    changed = res.session.close(ctx.plans, res.plans, ctx.t)
    out.update(
        changed=len(changed), collateral=len(changed - set(forced)),
        contacted=len(res.session.contacted), messages=res.session.n_messages,
        plan_distance=sum(plan_distance(ctx.plans[a], res.plans[a], ctx.t) for a in changed),
        soc_delta=sum(res.plans[a].end - ctx.plans[a].end for a in changed),
    )
    if ev.kind == "emergency":
        e = forced[0]
        st = w.agents[e]
        plan = res.plans.get(e, ctx.plans[e])
        out["emergency_lateness"] = _delivery_time(w, e, plan, st.deadline[0]) - st.emergency_earliest
    return out


# --------------------------------------------------------------------- jobs
def job_single(n_agents: int, kind: str, sample: int) -> list[dict]:
    seed = 10_000 + sample
    world, rng = _world_at(n_agents, seed)
    ev = sample_single(world, kind, rng)
    if ev is None:
        return [{"exp": "single", "n_agents": n_agents, "kind": kind, "sample": sample,
                 "skipped": True}]
    rows = []
    for strategy in STRATEGIES:
        r = _repair_once(world, ev, strategy, SimConfig(strategy=strategy, emergency_slack=0))
        if r is None:
            return [{"exp": "single", "n_agents": n_agents, "kind": kind, "sample": sample,
                     "skipped": True}]
        rows.append({"exp": "single", "n_agents": n_agents, "kind": kind, "sample": sample,
                     "strategy": strategy, "t": world.t, **r})
    return rows


def job_slack(n_agents: int, slack, sample: int) -> list[dict]:
    seed = 20_000 + sample
    world, rng = _world_at(n_agents, seed)
    ev = sample_single(world, "emergency", rng)
    if ev is None:
        return [{"exp": "slack", "slack": slack, "sample": sample, "skipped": True}]
    rows = []
    for strategy in ("krcbs", "local", "global"):
        r = _repair_once(world, ev, strategy, SimConfig(strategy=strategy, emergency_slack=slack))
        if r is None:
            return [{"exp": "slack", "slack": slack, "sample": sample, "skipped": True}]
        rows.append({"exp": "slack", "n_agents": n_agents, "slack": slack, "sample": sample,
                     "strategy": strategy, **r})
    return rows


def job_episode(exp: str, n_agents: int, rho: float, seed: int, strategy: str) -> list[dict]:
    inst = make_instance(n_agents, seed)
    plans = initial_plans(inst)
    horizon = max(p.end for p in plans.values())
    events = make_schedule(inst, horizon, EventRates(rho=rho), seed)
    r = run_episode(inst, plans, events, SimConfig(strategy=strategy, emergency_slack=0))
    reps = r.repairs
    compact = [[x["kind"], x["forced"], x["changed"], x["collateral"], x.get("level"),
                x.get("exact"), x.get("messages", 0), x.get("contacted", 0),
                x.get("plan_distance", 0), round(x.get("runtime_s", 0.0), 4)]
               for x in reps]
    return [{
        "exp": exp, "n_agents": n_agents, "rho": rho, "seed": seed, "strategy": strategy,
        "completed": r.completed, "soc": r.soc, "soc0": r.soc0, "makespan": r.makespan,
        "makespan0": horizon, "agents_done": r.agents_done, "deliveries_done": r.deliveries_done,
        "n_events": len(events), "n_repairs": len(reps),
        "levels": dict(r.levels), "realized_density": r.realized_density,
        "emergency_lateness": r.emergency_lateness, "runtime_s": r.runtime_s,
        "repairs": compact,
    }]


def job_exactness(n_agents: int, sample: int) -> list[dict]:
    seed = 30_000 + sample
    world, rng = _world_at(n_agents, seed, preset="small")
    kind = KINDS[sample % 3]
    ev = sample_single(world, kind, rng)
    base = {"exp": "exactness", "n_agents": n_agents, "sample": sample, "kind": kind}
    if ev is None:
        return [{**base, "skipped": True}]
    cfg = SimConfig(strategy="krcbs", emergency_slack=0)
    w = world.copy()
    status, forced, cell = apply_event(w, ev, cfg)
    if status != "applied" or not forced:
        return [{**base, "skipped": True}]
    ctx = make_context(w, forced, cell, cfg)
    t0 = time.perf_counter()
    res = repair("krcbs", ctx)
    t_kr = time.perf_counter() - t0
    changed = {a for a, p in res.plans.items() if plans_differ(ctx.plans[a], p, ctx.t)}
    soc = ctx.base_soc + sum(res.plans[a].end - ctx.plans[a].end for a in changed)
    t0 = time.perf_counter()
    try:
        best = brute_force_min_change(ctx)
        bf_error = None
    except RuntimeError as exc:
        best, bf_error = None, str(exc)
    t_bf = time.perf_counter() - t0
    return [{**base, "forced": len(forced), "kr_changed": len(changed), "kr_soc": soc,
             "kr_level": res.level, "kr_exact": res.exact, "kr_runtime_s": t_kr,
             "bf_changed": best[0] if best else None, "bf_soc": best[1] if best else None,
             "bf_runtime_s": t_bf, "bf_error": bf_error}]


# ------------------------------------------------------------------ driver
def build_jobs(which: list[str], P: dict) -> dict[str, list[tuple]]:
    jobs: dict[str, list[tuple]] = {}
    if "single" in which:
        jobs["e1_single"] = [("single", n, k, s) for n in P["agents"] for k in KINDS
                             for s in range(P["single_samples"])]
    if "agents" in which:
        jobs["e2a_agents"] = [("episode", "agents", n, P["base_rho"], s, st)
                              for n in P["agents"] for s in range(P["seeds"]) for st in STRATEGIES]
    if "density" in which:
        jobs["e2b_density"] = [("episode", "density", P["base_n"], rho, s, st)
                               for rho in P["densities"] for s in range(P["seeds"])
                               for st in STRATEGIES]
    if "slack" in which:
        jobs["e3_slack"] = [("slack", P["slack_n"], d, s) for d in P["slacks"]
                            for s in range(P["slack_samples"])]
    if "exactness" in which:
        jobs["e4_exactness"] = [("exactness", n, s) for n in P["exact_agents"]
                                for s in range(P["exact_samples"])]
    return jobs


def run_job(job: tuple) -> tuple[tuple, list[dict]]:
    kind, *args = job
    fn = {"single": job_single, "episode": job_episode, "slack": job_slack,
          "exactness": job_exactness}[kind]
    return job, fn(*args)


def job_key(job: tuple) -> str:
    return json.dumps(list(job))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="+",
                    choices=["all", "single", "agents", "density", "slack", "exactness"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    which = ["single", "agents", "density", "slack", "exactness"] if "all" in args.which else args.which
    P = QUICK if args.quick else FULL
    out_dir = args.out / ("quick" if args.quick else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, jobs in build_jobs(which, P).items():
        path = out_dir / f"{name}.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["job"])
        todo = [j for j in jobs if job_key(j) not in done]
        # longest jobs first keeps the pool busy at the end
        todo.sort(key=lambda j: -(j[2] if isinstance(j[2], int) else 0))
        print(f"{name}: {len(jobs)} jobs, {len(done)} already done, {len(todo)} to run", flush=True)
        if not todo:
            continue
        t0 = time.perf_counter()
        with path.open("a", encoding="utf-8") as fh, mp.Pool(args.jobs) as pool:
            for i, (job, rows) in enumerate(pool.imap_unordered(run_job, todo), 1):
                for row in rows:
                    row["job"] = job_key(job)
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
                if i % max(1, len(todo) // 20) == 0 or i == len(todo):
                    print(f"  {name}: {i}/{len(todo)}  {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
