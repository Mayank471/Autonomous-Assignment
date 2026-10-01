"""Experiment sweeps.

Answers the three questions the assignment asks, in order:

1. **Total time steps for all agents to finish their tasks** -- the headline
   metric, per repair strategy.
2. **Agents whose plans change per single disruption** -- the impact of plan
   modification.
3. **How both scale** with the number of agents and with the density of dynamic
   obstacles.

Plus secondary sweeps over the parameters that control the trade-off: ``beta``
(the penalty for changing a plan at all), ``tau`` (PGP's dampening threshold),
ADOPT's error bound ``epsilon``, the neighbourhood cap ``k_max``, and whether
social laws are in force.

Every configuration is run over many seeds.  Crucially, all four strategies see
the *same* scenarios and the *same* disruption schedules: the initial joint plan
is computed once per ``(n_agents, seed)`` and replayed, so differences in the
results come from the repair strategy and nothing else.

Usage::

    python experiments/run_experiments.py --quick     # seconds, for a smoke test
    python experiments/run_experiments.py             # the full sweep
    python experiments/run_experiments.py --jobs 4    # limit parallelism
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warehouse import scenario  # noqa: E402
from warehouse.metrics import RunMetrics, aggregate  # noqa: E402
from warehouse.repair import RepairConfig  # noqa: E402
from warehouse.simulator import simulate  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

STRATEGIES = ("selfish", "greedy_pgp", "adopt", "cbs", "full_replan")


@dataclass(frozen=True)
class Job:
    """One simulation to run."""

    strategy: str
    n_agents: int
    density: float
    seed: int
    beta: float = 4.0
    tau: int = 0
    k_max: int = 6
    epsilon: float = 0.0
    social_laws: bool = False
    tasks_per_agent: int = 3
    rows: int = 50
    cols: int = 60
    sweep: str = "main"


@dataclass
class Grids:
    """Lazily built, process-local grid cache.

    A grid is a few megabytes of precomputed adjacency and is immutable, so one
    instance per worker process is reused across every job it runs.
    """

    _cache: dict[tuple[int, int], Any] = field(default_factory=dict)

    def get(self, rows: int, cols: int):
        key = (rows, cols)
        if key not in self._cache:
            self._cache[key] = scenario.make_grid(n_rows=rows, n_cols=cols)
        return self._cache[key]


_GRIDS = Grids()


def run_job(job: Job) -> dict[str, Any]:
    """Execute one configuration and return its metrics as a CSV row."""
    grid = _GRIDS.get(job.rows, job.cols)
    sc = scenario.build(
        grid,
        job.n_agents,
        job.seed,
        tasks_per_agent=job.tasks_per_agent,
        social_laws=job.social_laws,
    )
    config = RepairConfig(
        k_max=job.k_max, beta=job.beta, tau=job.tau, epsilon=job.epsilon
    )

    started = time.perf_counter()
    result = simulate(
        grid,
        sc.fresh_tasks(),
        sc.solution,
        sc.disruptions(density=job.density),
        strategy=job.strategy,
        config=config,
        seed=job.seed,
        move_filter=sc.move_filter,
    )
    metrics: RunMetrics = result.metrics
    metrics.density = job.density
    metrics.social_laws = job.social_laws

    row = metrics.as_row()
    row["sweep"] = job.sweep
    row["wall_s"] = round(time.perf_counter() - started, 3)
    return row


# ------------------------------------------------------------------- sweeps


def main_sweep(seeds: Sequence[int], agents: Sequence[int], densities: Sequence[float]):
    """Strategy x agent count x obstacle density -- the assignment's core question."""
    for n, density, strategy, seed in itertools.product(
        agents, densities, STRATEGIES, seeds
    ):
        yield Job(
            strategy=strategy,
            n_agents=n,
            density=density,
            seed=seed,
            sweep="main",
        )


def parameter_sweeps(seeds: Sequence[int], n_agents: int, density: float):
    """The knobs that control the cost/churn trade-off, one at a time."""
    for seed in seeds:
        # beta: the penalty for altering a plan at all.
        for beta in (0.0, 1.0, 2.0, 4.0, 10.0, 25.0, 60.0):
            yield Job("adopt", n_agents, density, seed, beta=beta, sweep="beta")

        # tau: PGP's dampened responsiveness.
        for tau in (0, 2, 5, 10, 25):
            yield Job("adopt", n_agents, density, seed, tau=tau, sweep="tau")

        # epsilon: ADOPT's error bound -- optimality traded for cycles.
        for epsilon in (0.0, 2.0, 10.0, 50.0):
            yield Job("adopt", n_agents, density, seed, epsilon=epsilon, sweep="epsilon")

        # k_max: how far a single disruption may propagate.
        for k_max in (2, 4, 6, 8, 10):
            yield Job("adopt", n_agents, density, seed, k_max=k_max, sweep="k_max")

        # Social laws on/off, for every strategy.
        for strategy in STRATEGIES:
            for laws in (False, True):
                yield Job(
                    strategy, n_agents, density, seed, social_laws=laws, sweep="social_laws"
                )


def build_jobs(quick: bool) -> list[Job]:
    if quick:
        seeds = (1, 2, 3)
        jobs = list(main_sweep(seeds, agents=(10, 20), densities=(0.0, 0.02)))
        jobs += list(parameter_sweeps(seeds[:2], n_agents=15, density=0.02))
        return jobs

    # Density is the fraction of *free* cells that become permanently blocked
    # during the run.  The warehouse has ~1650 free cells, so 0.04 already walls
    # off about 65 of them -- a severe incident, not a mild one.  Going much
    # beyond that stops measuring plan repair and starts measuring how long a
    # destroyed warehouse takes to grind to a halt.
    #
    # Twelve seeds rather than thirty: the whole grid has to finish, and the
    # 80-agent full-replan cells cost seconds each.  Twelve still gives usable
    # confidence intervals on every point.
    seeds = tuple(range(1, 13))
    jobs = list(
        main_sweep(
            seeds,
            agents=(10, 20, 30, 40, 60, 80),
            densities=(0.0, 0.01, 0.02, 0.03, 0.04),
        )
    )
    jobs += list(parameter_sweeps(seeds[:8], n_agents=30, density=0.02))
    return jobs


# -------------------------------------------------------------------- driver


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def execute(
    jobs: list[Job],
    jobs_parallel: int,
    *,
    checkpoint: Path | None = None,
    every: int = 25,
) -> list[dict[str, Any]]:
    """Run every job, checkpointing results to disk as they complete.

    The sweep takes tens of minutes, so writing only at the end makes it
    all-or-nothing: an interrupted run loses everything it had already computed.
    Rows are flushed to ``checkpoint`` every ``every`` completions instead, which
    costs almost nothing (the file is a few megabytes) and means a partial run is
    still a usable partial result.
    """
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    total = len(jobs)

    def flush() -> None:
        if checkpoint is not None and rows:
            write_csv(checkpoint, rows)

    if jobs_parallel <= 1:
        for index, job in enumerate(jobs, 1):
            rows.append(run_job(job))
            if index % every == 0:
                flush()
            _progress(index, total, started)
        flush()
        return rows

    # Sorted longest-first so the big configurations start early and the tail
    # of the run is not one 80-agent job finishing alone.
    ordered = sorted(jobs, key=lambda j: -j.n_agents)
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=jobs_parallel) as pool:
        for index, row in enumerate(pool.map(run_job, ordered, chunksize=2), 1):
            rows.append(row)
            if index % every == 0:
                flush()
            _progress(index, total, started)
    flush()
    return rows


def _progress(index: int, total: int, started: float) -> None:
    if index % 25 and index != total:
        return
    # NB: printing unbuffered matters -- piping this through `tail` hides it.
    elapsed = time.perf_counter() - started
    rate = index / elapsed if elapsed else 0
    remaining = (total - index) / rate if rate else 0
    print(
        f"  {index:5d}/{total}  {elapsed/60:5.1f} min elapsed, "
        f"~{remaining/60:4.1f} min left",
        flush=True,
    )


def summarise(rows: list[dict[str, Any]]) -> None:
    """Print the headline comparison so the run is readable without the CSVs."""
    main = [r for r in rows if r["sweep"] == "main" and r["density"] > 0]
    if not main:
        return

    print("\n" + "=" * 78)
    print("Headline: repair strategy comparison (disrupted runs, all densities)")
    print("=" * 78)
    print(
        f"{'strategy':14s} {'total steps':>12s} {'chg/disrupt':>12s} "
        f"{'involved':>9s} {'ms/repair':>10s} {'done':>7s} {'stranded':>9s}"
    )
    for strategy in STRATEGIES:
        group = [r for r in main if r["strategy"] == strategy]
        if not group:
            continue

        def avg(key: str) -> float:
            return sum(float(r[key]) for r in group) / len(group)

        print(
            f"{strategy:14s} {avg('total_timesteps'):12.0f} "
            f"{avg('mean_agents_changed'):12.2f} {avg('mean_agents_involved'):9.2f} "
            f"{avg('mean_repair_ms'):10.1f} {avg('completion_rate'):7.2%} "
            f"{avg('agents_stranded'):9.2f}"
        )
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="tiny sweep for a smoke test"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="worker processes (1 disables parallelism)",
    )
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    jobs = build_jobs(args.quick)
    print(
        f"Running {len(jobs)} simulations "
        f"({'quick' if args.quick else 'full'} sweep) on {args.jobs} worker(s)",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    rows = execute(jobs, args.jobs, checkpoint=args.out / "runs.csv")
    write_csv(args.out / "runs.csv", rows)

    main_rows = [r for r in rows if r["sweep"] == "main"]
    metrics_by_key = aggregate(
        [_as_metrics(r) for r in main_rows], "strategy", "n_agents", "density"
    )
    write_csv(args.out / "summary_main.csv", metrics_by_key)

    for sweep in ("beta", "tau", "epsilon", "k_max", "social_laws"):
        subset = [r for r in rows if r["sweep"] == sweep]
        if subset:
            write_csv(args.out / f"summary_{sweep}.csv", subset)

    summarise(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


def _as_metrics(row: dict[str, Any]) -> RunMetrics:
    """Rebuild a metrics object from a CSV row, for the aggregation helper."""
    metrics = RunMetrics(
        strategy=row["strategy"],
        n_agents=int(row["n_agents"]),
        density=float(row["density"]),
        seed=int(row["seed"]),
    )
    metrics.total_timesteps = int(row["total_timesteps"])
    metrics.makespan = int(row["makespan"])
    metrics.repair_failures = int(row["repair_failures"])
    # The aggregator reads these through properties, so seed the backing lists
    # with a single representative value each.
    metrics.agents_changed = [float(row["mean_agents_changed"])]
    metrics.agents_involved = [float(row["mean_agents_involved"])]
    metrics.repair_ms = [float(row["mean_repair_ms"])]
    metrics.repair_messages = [float(row["mean_repair_messages"])]
    metrics.tasks_completed = int(row["tasks_completed"])
    metrics.tasks_total = int(row["tasks_total"])
    return metrics


if __name__ == "__main__":
    main()
