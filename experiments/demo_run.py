"""Watch one warehouse run, with every repair episode narrated.

Prints the floor, then steps through a simulation reporting each disruption as
it lands: how many agents it invalidated, how many were consulted, how many
actually changed their plan, and how long the negotiation took.  Running the
same scenario under different strategies side by side is the quickest way to see
what the numbers in the report mean.

Usage::

    python experiments/demo_run.py                      # ADOPT, 20 agents
    python experiments/demo_run.py --compare            # all four strategies
    python experiments/demo_run.py --strategy selfish --agents 30
    python experiments/demo_run.py --show-floor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warehouse import scenario  # noqa: E402
from warehouse.repair import SOLVERS, RepairConfig  # noqa: E402
from warehouse.simulator import simulate  # noqa: E402

RULE = "=" * 78


def describe_scenario(sc, disruptions) -> None:
    grid = sc.grid
    print(RULE)
    print(f"Warehouse {grid.n_rows}x{grid.n_cols}: {len(grid.free_cells)} free cells, "
          f"{len(grid.pickup_points)} shelf access points, "
          f"{len(grid.delivery_points)} stations, {len(grid.dock_points)} docking bays")
    print(f"Fleet: {sc.n_agents} agents, {sc.tasks_per_agent} tasks each")
    print(f"Initial joint plan: {sc.solution.sum_of_costs} total steps, "
          f"makespan {sc.solution.makespan}, "
          f"planned in {sc.solution.planning_time_ms:.0f} ms")
    print(f"  coordination cost over independent optima: "
          f"+{sc.solution.coordination_overhead} steps")

    kinds: dict[str, int] = {}
    for disruption in disruptions:
        kinds[disruption.kind] = kinds.get(disruption.kind, 0) + 1
    summary = ", ".join(f"{count} x {kind}" for kind, count in sorted(kinds.items()))
    print(f"Scheduled disruptions: {len(disruptions)} ({summary or 'none'})")
    print(RULE)


def run_one(sc, disruptions, strategy: str, args) -> None:
    print()
    print(f"--- {strategy}: {SOLVERS[strategy]} ---")
    result = simulate(
        sc.grid,
        sc.fresh_tasks(),
        sc.solution,
        disruptions,
        strategy=strategy,
        config=RepairConfig(k_max=args.k_max, beta=args.beta, tau=args.tau),
        seed=args.seed,
        move_filter=sc.move_filter,
        verbose=args.verbose,
    )
    m = result.metrics

    print(f"  total time steps (all agents) : {m.total_timesteps}")
    print(f"  makespan                      : {m.makespan}")
    print(f"  tasks completed               : {m.tasks_completed}/{m.tasks_total} "
          f"({m.completion_rate:.0%})")
    print(f"  agents stranded               : {m.agents_stranded}")
    print(f"  repairs performed             : {m.repairs_performed}")
    print(f"  agents replanned per disruption: {m.mean_agents_changed:.2f} "
          f"(max {m.max_agents_changed})")
    print(f"  agents consulted per disruption: {m.mean_agents_involved:.2f}")
    print(f"  largest negotiation group      : {m.max_episode_size}")
    print(f"  changes actually broadcast     : {m.mean_reported_changes:.2f}")
    print(f"  repair time per disruption     : {m.mean_repair_ms:.1f} ms "
          f"({m.total_repair_ms:.0f} ms total)")
    print(f"  messages per disruption        : {m.mean_repair_messages:.0f}")
    print(f"  repair failures / timeouts     : {m.repair_failures} / {m.solver_timeouts}")
    print(f"  collisions in final plan       : {m.collisions_detected}")
    if m.collisions_detected:
        print("  !! the joint plan is not collision-free -- this is a bug")
    for note in dict.fromkeys(m.notes):
        print(f"  note: {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="adopt", choices=sorted(SOLVERS))
    parser.add_argument("--compare", action="store_true", help="run all four strategies")
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--density", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--cols", type=int, default=60)
    parser.add_argument("--k-max", dest="k_max", type=int, default=6)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--tau", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="narrate each repair")
    parser.add_argument("--show-floor", action="store_true")
    args = parser.parse_args()

    grid = scenario.make_grid(n_rows=args.rows, n_cols=args.cols)
    sc = scenario.build(grid, args.agents, args.seed)
    disruptions = sc.disruptions(density=args.density)

    describe_scenario(sc, disruptions)
    if args.show_floor:
        print()
        print(grid.render())

    strategies = sorted(SOLVERS) if args.compare else [args.strategy]
    for strategy in strategies:
        run_one(sc, disruptions, strategy, args)
    print()


if __name__ == "__main__":
    main()
