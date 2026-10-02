"""Watch one warehouse run, with every repair narrated.

    python experiments/demo_run.py                        # 20 agents, KR-CBS
    python experiments/demo_run.py --agents 30 --rho 0.04 --strategy local
    python experiments/demo_run.py --compare              # all four strategies, same events
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warehouse.disruptions import EventRates, make_schedule  # noqa: E402
from warehouse.repair import STRATEGIES  # noqa: E402
from warehouse.scenario import make_instance  # noqa: E402
from warehouse.simulator import SimConfig, initial_plans, run_episode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=20)
    ap.add_argument("--rho", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategy", default="krcbs", choices=STRATEGIES)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    inst = make_instance(args.agents, args.seed)
    print(inst.grid.render())
    plans = initial_plans(inst)
    horizon = max(p.end for p in plans.values())
    print(f"\n{args.agents} agents, initial plan: total time {sum(p.end for p in plans.values())}, "
          f"makespan {horizon}")
    events = make_schedule(inst, horizon, EventRates(rho=args.rho), args.seed)
    print(f"{len(events)} disruptions scheduled\n")

    for strategy in (STRATEGIES if args.compare else [args.strategy]):
        r = run_episode(inst, plans, events, SimConfig(strategy=strategy))
        if not args.compare:
            for rec in r.repairs:
                print(f"t={rec['t']:4d} {rec['kind']:9s} forced={rec['forced']} "
                      f"changed={rec['changed']} (collateral {rec['collateral']}) "
                      f"via {rec['level']:13s} messages={rec.get('messages', 0)}")
        reps = r.repairs
        mean = sum(x["changed"] for x in reps) / max(1, len(reps))
        print(f"\n[{strategy}] completed={r.completed} total time={r.soc} (undisrupted {r.soc0}) "
              f"makespan={r.makespan} repairs={len(reps)} agents changed/repair={mean:.2f} "
              f"levels={dict(r.levels)}")


if __name__ == "__main__":
    main()
