# Multiagent Plan Repair in an Automated Warehouse

A fleet of robots on a 2D warehouse grid plans collision-free pickup-and-delivery
routes with prioritized multi-goal Space-Time A\*. When something goes wrong
during execution -- a cell is suddenly blocked, a robot breaks down, a robot is
given an emergency task -- the affected robots **repair** their plans by
negotiating with their neighbours. They never re-run the planner from scratch,
and they change **as few plans as possible**.

The repair algorithm is **Keep/Release CBS (KR-CBS)**: an exact search that
minimises first the number of agents whose plans change, then total time. Each
conflict between a re-planning agent and an untouched agent *j* is split into
*KEEP j* (j's plan becomes fixed and everyone routes around it) and *RELEASE j*
(j is asked, by message, to propose a new route).

The full write-up is in [report/report.md](report/report.md).

## Layout

```
warehouse/
  grid.py          warehouse generator (docks, stations, shelves), cached BFS distances
  tasks.py         tasks and each agent's goal sequence (task order chosen by the agent)
  scenario.py      reproducible instances per seed
  plan.py          the plan object; change detection and plan distance
  obstacles.py     blocked-cell time windows (blockages, broken-down agents)
  reservation.py   the shared space-time reservation board
  stastar.py       multi-goal Space-Time A* (hard/soft constraints, deadlines, budgets)
  planner.py       prioritized planning (Cooperative A*) -- the plan generator
  negotiation.py   the repair session: REPLAN_REQUEST / PROPOSAL / COMMIT / RELEASE
  repair/
    cbs.py         KR-CBS, plain coalition CBS, and the brute-force exactness oracle
    solvers.py     the four strategies: krcbs, local, cascade, global
    episode.py     repair context/result and the escalation chain
  disruptions.py   disruption schedules and the single-disruption sampler
  emergency.py     emergency tasks and their deadlines
  simulator.py     execution loop: apply disruptions, repair, check, move
  validate.py      independent collision checker run after every repair
  metrics.py       bootstrap confidence intervals
  dcop/            ADOPT (course material; not used by the repair system)
experiments/
  run_experiments.py   E1-E4, parallel and resumable -> results/*.jsonl
  summarise_results.py tables -> results/summary.md
  make_figures.py      figures -> figures/*.png
  make_animation.py    figures/warehouse_run.gif
  demo_run.py          narrated single run
tests/                 pytest suite
```

## Running

```
pip install -r requirements.txt
python -m pytest                                      # test suite
python experiments/demo_run.py --compare              # one run, all four strategies
python experiments/run_experiments.py all --quick     # smoke test (minutes)
python experiments/run_experiments.py all             # full study (about an hour on 10 cores)
python experiments/summarise_results.py
python experiments/make_figures.py
```

Everything is deterministic in its seed. No search is limited by wall-clock
time; every budget is a count of nodes or expansions.
