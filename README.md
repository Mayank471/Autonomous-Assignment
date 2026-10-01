# Multiagent Plan Repair in an Automated Warehouse

A fleet of robots on a 2D warehouse grid plans collision-free pickup-and-delivery
routes with prioritized Space-Time A\*, then **repairs** those plans locally when
the environment disrupts them — negotiating with neighbouring agents rather than
regenerating the joint plan, and minimising how many agents are disturbed.

---

## The idea in one paragraph

A repair episode is modelled as a DCOP. When a disruption invalidates some plans, the
affected agents and a bounded ring of their neighbours each generate a handful of
candidate plans; binary constraints forbid pairs of plans that collide in
space-time; and minimising the number of altered original plans becomes a term in
the objective function. ADOPT then solves the repair problem asynchronously.

---

## Quick start

```bash
pip install -r requirements.txt

python -m pytest tests/ -q             # 180 tests, ~2 min
python experiments/demo_dcop.py        # DCOP examples
python experiments/run_experiments.py --quick   # pipeline smoke test, ~30s
python experiments/run_experiments.py           # full sweep (~30 min, checkpointed)
python experiments/summarise_results.py         # the report's tables
python experiments/make_figures.py              # figures from results/runs.csv
python experiments/make_animation.py            # animated GIF of a run
python experiments/demo_run.py --compare        # one scenario, all four strategies
```

---

## The five strategies compared

| Strategy | Searches | What it does |
|---|---|---|
| `selfish` | — | Only disrupted agents replan; nobody yields. Lower bound on agents changed — and it shows what that bound costs. |
| `greedy_pgp` | candidate set | Distributed hill-climbing. Fast, no optimality guarantee. |
| `adopt` | candidate set | The same repair posed as a DCOP and solved optimally by ADOPT. |
| `cbs` | the plan space | Conflict-Based Search over the neighbourhood. |
| `full_replan` | the plan space | Prioritized replanning of the whole fleet — the baseline the assignment says to avoid, kept so the cost of avoiding it is measurable. |

All five are handed the *same* scenarios and the *same* disruption schedules, so
differences are attributable to the repair alone.

### Why CBS was added

The negotiated strategies share a limit that only showed up in the data: the
DCOP picks among about five precomputed plans per agent, and when no combination
of those fits, agents abandon their tasks. That left the negotiated strategies
stranding **more** agents than the naive baseline (14.25 vs 11.90). Losing to
the do-nothing baseline is not a tuning problem — the search space is too small.

CBS ([Sharon et al., *AIJ* 219, 2015](https://doi.org/10.1016/j.artint.2014.11.006))
branches on *constraints* instead of enumerating plans, so it searches the whole
plan space and is complete. It also happens to minimise churn structurally
rather than through a cost term: an agent is replanned only when a constraint
lands on it, so agents not party to any conflict are never touched.

It cuts agents-disturbed by 31–41% at 80 agents, at roughly 50× ADOPT's compute.
See [`warehouse/repair/cbs.py`](warehouse/repair/cbs.py).

---

## Correctness

The joint plan is **collision-free after every disruption**, for every strategy,
across seeds and obstacle densities — asserted in
[`tests/test_repair_invariants.py`](tests/test_repair_invariants.py), and checked
mid-run rather than only at the end. Also asserted: agents never teleport or
stand on a shelf, local repair *never* calls global plan generation, negotiation
groups never exceed `k_max`, and raising `beta` never increases churn.

ADOPT is cross-checked against exhaustive search on random networks.

---

## Layout

```
warehouse/
  grid.py  plan.py  reservation.py  obstacles.py  stastar.py   the world and the search
  tasks.py  planner.py  sociallaws.py  organization.py         initial plan generation
  contractnet.py  disruptions.py  emergency.py                 what goes wrong, and who responds
  abstractplan.py                                              plan abstraction + dampening
  dcop/     model  dfstree  adopt  bruteforce  examples        distributed optimization
  repair/   episode  candidates  solvers                       the plan-repair layer
            cbs                                                Conflict-Based Search
  simulator.py  metrics.py  scenario.py                        execution and measurement
experiments/  run_experiments  refresh_runs  summarise_results  make_figures
              make_animation  demo_dcop  demo_run
tests/        180 tests
results/      generated CSVs          figures/  generated PNGs + GIF
report/       report.md
```

## Design notes worth knowing

* **Agents dock on the perimeter wall.** A finished agent occupies its final cell
  forever, so that cell is a permanent reservation. Docks are therefore dedicated
  per agent and placed on the outer ring, where removing them cannot disconnect
  the floor.
* **Repair splices, it does not replan.** Only the leg an agent is currently
  flying is re-planned; the rest of its route is reattached where that leg ends.
  This is both the assignment's actual requirement and a ~14× speedup over
  rebuilding each route from scratch.
* **A disruption that invalidates many plans is repaired in successive small
  groups**, not one large meeting. `k_max` bounds the group, which is what keeps
  a complete DCOP solver tractable; the reported "agents changed" is the total
  across groups.
* **ADOPT needs non-negative costs.** It seeds child lower bounds at 0, so a
  negative cost silently breaks optimality. Maximisation problems are
  re-baselined rather than negated, and `solve_adopt` refuses negative costs
  outright.
* **Search budgets must be deterministic.** CBS was first bounded by wall-clock
  time, which made the same seed give 54% and 86% task completion on two runs.
  It is now bounded by low-level search count, and a test asserts three
  consecutive runs agree exactly.
