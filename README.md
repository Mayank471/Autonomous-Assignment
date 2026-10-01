# Multiagent Plan Repair in an Automated Warehouse

A fleet of robots on a 2D warehouse grid plans collision-free pickup-and-delivery
routes with prioritized Space-Time A\*, then **repairs** those plans locally when
the environment disrupts them — negotiating with neighbouring agents rather than
regenerating the joint plan, and minimising how many agents are disturbed.

Built for *Autonomous Systems*, following Weiss, **Multiagent Systems** (2nd ed.)
and Modi et al., **"Adopt: asynchronous distributed constraint optimization with
quality guarantees"**, *Artificial Intelligence* 161 (2005).

---

## The idea in one paragraph

The textbook supplies the bridge between its two relevant chapters on p497:

> "this view of multiagent planning is compatible with a distributed constraint
> satisfaction formulation, **where the variables are the agents' plans, and the
> constraints enforce that the plans dovetail together suitably**."

So a repair episode *is* a DCOP. When a disruption invalidates some plans, the
affected agents and a bounded ring of their neighbours each generate a handful of
candidate plans; binary constraints forbid pairs of plans that collide in
space-time; and the assignment's requirement to *"minimise the number of agents
whose original plans are altered"* becomes a term in the objective function
rather than a hope. ADOPT then solves it optimally, distributed and asynchronous.

---

## Quick start

```bash
pip install -r requirements.txt

python -m pytest tests/ -q             # 180 tests, ~2 min
python experiments/demo_dcop.py        # the textbook + paper DCOP examples
python experiments/run_experiments.py --quick   # pipeline smoke test, ~30s
python experiments/run_experiments.py           # full sweep (~30 min, checkpointed)
python experiments/summarise_results.py         # the report's tables
python experiments/make_figures.py              # figures from results/runs.csv
python experiments/make_animation.py            # animated GIF of a run
python experiments/demo_run.py --compare        # one scenario, all four strategies
```

---

## Where the syllabus material lives

| Textbook / paper | Implemented in |
|---|---|
| Ch 11 §3.1 Social laws and conventions | [`sociallaws.py`](warehouse/sociallaws.py) — one-way aisles, with a strong-connectivity guard |
| Ch 11 §3.2 Organizational structuring, and PGP's meta-level organization | [`organization.py`](warehouse/organization.py) — authority ranking, zone coordinators |
| Ch 11 §3.3 Contract-net protocol | [`contractnet.py`](warehouse/contractnet.py) — re-auctioning a broken agent's tasks |
| Ch 11 §4 Local planning prior to coordination (MPCP) | [`planner.py`](warehouse/planner.py) — plan independently, then coordinate |
| Ch 11 §4.3 Algorithm 11.2, superior/inferior protocol | [`repair/solvers.py`](warehouse/repair/solvers.py) — agents settle in authority order |
| Ch 11 §6.1–6.2 Monitoring and recovery | [`disruptions.py`](warehouse/disruptions.py), [`repair/episode.py`](warehouse/repair/episode.py) |
| Ch 11 §6.3 PGP's six mechanisms | [`abstractplan.py`](warehouse/abstractplan.py) + [`repair/`](warehouse/repair/) — see below |
| Ch 12 §2 Constraint networks, DCOP | [`dcop/model.py`](warehouse/dcop/model.py) — `N = <X, D, C>`, agent partition |
| Ch 12 §3.1.2 Target tracking | [`dcop/examples.py`](warehouse/dcop/examples.py), applied in [`emergency.py`](warehouse/emergency.py) |
| Ch 12 §4.1 / Modi et al. — ADOPT | [`dcop/adopt.py`](warehouse/dcop/adopt.py) |

### PGP's six mechanisms, concretely

Ch 11 §6.3 (p530–531) lists six mechanisms PGP uses to keep continuous
replanning affordable. Each has a specific realisation here:

1. **Abstraction** — agents exchange zone-occupancy intervals, not cell paths
   (about 6× smaller). `abstractplan.py`
2. **Decentralization (MLO)** — an authority ranking inherited from the planning
   priority decides who mediates and who yields. `organization.py`
3. **Partial global reasoning** — greedy hill-climbing over the merged plan view,
   exactly the textbook's description. `repair/solvers.py::solve_greedy_pgp`
4. **Communication planning** — messages are explicit objects and are counted.
5. **Asynchrony** — repair touches only the involved agents; the rest of the
   fleet keeps executing. There is no stop-the-world barrier.
6. **Dampened responsiveness** — the `tau` threshold: a change smaller than
   `tau` is absorbed and never broadcast. `AbstractPlan.differs_beyond`

---

## The five strategies compared

| Strategy | Searches | What it does |
|---|---|---|
| `selfish` | — | Only disrupted agents replan; nobody yields. Lower bound on agents changed — and it shows what that bound costs. |
| `greedy_pgp` | candidate set | PGP mechanism 3: distributed hill-climbing. Fast, no optimality guarantee. |
| `adopt` | candidate set | The same repair posed as a DCOP and solved optimally by ADOPT. |
| `cbs` | the plan space | Conflict-Based Search over the neighbourhood. **Outside the syllabus** — see below. |
| `full_replan` | the plan space | Prioritized replanning of the whole fleet — the baseline the assignment says to avoid, kept so the cost of avoiding it is measurable. |

All five are handed the *same* scenarios and the *same* disruption schedules, so
differences are attributable to the repair alone.

### Why CBS was added

The four syllabus strategies share a limit that only showed up in the data: the
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

ADOPT is validated against **both** taught examples:

* **Weiss Figure 12.1** — the DFS pseudo-tree of Figure 12.2, the δ values quoted
  on p558, and the optimum of 1. [`tests/test_adopt_textbook.py`](tests/test_adopt_textbook.py)
* **Modi et al. Figure 2** — a *different* network (x1–x4 are not neighbours,
  different cost table), its stated totals `F(all 0) = 4` and `F(all 1) = 0`, its
  DFS tree x1→x2→{x3,x4}, and its optimum `A* = all ones` at cost 0.
  [`tests/test_adopt_paper.py`](tests/test_adopt_paper.py)

plus cross-checks against exhaustive search on random networks.

---

## Layout

```
warehouse/
  grid.py  plan.py  reservation.py  obstacles.py  stastar.py   the world and the search
  tasks.py  planner.py  sociallaws.py  organization.py         initial plan generation
  contractnet.py  disruptions.py  emergency.py                 what goes wrong, and who responds
  abstractplan.py                                              PGP abstraction + dampening
  dcop/     model  dfstree  adopt  bruteforce  examples        Ch 12 / Modi et al.
  repair/   episode  candidates  solvers                       the plan-repair layer
            cbs                                                CBS (beyond the syllabus)
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
