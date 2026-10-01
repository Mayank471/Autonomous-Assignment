# Multiagent Plan Repair in an Automated Warehouse

*Autonomous Systems — plan generation, disruption, and distributed repair*

---

## 1. Problem

A team of robots works the floor of an automated warehouse laid out as a 2D
grid. Each robot starts at a dock, is assigned a set of objects to collect, and
must carry each to its delivery station. Routes are planned centrally at the
start with a multiagent pathfinding algorithm, and are collision-free by
construction.

Then the environment refuses to cooperate. A shelf collapses and a grid cell
becomes impassable; a robot breaks down where it stands and becomes an obstacle
with unfinished work on its hands; an emergency appears somewhere on the floor
and someone has to attend to it now. Plans that were valid a moment ago are not.

The system must respond by **repairing** the affected plans, subject to three
constraints that between them define the problem:

1. the disrupted agents negotiate with **neighbouring agents** to agree
   alternative paths;
2. the repair must **not re-invoke the multiagent plan generation algorithm**
   from scratch;
3. the repair should **minimise the number of agents whose original plans are
   altered**.

and it is evaluated on:

* the **total time steps** required by all agents to accomplish their tasks;
* the **number of agents whose plans change** to handle a single disruption;
* how both **scale** with the number of agents and with the density of dynamic
  obstacles.

## 2. Background, and where the design comes from

The design is assembled from the taught material rather than invented alongside
it. Two chapters of Weiss do most of the work, and the textbook itself supplies
the join between them. Discussing the multiagent plan coordination problem, it
observes (p497):

> "this view of multiagent planning is compatible with a distributed constraint
> satisfaction formulation, **where the variables are the agents' plans, and the
> constraints enforce that the plans dovetail together suitably**."

That sentence is the thesis of this implementation. A repair episode is treated
as a distributed constraint optimization problem in exactly those terms: the
variables are the agents' candidate plans, the constraints forbid plans that
collide, and — the part the assignment cares about — *minimising how many agents
are altered* is written into the objective function rather than hoped for.

### 2.1 What Chapter 11 supplies

**Coordination prior to local planning (§3).** Some coordination is cheapest
done once, at design time. Social laws (§3.1) are conventions that "relieve
agents of the burden of explicitly coordinating"; here that is one-way aisles.
Organizational structuring (§3.2) provides roles and authority; here that is a
zone decomposition and a ranking over agents. The contract-net protocol (§3.3)
handles task reallocation; here it re-auctions a broken robot's remaining work.

**Local planning prior to coordination (§4).** The initial plan follows the
chapter's own two-phase structure. Every agent first plans its route alone; the
union of those routes is the initial, uncoordinated multiagent plan — the
analogue of the flawed parallel plan in Figure 11.4 — and coordination then
resolves the interactions. Algorithm 11.2's step 5 describes the resolution
protocol used: agents form a total order, the superior sends down its plan, and
inferiors change theirs to work with it while double-checking against previous
superiors. Prioritized planning is precisely that protocol.

**Execution (§6).** Section 6.2 states the dilemma this assignment is built
around, and warns about the trap:

> "Plan repair is potentially more cost-efficient, but more complicated. If only
> some agents need to repair their plans (the others have not deviated), repair
> nonetheless could involve all of the agents as the repaired plans need to be
> coordinated with others' unchanged plans. This could lead to others needing to
> change their plans, **in a chain reaction** of planning and coordination
> efforts."

Bounding that chain reaction is the central engineering problem, and §6.3's
account of **partial global planning** supplies the means. PGP's six mechanisms
map onto the repair layer one for one:

| PGP mechanism (p530–531) | In this system |
|---|---|
| **Abstraction** — agents exchange "what partial interpretations they plan to construct and when", not their detailed plans | agents exchange zone-occupancy intervals `(zone, enter, exit)`, ~6× smaller than the cell path, and negotiate over those |
| **Decentralization** — a meta-level organization defining "roles, protocols, and authority structures" | an authority ranking inherited from the planning priority; the disrupted agent mediates its own episode |
| **Partial global reasoning** — combine others' abstract plans with your own and search modifications "in a greedy hill-climbing fashion" | the `greedy_pgp` strategy, literally that |
| **Communication planning** | messages are explicit objects and are counted as a metric |
| **Asynchrony** — "agents cannot afford to wait for convergence" | uninvolved agents keep executing throughout a repair; there is no barrier |
| **Dampened responsiveness** — "if an abstract plan step will occur earlier or later than expected by more than a parameterized number of time steps, then the agent should alert others" | the threshold `tau`; changes below it are absorbed and never broadcast |

### 2.2 What Chapter 12 and the ADOPT paper supply

Chapter 12 gives the formal object: a constraint network `N = <X, D, C>` with
variables partitioned among agents, where "two agents are considered neighbors if
there is at least one constraint that depends on variables that each controls"
and "only neighboring agents can directly communicate with each other". Hard
constraints are encoded as soft ones priced prohibitively (p550), which is how
collisions are expressed here.

ADOPT itself was taught from Modi, Shen, Tambe and Yokoo (2005) rather than from
the textbook, and both presentations are reproduced (§5). Its three components —
local lower-bound estimates, backtrack thresholds, and bound-interval termination
— are implemented as described, with `VALUE`, `COST`, `THRESHOLD` and `TERMINATE`
messages over a DFS pseudo-tree, and the paper's bounded-error extension
(`min(LB + b, UB) = threshold` at the root) is available as a real-time escape
hatch.

The target-tracking problem of §3.1.2 is used twice: as a standalone
demonstration, and as the actual mechanism for deciding which robots divert to an
emergency.

---

## 3. System design

### 3.1 The warehouse

A Kiva-style floor: shelf blocks in a lattice of single-width aisles, ringed by a
two-wide highway. The default layout is 50×60 with 1650 free cells, 12 delivery
stations and 216 perimeter docking bays.

One modelling decision shapes everything downstream. An agent that finishes its
route **stays on its final cell for the rest of the run**, so that cell is a
*permanent* reservation. With far fewer delivery stations than agents, parking on
one would make it unreachable for every agent routed there afterwards — they
could never complete. Agents therefore return to a **dedicated dock**, placed on
the outermost ring where reserving it cannot disconnect the floor (the inner
highway row still joins every aisle). This single change took the 30-agent
instance from repeated planning failures at 73 s to a clean solution in 0.11 s.

### 3.2 Initial plan generation

Space-Time A\* over `(cell, time)` states with wait actions, guided by an exact
backward-BFS distance-to-goal cached per goal. Other agents enter only through a
reservation table holding vertex, edge (head-on swap) and permanent reservations,
so the same routine serves independent planning, prioritized planning and repair
unchanged. Agents are then planned in priority order, longest route first, with
randomised priority restarts on failure.

### 3.3 The repair layer

When a disruption lands:

1. **Detect** which agents' remaining plans it invalidated (§6.1 monitoring).
2. **Gather a neighbourhood.** Agents are linked when their zone-time intervals
   overlap within a window around the disruption; a ring is grown outward from
   the disrupted agents, strongest interaction first, capped at `k_max`. Working
   from abstract plans makes this one pass over intervals rather than an
   all-pairs comparison of paths.
3. **Generate options locally.** Each agent produces its own small domain, in
   increasing order of disturbance: `keep`, `wait_k` (a pure delay insertion —
   the route is untouched, every step happens `k` ticks later), `splice`,
   `detour`, and `reroute` only if none of those worked.
4. **Add one coordinated option per agent** (§3.4).
5. **Negotiate**, by hill-climbing or by ADOPT.
6. **Commit atomically**, leaving uninvolved agents running.

**Splicing is the heart of "repair, not replan".** Only the leg the agent is
currently flying is re-planned; the remainder of its route is reattached where
that leg ends, sliding later in time to meet it. That is one search instead of
one per remaining waypoint. It is also where nearly all the system's cost used to
sit: replacing full rerouting with splicing cut a 20-agent run from 39.5 s to
2.9 s, a factor of 14.

### 3.4 Making the DCOP feasible

Generating each agent's options independently is not a convenience, it is what
the DCOP formulation demands: no agent may enumerate anyone else's choices. But
that has a consequence which is easy to miss, and which only surfaced once the
system was measured rather than reasoned about.

Several agents in a neighbourhood routinely plan through the same free corridor,
because each is planning around the *uninvolved* fleet and is blind to the
others' options. Their candidates therefore collide with one another, and with
only a handful of options each there may be **no collision-free combination at
all**. Instrumenting a 16-agent floor showed 13 of 16 episodes had no feasible
joint assignment: the solver's answer was rejected as unsafe and the sequential
fallback quietly did the real work 62% of the time. The negotiation was
decorative — which also explains why the `beta` knob appeared to do nothing, as
the assignment it influenced was being discarded.

The fix is one further option per agent, planned in the meta-level
organization's authority order against everything already settled: the superior
commits, and each inferior adapts around it. That is precisely Algorithm 11.2's
step 5, but offered as a *candidate* rather than imposed as the outcome. It makes
"everybody takes their coordinated plan" feasible by construction, so the DCOP
starts from a solvable problem and searches for something strictly better —
fewer agents disturbed, less delay — instead of failing and being overruled.
Afterwards every episode was feasible, the fallback stopped firing, and both the
churn and the total cost improved.

### 3.5 The repair DCOP

For each involved agent `i`, variable `x_i` ranges over its candidate plans.

* **Binary constraints.** `F_ij(a, b) = conflict_cost` when plans `a` and `b`
  collide in space-time, else 0 — a hard constraint encoded as a soft one, per
  p550. Pairs with no colliding combination get no edge, keeping the graph as
  sparse as the problem really is.
* **Unary costs.** `f_i(a) = alpha · delay(a) + beta · [a ≠ keep]`.

The `beta` term **is** the assignment's "minimise the number of agents whose
original plans are altered", stated as an objective. At `beta = 0` the system
buys any time saving with any amount of churn; as `beta` rises it will accept a
slower joint plan in exchange for leaving more agents alone. Sweeping it traces
the trade-off directly (§6.4).

Delay is measured against each agent's *own best* option rather than its current
plan, which keeps every cost non-negative — a requirement of ADOPT, not a detail
(§5.3) — while shifting the objective only by a constant per agent, so the
optimal joint assignment is unchanged.

### 3.6 Bounding the chain reaction

Three mechanisms bound propagation, which is what §6.2 warns is otherwise
unbounded:

* `k_max` caps a single negotiation group.
* A disruption invalidating more plans than one group can hold is repaired in
  **successive small groups**, each committing before the next begins — local
  huddles rather than one fleet-wide meeting. This matters for tractability: a
  breakdown in a busy aisle routinely invalidates a dozen plans, and a complete
  DCOP solver cannot cope with a dense thirteen-variable problem.
* `tau` suppresses announcements of changes too small to matter.

### 3.7 Handling each kind of disruption

**Cell blockage** — a permanent vertex constraint from that time on; every agent
routed through it is a trigger.

**Agent breakdown** — the agent freezes and its cell becomes an obstacle; its
unfinished tasks are re-auctioned by contract net, with bids priced at the
marginal cost of accepting; agents routed through its cell are triggers.

**Emergency task** — posed as the target-tracking DCOP of Ch 12 §3.1.2, with the
mapping: robots are sensors, "which target to aim at" is the response modality,
overlapping sensing range is "can reach the emergency in time", and the energy
cost of aiming is the delay to that robot's own deliveries. The chosen responders
get the emergency spliced to the front of their queues, and an ordinary repair
episode re-plans them.

### 3.8 What happens when repair fails

An agent that cannot complete its route has to go *somewhere*, and "stop where
you are" is not safe: a stationary robot occupies its cell forever, so any agent
already routed through that cell later would drive into it. Parking is a planning
problem in its own right. The fallback order is: route to the agent's own dock
(safe by construction, since a dock is exclusive to it); else advance as far as
it can and stop somewhere holdable; else stand still, but only if standing still
is verified safe.

Likewise, a negotiated assignment is never trusted to be feasible: when every
joint choice in the domains collides, an optimal solver returns the least-bad
one, not a legal one. Every result is checked, and anything that fails falls back
to sequential local repair — still bounded to the episode's agents, but
collision-free by construction. How often that happens is reported, not hidden.

Even with all of that, a first sweep of 1440 runs turned up twelve in which a
collision survived, every one of them on a badly degraded floor with many
stranded agents. The cause was the last rung of the parking ladder: when an
agent's dock was unreachable *and* every nearby cell was already spoken for, it
was left standing where it was, unchecked. Widening that search fixed the cases
observed, but "fixed the cases observed" is not an invariant, so the simulator
now also runs a final pass that rebuilds the joint plan in authority order and
trims any agent that cannot be accommodated — a queue forming behind a stalled
robot. It terminates because plans only ever shorten, it leaves conflict-free
plans untouched, and how often it has to intervene is itself a reported metric.

---

## 4. The five strategies

| | Negotiates? | Searches | Role |
|---|---|---|---|
| `selfish` | no | — | Only disrupted agents replan, against everyone else held fixed. Lower bound on agents changed. |
| `greedy_pgp` | yes | fixed candidate set | PGP mechanism 3: distributed hill-climbing in authority order. |
| `adopt` | yes | fixed candidate set | The same repair as a DCOP, solved optimally by ADOPT. |
| `cbs` | yes | the plan space | Conflict-Based Search over the neighbourhood. **Outside the syllabus** — see §4.1. |
| `full_replan` | — | the plan space | Prioritized replanning of the whole fleet. The baseline the assignment says to avoid. |

All five receive the same scenarios and the same disruption schedules, so
differences are attributable to the repair alone.

### 4.1 Going outside the syllabus: Conflict-Based Search

The four syllabus strategies share a limitation that only became visible once
the system was measured. The DCOP formulation asks each agent for a handful of
candidate plans and then asks which combination fits. That is exactly what Ch 12
prescribes — an agent controls its own variable and knows nothing of anyone
else's options — but it means the negotiation can only ever choose among about
five plans per agent. When no combination of those is collision-free, agents
abandon their remaining tasks and retreat.

The consequence was stark: across 1152 disrupted runs the negotiated strategies
stranded **more** agents than the naive `selfish` baseline (14.25 against 11.90)
and abandoned roughly nine agents' tasks per run. Losing to the do-nothing
baseline on task completion is not a parameter that needs tuning; the search
space is simply too small.

**Conflict-Based Search** (Sharon, Stern, Felner and Sturtevant, *Artificial
Intelligence* 219, 2015) removes that limitation. It does not enumerate plans in
advance. Its high level searches a tree of *constraints*: find a conflict
between two agents, branch by forbidding one of them from that cell at that
time, and replan just that agent. The reachable space is therefore every plan,
not a sample of five, and CBS is complete — if a collision-free joint plan
exists for those agents, it finds one. No agent gives up merely because the
options it happened to generate did not fit together.

It also turns out to suit this assignment's objective unusually well, and for
free. The root node holds the agents' *current* plans, and an agent is replanned
only when a constraint is placed on it — so an agent not party to any conflict
is never touched. The set of agents that change is exactly the set the conflicts
reach. Where the DCOP has to *price* churn through `beta` and hope the optimum
lands somewhere sensible, CBS minimises it structurally.

Two costs come with it. CBS is exponential in the number of conflicts, so the
search is bounded and falls back to sequential local repair when the bound is
hit; and it is a centralised search over the neighbourhood, so unlike ADOPT it
is not a distributed algorithm. It is bolted onto the same neighbourhood
machinery — the same interaction graph, the same `k_max`, the same commit path —
so the comparison isolates the search and nothing else.

**Adaptive escalation.** Because CBS searches the real plan space, a failure is
evidence that the *neighbourhood* was too narrow rather than that no repair
exists. So a failed CBS episode is retried on a wider ring before giving up,
which turns `k_max` from a fixed guess into a floor. This is only worth doing
for a solver that can exploit the extra room; retrying a fixed candidate set on
more agents mostly just costs more.

---

## 5. Validation

### 5.1 ADOPT against Weiss Figure 12.1

The implementation reproduces the taught example exactly: the DFS pseudo-tree of
Figure 12.2 (x1 root, x2 and x3 its children, x4 under x2, and **x1 as x4's
pseudo-parent** — which is the figure's own point about why x1 sends x4 a `VALUE`
message despite not being its parent); the local costs quoted on p558
(`δ(x4=0) = 4`, `δ(x4=1) = 0` under context `{x1=0, x2=0}`; `δ(x2=0) = 2`,
`δ(x2=1) = 0` under `{x1=0}`); and the optimum of 1, confirmed against exhaustive
search. All four message types are exercised, and `LB = UB = 1` at termination.

### 5.2 ADOPT against Modi et al. Figure 2

The paper's example is a **different network**: its edges are x1-x2, x1-x3,
x2-x3, x2-x4, so "x1 and x3 are neighbors but x1 and x4 are not" — the reverse of
the textbook — and its cost function is `f(0,0)=1, f(0,1)=f(1,0)=2, f(1,1)=0`.
Both of the totals the paper quotes are reproduced (`F(all 0) = 4`,
`F(all 1) = 0`), as is the DFS tree of Fig. 2(b) (x1 root, x1 parent of x2, x2
parent of both x3 and x4, x1 pseudo-parent of x3) and the stated optimum
`A* = {(x1,1), (x2,1), (x3,1), (x4,1)}` at cost 0 — found whichever agent is
chosen as root.

The paper's §6 bounded-error extension is implemented as specified, relaxing the
**root's** threshold invariant to `min(LB + b, UB)` while every other agent keeps
the strict one. That placement is what preserves the guarantee: the cost returned
is the root's own upper bound, so it is provably within `b` of the optimum.

### 5.3 A trap worth recording

ADOPT initialises every child's lower bound to 0 and refines upward. That is only
an admissible bound if no subtree can cost *less* than 0 — so a **negative cost
silently breaks optimality**, closing `LB = UB` on a suboptimal answer. The
target-tracking formulation hit this: written naturally as a reward of `-10` for
a covered target, ADOPT confidently returned "all sensors idle". Re-baselining it
as the *cost of not covering* (0 when a pair agrees, `reward` otherwise) is a
constant shift per edge, leaves the optimum unchanged, and is non-negative.
`solve_adopt` now refuses negative costs outright rather than returning a wrong
answer quietly.

### 5.4 Safety of the repaired plans

The property that matters most: **after every disruption is repaired, the joint
plan contains no vertex or edge collision** — for every strategy, across seeds
and obstacle densities, checked mid-run rather than only at the end. A repair
that produced a faster schedule by driving two robots through each other would
not be a repair.

Also asserted: agents never teleport or stand on a shelf; the local strategies
**never call global plan generation** (the assignment's core constraint, asserted
rather than assumed); negotiation groups never exceed `k_max`; raising `beta`
never increases churn; raising `tau` never increases what is broadcast; and an
undisrupted run completes every task with no repairs at all.

### 5.5 Determinism

Every configuration must be reproducible from its seed, or the evaluation means
nothing. That is asserted rather than assumed, and the assertion earned its
keep: the first version of CBS bounded its search by **wall-clock time**, which
bounds cost correctly but makes the result depend on how busy the machine is.
Two runs of one configuration, same seed, same scenario, disagreed by 34
percentage points of task completion (54% against 86%) — and neither run looked
anomalous on its own. It was caught only because a repeat happened to disagree
with itself.

The bound is now on *low-level searches*, a quantity the search counts itself,
and `test_cbs_is_deterministic` checks three consecutive runs agree exactly. The
general lesson: a real-time bound is the right engineering choice for a
deployed repair layer and the wrong one for a measured one, and if a system is
going to be evaluated then reproducibility has to outrank realism.

### 5.6 What the tests missed, and what found it

Worth recording, because it shaped the final design. The test suite ran the
safety invariant over dozens of scenarios and passed — and the invariant was
still being violated. A sweep of 1440 runs found twelve in which a collision
survived, every one at 60–80 agents on a heavily blocked floor.

The tests use 12–16 agents on a 30×36 floor, which keeps the suite fast, and at
that size the failing path essentially never fires: it needs an agent whose dock
is unreachable *and* whose every nearby cell is already claimed. That only
happens when the floor is both crowded and badly degraded. The bug was not
subtle once seen, but it was invisible at test scale.

Two conclusions were drawn and applied. The first is that a repair layer which
can *fabricate* a plan when it fails is worse than one that reports failure: the
fabricated plan satisfies every downstream check while being physically wrong, so
the error surfaces far from its cause. `find_refuge` now returns `None` rather
than inventing a standstill, and its callers must decide what to do. The second
is that when placing agents that cannot move, ordering matters more than
seniority — an immovable agent is an obstacle, not a participant, and must be
placed before the movers so they route around it. Ordering by authority instead
let a senior mover claim the cell a stalled agent was standing on, which is
unfixable on that pass.

The large-scale sweep is therefore not only the evaluation; it is also the only
thing that exercised these paths at all.

---

## 6. Experimental setup

**Layout.** A 50x60 Kiva warehouse: 1650 free cells, 1020 shelf-access points,
12 delivery stations, 216 perimeter docking bays, single-width aisles.

**Fleet.** 10 to 80 agents, three pickup-and-delivery tasks each, so the largest
configuration is 240 tasks over 480 waypoints.

**Disruption density.** The fraction of free cells that become permanently
blocked over a run, swept over 0, 0.01, 0.02, 0.03 and 0.04. The top of that
range walls off about 65 cells of 1650 — a severe incident, not a mild one. Going
much further stops measuring plan repair and starts measuring how long a
destroyed warehouse takes to grind to a halt. Roughly 70% of events are cell
blockages, 15% agent breakdowns and 15% emergency tasks, drawn with a bias
toward cells the fleet actually intends to use: a blockage on a cell nobody was
going to visit is not a disruption at all.

**Repetitions.** Twelve seeds per configuration, 1672 runs in total. Every
strategy is replayed against an identical scenario and an identical disruption
schedule, drawn once up front, so a paired comparison is valid. Error bars are
95% confidence intervals over seeds.

**Defaults.** `k_max = 6`, `alpha = 1`, `beta = 4`, `tau = 0`, `epsilon = 0`
(ADOPT exact), social laws off, with each swept separately.

**Metrics.** Total time steps summed over agents; agents whose plan changed per
disruption; agents consulted per disruption; largest negotiation group; repair
wall-clock time and message count; tasks completed; agents stranded; and — as a
correctness check rather than a result — collisions in the final joint plan.

An agent that never finishes is charged to the end of the run rather than having
its clock stopped early, so a strategy cannot look fast by stranding agents.
Emergency tasks are added to the task denominator, so completion rates cannot
exceed 100% by counting extra work.

### 6.1 A caveat on the impact metric

"Agents whose plans change per disruption" is the number the assignment asks
for, and taken alone it is **misleading**. A strategy scores well on it by
repairing efficiently — or by giving up.

The clearest case came from the hardest configuration tested, 80 agents on a
floor with 4% of its cells blocked:

| | agents changed | tasks completed | stranded |
|---|---|---|---|
| ADOPT | **4.10** | **12%** | 63 of 80 |
| CBS | 6.70 | **74%** | 31 of 80 |

ADOPT looks 40% better on the metric the assignment names. It achieves that by
abandoning: agents give up their tasks and park, and a parked agent's plan never
needs changing again. The fleet delivered an eighth of its work.

So every impact figure in what follows is reported next to task completion, and
neither should be read without the other. A repair layer that changes nobody's
plan because nobody is still trying to do anything is not a good repair layer.

---

## 7. Results

2048 runs, twelve seeds per configuration. **No joint plan in any run contains a
collision**, for any strategy, at any fleet size or obstacle density.

Read §6.1 first: "agents changed" is reported beside task completion throughout,
because on its own it rewards giving up.

### 7.1 The headline comparison

Averaged over every disrupted run:

| strategy | total time steps | agents replanned per disruption | tasks done | repair time | agents abandoned/run |
| --- | --- | --- | --- | --- | --- |
| selfish | 16909 ± 1040 | 5.48 ± 0.30 | 72.6% | 43.2 ms | 7.59 |
| greedy PGP | 16990 ± 1061 | 5.50 ± 0.33 | 69.1% | 9.3 ms | 7.61 |
| ADOPT | 16947 ± 1062 | 5.38 ± 0.33 | 68.5% | 6.9 ms | 7.08 |
| **CBS** | 17366 ± 1064 | **3.80 ± 0.22** | 69.7% | 122.8 ms | **1.10** |
| full replan | 15432 ± 966 | 18.88 ± 1.76 | 81.5% | 1876.4 ms | 0.00 |

Against the syllabus strategies, CBS alters **29% fewer plans per disruption**
than the DCOP while completing slightly more work, and abandons **six times
fewer** agents. Against replanning the whole fleet it alters **80% fewer** plans
and decides **15× faster**, for 12% more total time steps and 12 points of task
completion.

The four candidate-set strategies are almost indistinguishable from one another
— 5.38 to 5.50 agents changed, within each other's confidence intervals. Whether
the negotiation hill-climbs or solves the DCOP optimally barely matters. What
matters is the size of the space being searched, which is the one thing they
have in common and the one thing CBS changes.

### 7.2 Scaling with fleet size

**Agents replanned per disruption:**

| agents | selfish | greedy PGP | ADOPT | CBS | full replan |
| --- | --- | --- | --- | --- | --- |
| 10 | 2.22 | 2.05 | 1.99 | **1.49** | 2.72 |
| 20 | 3.22 | 3.17 | 3.05 | **2.09** | 6.19 |
| 30 | 4.74 | 4.78 | 4.63 | **3.24** | 11.03 |
| 40 | 5.83 | 5.91 | 5.74 | **4.11** | 17.72 |
| 60 | 7.63 | 7.62 | 7.62 | **5.38** | 30.74 |
| 80 | 9.27 | 9.46 | 9.26 | **6.50** | 44.86 |

Full replanning alters a number of agents that grows *linearly* with the fleet —
2.7 at ten agents, 44.9 at eighty, which is 56% of everyone. Local repair grows
roughly with the square root. CBS is the best at every fleet size and its margin
widens: 25% better than ADOPT at ten agents, 30% at eighty, and **86% better
than replanning** at eighty.

**Repair time per disruption (ms):**

| agents | greedy PGP | ADOPT | CBS | full replan |
| --- | --- | --- | --- | --- |
| 10 | 2.1 | 3.2 | 22.2 | 58.3 |
| 20 | 3.7 | 4.6 | 42.1 | 284.3 |
| 40 | 9.2 | 7.4 | 104.4 | 1431.1 |
| 60 | 13.7 | 9.0 | 200.6 | 2391.3 |
| 80 | 19.8 | **10.1** | 283.6 | **6357.0** |

ADOPT's repair time is very nearly flat in fleet size (3.2 ms to 10.1 ms) because
its work is bounded by `k_max`, not by the fleet. CBS is bounded the same way but
its constant is far larger — every node of its constraint tree costs a full
multi-leg replan — so it lands between the DCOP and full replanning, roughly 28×
the former and 22× cheaper than the latter. Full replanning reaches 6.4 seconds
per disruption at eighty agents, which on a floor where disruptions arrive every
few seconds is no longer a repair strategy at all.

### 7.3 Scaling with obstacle density

**Agents replanned per disruption:**

| density | selfish | greedy PGP | ADOPT | CBS | full replan |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.01 | 5.52 | 5.38 | 5.31 | **3.97** | 20.90 |
| 0.02 | 5.46 | 5.53 | 5.36 | **3.92** | 18.89 |
| 0.03 | 5.44 | 5.51 | 5.40 | **3.69** | 18.29 |
| 0.04 | 5.50 | 5.57 | 5.46 | **3.64** | 17.41 |

Impact per disruption is **flat** in density for every strategy — quadrupling the
obstacle density leaves it essentially unchanged. Density determines *how many*
disruptions occur, not how far each one propagates: the blast radius is set by
the local structure of the floor and by `k_max`, neither of which depends on how
damaged the rest of the warehouse is. This is the cleanest result in the study
and it holds across all five strategies.

CBS's figure drifts slightly *downward* as the floor degrades (3.97 to 3.64),
which is not noise in the expected direction: on a more blocked floor there are
fewer legal alternatives, so the constraint tree closes sooner and fewer agents
end up constrained.

Total cost roughly doubles from the undisrupted baseline (10543 steps) to 4%
density (about 21000), and full replanning's repair time more than triples
(911 ms to 3273 ms) as each replan runs on a more congested floor. CBS's repair
time rises only 30% (111 ms to 145 ms), and ADOPT's 23%.

### 7.4 Why CBS wins: completeness, not cleverness

| strategy | repair failures/run | agents abandoned/run | safety-net trims/run |
| --- | --- | --- | --- |
| selfish | 4.24 | 7.59 | 0.04 |
| greedy PGP | 4.15 | 7.61 | 2.94 |
| ADOPT | 3.81 | 7.08 | 3.08 |
| **CBS** | **0.53** | **1.10** | **0.69** |
| full replan | 23.51 | 0.00 | 0.00 |

This is the mechanism behind every other CBS number. The candidate-set strategies
fail to find a repair about four times per run and abandon roughly seven agents'
tasks; CBS fails half a time and abandons one. Searching the actual plan space
rather than five samples of it means an agent rarely has to give up — and an
agent that keeps working keeps a plan that can be repaired later, rather than
becoming a permanent obstacle that forces *other* agents to replan.

That also explains why CBS changes fewer plans while completing more work: the
two are not in tension, they have a common cause.

`full_replan`'s 23.51 "failures" are a different thing — individual agents its
prioritized ordering could not place on a given attempt, which the restart
mechanism then resolves. It abandons nobody, which is what the whole-fleet
replan buys.

### 7.5 The `beta` knob

| beta | agents changed | total steps |
| --- | --- | --- |
| 0 | 4.96 | 13490 |
| 1 | 4.81 | 13644 |
| 4 | 4.57 | 13674 |
| 10 | 4.45 | 13756 |
| 60 | 4.46 | 13779 |

Raising the penalty for altering a plan reduces churn by 10% for about 2% more
total time steps, then saturates past `beta = 10` — what remains are agents that
genuinely cannot keep their plans. Note the scale: the entire useful range of
this knob (0.5 agents) is smaller than the gap CBS opens by changing the search
(1.6 agents). Tuning the objective is worth less than fixing what it is
optimising over.

### 7.6 PGP's dampening threshold

| tau | changes broadcast | changes actually made | total steps |
| --- | --- | --- | --- |
| 0 | 4.44 | 4.57 | 13674 |
| 2 | 3.92 | 4.57 | 13674 |
| 5 | 3.61 | 4.57 | 13674 |
| 10 | 3.58 | 4.57 | 13674 |

Exactly what PGP's mechanism 6 predicts: slack cuts coordination traffic without
touching the plans. At `tau = 5`, 19% of changes are absorbed silently and the
total cost is *identical*. Beyond `tau = 10` it flattens — what remains are
changes too large to hide.

### 7.7 The neighbourhood cap

| k_max | agents changed | total steps | repair ms |
| --- | --- | --- | --- |
| 2 | 4.18 | 13894 | 1.4 |
| 4 | 4.38 | 13604 | 3.0 |
| 6 | 4.57 | 13674 | 6.8 |
| 8 | 4.87 | 13907 | 13.3 |
| 10 | 5.02 | 13330 | 33.9 |

A wider huddle disturbs more agents and costs sharply more to solve — repair time
rises 24× from `k_max = 2` to `10`, which is the exponential in ADOPT showing
through — while total time steps barely move. Little to gain past six on this
floor.

### 7.8 ADOPT's error bound

| b | repair ms | cycles | messages |
| --- | --- | --- | --- |
| 0 | 7.1 | 10.7 | 202 |
| 10 | 5.9 | 7.8 | 163 |
| 50 | 5.7 | 7.0 | 148 |

Bounded-error approximation buys about 20% in time and 27% in messages. Modest
here because the neighbourhoods are small; the mechanism matters more at larger
`k_max`, where the exponential binds.

### 7.9 Social laws

One-way aisles made things **worse** for every strategy — 2.7% to 4.7% more total
time steps. On single-width aisles with a two-wide perimeter, the convention
removes head-on conflicts that prioritized planning already resolved cheaply, and
in exchange forces a detour on every agent travelling against the flow. Reported
because it is a genuine negative result: a social law pays only when the
coordination it saves exceeds the flexibility it removes, and here it does not.

---

## 8. Conclusions

The assignment's three questions, answered directly.

**Total time steps.** Local negotiated repair costs 10–12% more than replanning
the whole fleet after every disruption (16947 for ADOPT, 17366 for CBS, against
15432). Replanning globally produces better schedules; the comparison measures
what locality costs, not planning quality.

**Agents altered per disruption.** 3.80 for CBS and 5.38 for ADOPT, against 18.88
for full replanning — 5.0× and 3.5× fewer. At eighty agents the gap is widest:
6.50 and 9.26 against 44.86, so local repair touches a seventh of what a global
replan does. This holds because the neighbourhood is bounded by `k_max` rather
than by the fleet.

**Scaling.** The two axes behave differently, which was the most informative
finding. In **fleet size** the approaches diverge: full replanning's impact and
cost grow with the number of agents, while local repair's impact grows
sub-linearly and its decision time stays bounded. In **obstacle density**, impact
per disruption is flat for every strategy — density changes how often repair is
needed, not how far it reaches.

**What going outside the syllabus was worth.** The four syllabus strategies sit
within each other's error bars on every headline metric (5.38 to 5.50 agents
changed). Whether agents hill-climb or solve the DCOP optimally is nearly
irrelevant, because they are all choosing among the same five precomputed plans.
Replacing that with a search over the real plan space — CBS — cut churn by 29%,
cut abandonment six-fold, and slightly improved completion. The lesson is that
the binding constraint was the *formulation*, not the solver: ADOPT was solving
the given problem optimally, and the given problem was the wrong one.

That is not a criticism of the taught material. Ch 12's insistence that an agent
controls only its own variable is what makes the DCOP genuinely distributed, and
CBS gives that up — it is a centralised search over the neighbourhood. The
comparison is really between a distributed algorithm with a restricted view and a
centralised one with a complete view, and on these instances the complete view
wins by more than optimality does.

**Limitations.** CBS costs 18× ADOPT's compute and is not distributed. Local
repair of any kind still completes less work than global replanning (69–73%
against 82%) and strands more agents. `k_max` is fixed ahead of time rather than
adapted to congestion, with only a single escalation step. The disruption model
is permanent — nothing is ever cleared — which is the pessimistic case. And both
complete solvers have exponential worst cases that are visible in the data:
ADOPT's in the `k_max` sweep, CBS's in the search budget it has to be given.
