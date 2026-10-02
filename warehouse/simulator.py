"""Discrete-time execution of a warehouse run, with disruptions and repair.

Each time step:

1. apply the disruptions due now; each agent checks its own plan against the
   new world, and the agents whose plans broke (*forced* agents) start a repair
   session using the chosen strategy;
2. commit the repaired plans, then check them with the independent checker
   (and check that every new plan still achieves its agent's goals);
3. move every agent one step along its committed plan.

A repair that fails is recorded as a failure and ends the episode: no plan is
ever invented to paper over it.
"""

from __future__ import annotations

import heapq
import time
from collections import Counter
from dataclasses import dataclass, field

from .disruptions import BLOCKAGE, BREAKDOWN, EMERGENCY, Event
from .emergency import earliest_delivery, insert_emergency
from .obstacles import ObstacleWindows
from .plan import Plan, plan_distance
from .planner import prioritized
from .repair import RepairConfig, RepairContext, repair
from .scenario import Instance
from .stastar import Query, advance
from .tasks import DELIVERY, E_DELIVERY, Mission
from .validate import PlanViolation, assert_plans


@dataclass
class SimConfig:
    strategy: str = "krcbs"
    validate: bool = True
    t_cap: int = 3000
    emergency_slack: int | None = 0         # None = no deadline at all
    max_deferral: int = 20
    repair: RepairConfig = field(default_factory=RepairConfig)


@dataclass
class AgentState:
    mission: Mission
    progress: int = 0
    done_time: int | None = None
    broken_until: int = -1
    deadline: tuple[int, int] | None = None     # (absolute goal index, latest time)
    emergency_earliest: int | None = None


def initial_plans(inst: Instance) -> dict[int, Plan]:
    queries = [Query(m.agent, m.dock, 0, tuple(m.goals), m.dock) for m in inst.missions]
    out = prioritized(inst.grid, queries, obstacles=ObstacleWindows())
    if not out.ok:
        raise RuntimeError(f"initial planning failed (seed {inst.seed})")
    return out.plans


class World:
    def __init__(self, inst: Instance, plans: dict[int, Plan]) -> None:
        self.inst = inst
        self.grid = inst.grid
        self.t = 0
        self.plans = dict(plans)
        self.obstacles = ObstacleWindows()
        self.agents = [AgentState(m.copy()) for m in inst.missions]
        self.failed = False
        self.emergency_lateness: list[int] = []
        for a, st in enumerate(self.agents):
            st.progress = advance(st.mission.goals, 0, self.plans[a].at(0))

    # ---------------------------------------------------------------- queries
    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def position(self, a: int) -> int:
        return self.plans[a].at(self.t)

    def active_agents(self) -> list[int]:
        return [a for a, st in enumerate(self.agents) if st.done_time is None]

    def is_broken(self, a: int) -> bool:
        return self.agents[a].broken_until > self.t

    def all_done(self) -> bool:
        return all(st.done_time is not None for st in self.agents)

    def docks(self) -> dict[int, int]:
        return {a: st.mission.dock for a, st in enumerate(self.agents)}

    def query(self, a: int) -> Query:
        st = self.agents[a]
        hold = st.broken_until if st.broken_until > self.t else -1
        dl = None
        if st.deadline is not None and st.deadline[0] >= st.progress:
            dl = (st.deadline[0] - st.progress, st.deadline[1])
        return Query(a, self.position(a), self.t, tuple(st.mission.goals[st.progress:]),
                     st.mission.dock, hold, dl)

    def copy(self) -> "World":
        w = World.__new__(World)
        w.inst, w.grid, w.t = self.inst, self.grid, self.t
        w.plans = dict(self.plans)
        w.obstacles = self.obstacles.copy()
        w.agents = [AgentState(st.mission.copy(), st.progress, st.done_time, st.broken_until,
                               st.deadline, st.emergency_earliest) for st in self.agents]
        w.failed = self.failed
        w.emergency_lateness = list(self.emergency_lateness)
        return w

    # -------------------------------------------------------------- dynamics
    def step(self) -> None:
        self.t += 1
        t = self.t
        for a, st in enumerate(self.agents):
            if st.done_time is not None:
                continue
            st.progress = advance(st.mission.goals, st.progress, self.plans[a].at(t))
            if st.deadline is not None and st.progress > st.deadline[0]:
                if st.emergency_earliest is not None:
                    self.emergency_lateness.append(t - st.emergency_earliest)
                st.deadline = None
                st.emergency_earliest = None
            if st.progress == len(st.mission.goals):
                st.done_time = t
        self.obstacles.prune(t)

    def soc(self) -> int:
        return sum(st.done_time for st in self.agents)

    def projected_soc(self) -> int:
        """Total completion time if nothing else goes wrong."""
        return sum(st.done_time if st.done_time is not None else self.plans[a].end
                   for a, st in enumerate(self.agents))


# ------------------------------------------------------------------- events
def _forced_by_cell(world: World, cell: int, t1: int, t2: int, skip: int = -1) -> dict[int, int]:
    """Agents whose plan occupies ``cell`` at some time in [t1, t2] -> first time."""
    out = {}
    for a in world.active_agents():
        if a == skip:
            continue
        p = world.plans[a]
        for tau in range(t1, min(t2, p.end) + 1):
            if p.at(tau) == cell:
                out[a] = tau
                break
    return out


def apply_event(world: World, ev: Event, cfg: SimConfig) -> tuple[str, list[int], int | None]:
    """Apply ``ev`` at ``world.t``.

    Returns ``(status, forced agents ordered by when their plan breaks,
    disruption cell)`` where status is ``applied``, ``deferred`` or ``skipped``.
    """
    t = world.t
    g = world.grid
    if ev.kind == BLOCKAGE:
        if any(world.position(a) == ev.cell for a in range(world.n_agents)):
            return "deferred", [], ev.cell
        world.obstacles.add(ev.cell, t, t + ev.duration)
        hit = _forced_by_cell(world, ev.cell, t + 1, t + ev.duration - 1)
        return "applied", sorted(hit, key=lambda a: (hit[a], a)), ev.cell

    if ev.kind == BREAKDOWN:
        b = ev.agent
        st = world.agents[b]
        if st.done_time is not None or world.is_broken(b):
            return "skipped", [], None
        c = world.position(b)
        until = t + ev.duration
        world.obstacles.add(c, t, until + 1, owner=b)
        st.broken_until = until
        hit = _forced_by_cell(world, c, t + 1, until, skip=b)
        p = world.plans[b]
        moves = [tau for tau in range(t, until + 1) if p.at(tau) != c]
        if moves:
            hit[b] = moves[0]
        return "applied", sorted(hit, key=lambda a: (hit[a], a)), c

    if ev.kind == EMERGENCY:
        n = world.n_agents
        e = None
        for k in range(n):
            a = (ev.agent + k) % n
            st = world.agents[a]
            if st.done_time is None and not world.is_broken(a) and st.deadline is None:
                e = a
                break
        if e is None:
            return "skipped", [], None
        st = world.agents[e]
        pos = world.position(e)
        earliest = earliest_delivery(g, e, pos, t, ev.pickup, ev.station, st.mission.dock,
                                     world.obstacles)
        if earliest is None:
            return "skipped", [], None
        d_idx = insert_emergency(st.mission, st.progress, ev.pickup, ev.station)
        st.progress = advance(st.mission.goals, st.progress, pos)
        if cfg.emergency_slack is not None:
            st.deadline = (d_idx, earliest + cfg.emergency_slack)
        else:
            st.deadline = (d_idx, 1 << 30)
        st.emergency_earliest = earliest
        return "applied", [e], ev.pickup

    raise ValueError(ev.kind)


def _check_goals(world: World, a: int, plan: Plan) -> None:
    st = world.agents[a]
    goals = st.mission.goals
    p = plan.suffix(world.t)
    if p.t0 != world.t or p.cells[0] != world.position(a):
        raise PlanViolation(f"agent {a}: plan does not start where the agent is")
    k = st.progress
    for c in p.cells:
        k = advance(goals, k, c)
    if k != len(goals) or p.cells[-1] != st.mission.dock:
        raise PlanViolation(f"agent {a}: plan does not complete its goals")


def make_context(world: World, forced: list[int], cell: int | None,
                 cfg: SimConfig) -> RepairContext:
    return RepairContext(world.grid, world.t, dict(world.plans), world.obstacles,
                         tuple(forced), frozenset(world.active_agents()), world.query,
                         cell, cfg.repair)


def handle_event(world: World, ev: Event, cfg: SimConfig) -> tuple[str, dict]:
    """Apply one event and repair.  Returns ``(status, record)``."""
    status, forced, cell = apply_event(world, ev, cfg)
    rec = {"eid": ev.eid, "kind": ev.kind, "t": world.t, "status": status,
           "forced": len(forced), "n_active": len(world.active_agents())}
    if status != "applied" or not forced:
        rec.update(changed=0, collateral=0)
        return status, rec
    ctx = make_context(world, forced, cell, cfg)
    old = ctx.plans
    t0 = time.perf_counter()
    res = repair(cfg.strategy, ctx)
    runtime = time.perf_counter() - t0
    rec.update(level=res.level, hl_nodes=res.hl_nodes, ll_calls=res.session.ll_calls,
               ll_expansions=res.session.ll_expansions, runtime_s=runtime,
               exact=res.exact, deadline_relaxed=res.deadline_relaxed)
    if not res.ok:
        world.failed = True
        rec.update(changed=0, collateral=0, failed=True)
        return status, rec
    missing = [a for a in forced if a not in res.plans]
    if missing:
        raise PlanViolation(f"strategy {cfg.strategy} left forced agents {missing} unrepaired")
    changed = res.session.close(old, res.plans, world.t)
    new = dict(old)
    for a in changed:
        new[a] = res.plans[a]
    if cfg.validate:
        for a in sorted(changed | set(forced)):
            _check_goals(world, a, new[a])
        assert_plans(world.grid, {a: new[a] for a in range(world.n_agents)}, world.t,
                     docks=world.docks(), obstacles=world.obstacles,
                     context=f"{cfg.strategy} after event {ev.eid} ({ev.kind}) at t={world.t}")
    world.plans = new
    rec.update(
        changed=len(changed),
        collateral=len(changed - set(forced)),
        contacted=len(res.session.contacted),
        messages=res.session.n_messages,
        plan_distance=sum(plan_distance(old[a], new[a], world.t) for a in changed),
        soc_delta=sum(new[a].end - old[a].end for a in changed),
        failed=False,
    )
    return status, rec


# ------------------------------------------------------------------ episodes
@dataclass
class EpisodeResult:
    strategy: str
    n_agents: int
    seed: int
    completed: bool
    soc: int | None
    makespan: int | None
    soc0: int
    agents_done: float
    deliveries_done: float
    records: list[dict]
    levels: Counter
    realized_density: float
    emergency_lateness: list[int]
    runtime_s: float

    @property
    def repairs(self) -> list[dict]:
        return [r for r in self.records if r["status"] == "applied" and r["forced"] > 0]


def _delivery_fraction(world: World) -> float:
    total = done = 0
    for st in world.agents:
        for k, kind in enumerate(st.mission.kinds):
            if kind in (DELIVERY, E_DELIVERY):
                total += 1
                done += k < st.progress
    return done / total if total else 1.0


def run_episode(inst: Instance, plans0: dict[int, Plan], events: list[Event],
                cfg: SimConfig) -> EpisodeResult:
    t_start = time.perf_counter()
    world = World(inst, plans0)
    soc0 = sum(p.end for p in plans0.values())
    heap = [(e.t, e.eid, 0, e) for e in events]
    heapq.heapify(heap)
    records: list[dict] = []
    blocked_area = 0
    steps = 0
    while not world.all_done() and not world.failed and world.t <= cfg.t_cap:
        while heap and heap[0][0] <= world.t:
            _, eid, deferrals, ev = heapq.heappop(heap)
            status, rec = handle_event(world, ev, cfg)
            if status == "deferred":
                if deferrals < cfg.max_deferral:
                    heapq.heappush(heap, (world.t + 1, eid, deferrals + 1, ev))
                continue
            records.append(rec)
            if world.failed:
                break
        if world.failed:
            break
        blocked_area += world.obstacles.active_count(world.t)
        steps += 1
        world.step()
    completed = world.all_done()
    levels = Counter(r.get("level") for r in records if r.get("level"))
    n_blockable = len(inst.grid.blockable)
    return EpisodeResult(
        strategy=cfg.strategy, n_agents=inst.n_agents, seed=inst.seed, completed=completed,
        soc=world.soc() if completed else None,
        makespan=max(st.done_time for st in world.agents) if completed else None,
        soc0=soc0,
        agents_done=sum(st.done_time is not None for st in world.agents) / world.n_agents,
        deliveries_done=_delivery_fraction(world),
        records=records, levels=levels,
        realized_density=blocked_area / max(1, steps) / n_blockable,
        emergency_lateness=world.emergency_lateness,
        runtime_s=time.perf_counter() - t_start,
    )
