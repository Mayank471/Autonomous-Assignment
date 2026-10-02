"""A repair episode: what a strategy is given, what it returns, and escalation.

A strategy receives a :class:`RepairContext` -- the world at the moment of the
disruption, every committed plan, and the *forced* agents whose plans the
disruption has made infeasible -- and returns new plans for some agents.

If a strategy cannot find a repair it escalates, and never invents a plan:

1. neighbourhood: replan everyone within radius r of the disruption with
   prioritized planning (r = 4, 8, 16), everyone else fixed;
2. global: replan every active agent from the current state;
3. if an emergency deadline made all of that impossible, repeat without it
   (recorded as ``deadline_relaxed``);
4. gridlock: every active agent pauses in place for d steps (d = 2, 4, ...,
   64) and the fleet is replanned from there -- standing still is always
   physically safe, and obstacle windows expire while it waits;
5. otherwise the repair *fails* and the episode records the failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..grid import WarehouseGrid
from ..negotiation import Session
from ..obstacles import ObstacleWindows
from ..plan import Plan
from ..planner import prioritized
from ..reservation import ReservationBoard
from ..stastar import Query

LEVEL_FAST, LEVEL_KRCBS, LEVEL_LOCAL, LEVEL_CASCADE = "fast", "krcbs", "local", "cascade"
LEVEL_GREEDY = "krcbs_greedy"
LEVEL_PAUSE = "global_pause"
LEVEL_NEIGHBOURHOOD, LEVEL_GLOBAL, LEVEL_FAILED = "neighbourhood", "global", "failed"


@dataclass
class RepairConfig:
    max_hl_nodes: int = 1500              # KR-CBS high-level node budget (per stage)
    max_total_expansions: int = 3_000_000  # all low-level searches of one repair
    max_expansions_per_call: int = 400_000
    radii: tuple[int, ...] = (4, 8, 16)
    pause_steps: tuple[int, ...] = (2, 4, 8, 16, 32, 64)


@dataclass
class RepairContext:
    grid: WarehouseGrid
    t: int
    plans: dict[int, Plan]                 # committed plans of *all* agents
    obstacles: ObstacleWindows
    forced: tuple[int, ...]                # ordered: plan breaks soonest first
    active: frozenset[int]                 # agents that have not finished
    query_of: Callable[[int], Query]
    disruption_cell: int | None = None
    config: RepairConfig = field(default_factory=RepairConfig)
    board: ReservationBoard = field(init=False)

    def __post_init__(self) -> None:
        self.board = ReservationBoard.from_plans(self.grid.n_cells, self.plans, self.t)

    @property
    def leader(self) -> int:
        return self.forced[0]

    def session(self) -> Session:
        return Session(self.grid, self.obstacles, self.board, self.query_of, self.leader,
                       max_expansions_per_call=self.config.max_expansions_per_call)

    @property
    def base_soc(self) -> int:
        return sum(p.end for p in self.plans.values())


@dataclass
class RepairResult:
    plans: dict[int, Plan] | None          # new plans (any subset of agents)
    level: str
    session: Session
    hl_nodes: int = 0
    exact: bool = True
    deadline_relaxed: bool = False

    @property
    def ok(self) -> bool:
        return self.plans is not None


def run_prioritized(ctx: RepairContext, session: Session, agents: list[int],
                    drop_deadline: bool = False, pause_until: int = -1,
                    stationary_stage: bool = True) -> dict[int, Plan] | None:
    """Prioritized planning of ``agents`` with every other plan fixed.
    ``pause_until``: every planned agent first stands still until that time."""
    queries = []
    for a in agents:
        q = ctx.query_of(a)
        dl = None if drop_deadline else q.deadline
        q = Query(q.agent, q.start, q.t0, q.goals, q.dock, max(q.hold_until, pause_until), dl)
        queries.append(q)
    out = prioritized(ctx.grid, queries, obstacles=ctx.obstacles, board=ctx.board,
                      max_expansions=ctx.config.max_expansions_per_call,
                      stationary_stage=stationary_stage)
    session.ll_calls += out.calls
    session.ll_expansions += out.expansions
    for a in agents:                      # one request/answer per agent asked
        if a != session.leader:
            session._send("REPLAN_REQUEST", a)
            session.messages["PROPOSAL" if out.ok else "CANNOT_COMPLY"] += 1
    return out.plans


def neighbourhood(ctx: RepairContext, radius: int) -> list[int]:
    g = ctx.grid
    centres = [ctx.plans[a].at(ctx.t) for a in ctx.forced]
    if ctx.disruption_cell is not None:
        centres.append(ctx.disruption_cell)
    out = set(ctx.forced)
    for a in ctx.active:
        pos = ctx.plans[a].at(ctx.t)
        if any(g.manhattan(pos, c) <= radius for c in centres):
            out.add(a)
    return sorted(out)


def escalate(ctx: RepairContext, session: Session, *, hl_nodes: int = 0,
             skip_neighbourhood: bool = False) -> RepairResult:
    has_deadline = any(ctx.query_of(a).deadline is not None for a in ctx.active)
    for drop_deadline in (False, True):
        if not skip_neighbourhood:
            for r in ctx.config.radii:
                agents = neighbourhood(ctx, r)
                plans = run_prioritized(ctx, session, agents, drop_deadline,
                                        stationary_stage=False)
                if plans is not None:
                    return RepairResult(plans, LEVEL_NEIGHBOURHOOD, session, hl_nodes,
                                        exact=False, deadline_relaxed=drop_deadline)
        plans = run_prioritized(ctx, session, sorted(ctx.active), drop_deadline)
        if plans is not None:
            return RepairResult(plans, LEVEL_GLOBAL, session, hl_nodes, exact=False,
                                deadline_relaxed=drop_deadline)
        if not has_deadline:
            break
    # Gridlock: every active agent pauses in place (always physically safe:
    # nobody moves, and blockages never start on an occupied cell), then the
    # fleet is replanned from where it stands once more windows have expired.
    for d in ctx.config.pause_steps:
        plans = run_prioritized(ctx, session, sorted(ctx.active), True, ctx.t + d,
                                stationary_stage=False)
        if plans is not None:
            return RepairResult(plans, LEVEL_PAUSE, session, hl_nodes, exact=False,
                                deadline_relaxed=has_deadline)
    return RepairResult(None, LEVEL_FAILED, session, hl_nodes, exact=False)
