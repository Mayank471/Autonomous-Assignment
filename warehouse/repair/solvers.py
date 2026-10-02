"""The four repair strategies, compared head to head on identical events.

``global``   everyone replans with prioritized planning from the current state.
             This is what the assignment forbids; it is the reference point.
``local``    no negotiation: only the forced agents replan, one after another,
             with every other plan fixed.  If one fails, fall back to global.
``cascade``  naive negotiation: forced agents take priority, and any agent
             whose plan collides with a new route replans around it, and so on.
``krcbs``    the proposed method: the smallest set of agents, found exactly by
             Keep/Release CBS (fast path first: "only the forced agents change").
"""

from __future__ import annotations

from ..plan import Plan
from .cbs import coalition_cbs, kr_cbs
from .episode import (LEVEL_CASCADE, LEVEL_FAST, LEVEL_GLOBAL, LEVEL_GREEDY, LEVEL_KRCBS,
                      LEVEL_LOCAL, RepairContext, RepairResult, escalate, run_prioritized)


def solve_global(ctx: RepairContext) -> RepairResult:
    s = ctx.session()
    plans = run_prioritized(ctx, s, sorted(ctx.active))
    if plans is None:
        return escalate(ctx, s, skip_neighbourhood=True)
    return RepairResult(plans, LEVEL_GLOBAL, s, exact=False)


def solve_local(ctx: RepairContext) -> RepairResult:
    s = ctx.session()
    board = ctx.board.copy()
    for a in ctx.forced:
        board.remove(a)
    plans: dict[int, Plan] = {}
    for a in ctx.forced:
        res = s.replan(a, board=board, hard=None)
        if not res.ok:
            return escalate(ctx, s, skip_neighbourhood=True)
        plans[a] = res.plan
        board.add(a, res.plan, ctx.t)
    return RepairResult(plans, LEVEL_LOCAL, s, exact=False)


def _colliding_agents(board, me: int, path: Plan) -> list[int]:
    out = set()
    prev = None
    for k, c in enumerate(path.cells):
        t = path.t0 + k
        o = board.occupant(t, c)
        if o is not None and o != me:
            out.add(o)
        if prev is not None and prev != c:
            o = board.mover(t - 1, c, prev)
            if o is not None and o != me:
                out.add(o)
        prev = c
    return sorted(out)


def solve_cascade(ctx: RepairContext) -> RepairResult:
    s = ctx.session()
    board = ctx.board.copy()
    queue = list(ctx.forced)
    queued = set(queue)
    done: list[int] = []
    plans: dict[int, Plan] = {}
    while queue:
        a = queue.pop(0)
        pending = frozenset(queued - set(done) - {a})
        res = s.replan(a, board=board, ignore=pending, hard=frozenset(done))
        if not res.ok:
            return escalate(ctx, s, skip_neighbourhood=True)
        plans[a] = res.plan
        # find who is displaced *before* posting: posting overwrites their entries
        for o in _colliding_agents(board, a, res.plan):
            if o not in queued:
                queue.append(o)
                queued.add(o)
        board.add(a, res.plan, ctx.t)
        done.append(a)
    return RepairResult(plans, LEVEL_CASCADE, s, exact=False)


def _krcbs_stages(ctx: RepairContext, s) -> tuple[RepairResult | None, int, bool]:
    """Fast path, then exact KR-CBS, then greedy KR-CBS.

    Returns ``(result or None, high-level nodes used, proven infeasible)``.
    """
    cfg = ctx.config
    fast = coalition_cbs(ctx, s, ctx.forced, cfg.max_hl_nodes)
    hl = fast.hl_nodes
    if fast.ok:
        return RepairResult(fast.plans, LEVEL_FAST, s, hl, exact=fast.exact), hl, False
    kr = kr_cbs(ctx, s, cfg.max_hl_nodes)
    hl += kr.hl_nodes
    if kr.ok:
        return RepairResult(kr.plans, LEVEL_KRCBS, s, hl, exact=kr.exact), hl, False
    if kr.status == "infeasible" and kr.exact:
        return None, hl, True          # not even with every other agent making way
    greedy = kr_cbs(ctx, s, cfg.max_hl_nodes, greedy=True)
    hl += greedy.hl_nodes
    if greedy.ok:
        return RepairResult(greedy.plans, LEVEL_GREEDY, s, hl, exact=False), hl, False
    return None, hl, False


def solve_krcbs(ctx: RepairContext) -> RepairResult:
    s = ctx.session()
    res, hl, infeasible = _krcbs_stages(ctx, s)
    if res is not None:
        return res
    has_deadline = any(ctx.query_of(a).deadline is not None for a in ctx.active)
    if infeasible and has_deadline:
        # An emergency deadline has become impossible (e.g. the agent carrying it
        # broke down): repair again with deadlines relaxed, and say so.
        s.relax_deadlines = True
        res, hl2, _ = _krcbs_stages(ctx, s)
        if res is not None:
            res.deadline_relaxed = True
            res.hl_nodes = hl + hl2
            return res
        hl += hl2
    return escalate(ctx, s, hl_nodes=hl)


SOLVERS = {
    "global": solve_global,
    "local": solve_local,
    "cascade": solve_cascade,
    "krcbs": solve_krcbs,
}
STRATEGIES = tuple(SOLVERS)


def repair(strategy: str, ctx: RepairContext) -> RepairResult:
    return SOLVERS[strategy](ctx)
