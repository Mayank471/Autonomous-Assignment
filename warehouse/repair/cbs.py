"""Keep/Release Conflict-Based Search (KR-CBS): minimum-change plan repair.

Objective for one repair, in strict priority order:

1. the number of agents whose plans change;
2. total time (sum over all agents of completion times, SOC).

Search node: R (agents released to replan; starts as the forced agents),
K (agents being kept: their committed plans are hard for everyone in R), CBS
constraints, and one path per agent in R.  Every other agent is *undecided*:
its committed plan is a soft obstacle for the low-level search.

Expanding a node picks its earliest conflict:

* between two agents of R -> the ordinary CBS split (constrain one or the
  other);
* between i in R and an undecided agent j -> **KEEP j** (j joins K; agents of
  R that collided with j replan around it) or **RELEASE j** (j joins R and is
  asked, by message, to propose a new route).

Why the first solution popped is optimal: any repair either leaves j's plan
untouched (it lies under KEEP j) or changes it (it lies under RELEASE j), so the
two children cover every repair between them; the key is ``(|R|, SOC, soft)``
and, within one value of |R|, SOC can only grow down a branch (constraints and
kept agents only remove options).  So nodes are popped level by level, and the
first conflict-free node has the fewest changed agents and, among those, the
least total time.  A released agent that ends up with its old plan would mean
the same repair exists one level lower, where it would already have been found.

:func:`coalition_cbs` is plain CBS over a fixed coalition with everyone else
hard.  It is the *fast path* (coalition = forced agents: "only the disrupted
agents change") and the inner step of the brute-force oracle used to verify
KR-CBS.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

from ..negotiation import Session
from ..plan import Plan
from ..reservation import ReservationBoard
from ..stastar import NO_CONSTRAINTS, Constraints
from .episode import RepairContext

OK, INFEASIBLE, BUDGET = "ok", "infeasible", "budget"


@dataclass
class _Node:
    released: frozenset[int]
    kept: frozenset[int]
    cons: dict[int, Constraints]
    paths: dict[int, Plan]
    soft: dict[int, int]
    soc: int
    nid: int

    hit: int = 0          # distinct undecided agents the current paths collide with

    def key(self, greedy: bool = False):
        if greedy:
            return (len(self.released) + self.hit, self.soc, sum(self.soft.values()), self.nid)
        return (len(self.released), self.soc, sum(self.soft.values()), self.nid)


@dataclass
class CBSOutcome:
    plans: dict[int, Plan] | None
    status: str
    hl_nodes: int
    exact: bool

    @property
    def ok(self) -> bool:
        return self.status == OK


class _Search:
    """``greedy=True`` orders nodes by |R| + (distinct undecided agents still
    hit) instead of |R|: much faster when many agents must make way, but no
    longer guaranteed minimal.  Used only after the exact search runs out of
    budget, and its results are flagged as not exact."""

    def __init__(self, ctx: RepairContext, session: Session, allow_release: bool,
                 greedy: bool = False) -> None:
        self.ctx = ctx
        self.s = session
        self.allow_release = allow_release
        self.greedy = greedy
        self.nc = ctx.grid.n_cells
        self.exact = not greedy
        self.ids = itertools.count()

    # ---------------------------------------------------------------- helpers
    def plan(self, a: int, released: frozenset[int], kept: frozenset[int],
             cons: Constraints, paths: dict[int, Plan]):
        hard = kept if self.allow_release else None
        cat = None
        others = [b for b in paths if b != a]
        if others:
            cat = ReservationBoard(self.nc)
            for b in sorted(others):
                cat.add(b, paths[b], self.ctx.t)
        res = self.s.replan(a, ignore=released, hard=hard, constraints=cons, cat=cat)
        if res.status == BUDGET:
            self.exact = False
        return res

    def soc(self, paths: dict[int, Plan]) -> int:
        plans = self.ctx.plans
        return self.ctx.base_soc + sum(p.end - plans[a].end for a, p in paths.items())

    def hit_count(self, released: frozenset[int], paths: dict[int, Plan]) -> int:
        board = self.ctx.board
        hit = set()
        for i, p in paths.items():
            prev = None
            for k, c in enumerate(p.cells):
                t = p.t0 + k
                o = board.occupant(t, c)
                if o is not None and o not in released:
                    hit.add(o)
                if prev is not None and prev != c:
                    o = board.mover(t - 1, c, prev)
                    if o is not None and o not in released:
                        hit.add(o)
                prev = c
        return len(hit)

    def make(self, released, kept, cons, paths, soft) -> _Node:
        node = _Node(released, kept, cons, paths, soft, self.soc(paths), next(self.ids))
        if self.greedy:
            node.hit = self.hit_count(released, paths)
        return node

    # ----------------------------------------------------------- conflicts
    def internal_conflict(self, node: _Node):
        """Earliest vertex/swap conflict between two released agents."""
        nc = self.nc
        occ: dict[int, int] = {}
        moves: dict[tuple[int, int, int], int] = {}
        best = None
        for i in sorted(node.paths):
            p = node.paths[i]
            prev = None
            for k, c in enumerate(p.cells):
                t = p.t0 + k
                if best is not None and t >= best[0]:
                    break
                j = occ.get(t * nc + c)
                if j is not None:
                    best = (t, "v", j, i, c)
                    break
                occ[t * nc + c] = i
                if prev is not None and prev != c:
                    j = moves.get((t - 1, c, prev))
                    if j is not None:
                        best = (t - 1, "e", j, i, (c, prev))   # j moved c->prev, i prev->c
                        break
                    moves[(t - 1, prev, c)] = i
                prev = c
        return best

    def soft_conflict(self, node: _Node):
        """Earliest collision of a released agent with an undecided agent's plan."""
        board = self.ctx.board
        best = None
        for i in sorted(node.paths):
            p = node.paths[i]
            prev = None
            for k, c in enumerate(p.cells):
                t = p.t0 + k
                if best is not None and t >= best[0]:
                    break
                o = board.occupant(t, c)
                if o is not None and o != i and o not in node.released:
                    best = (t, i, o)
                    break
                if prev is not None and prev != c:
                    o = board.mover(t - 1, c, prev)
                    if o is not None and o != i and o not in node.released:
                        best = (t - 1, i, o)
                        break
                prev = c
        return best

    def collides_with(self, path: Plan, j: int) -> bool:
        board = self.ctx.board
        prev = None
        for k, c in enumerate(path.cells):
            t = path.t0 + k
            if board.occupant(t, c) == j:
                return True
            if prev is not None and prev != c and board.mover(t - 1, c, prev) == j:
                return True
            prev = c
        return False

    # ------------------------------------------------------------- children
    def constrained_child(self, node: _Node, a: int, cons: Constraints) -> _Node | None:
        res = self.plan(a, node.released, node.kept, cons, node.paths)
        if not res.ok:
            return None
        c2 = dict(node.cons)
        c2[a] = cons
        p2 = dict(node.paths)
        p2[a] = res.plan
        s2 = dict(node.soft)
        s2[a] = res.soft
        return self.make(node.released, node.kept, c2, p2, s2)

    def keep_child(self, node: _Node, j: int) -> _Node | None:
        kept = node.kept | {j}
        p2, s2 = dict(node.paths), dict(node.soft)
        for a in sorted(node.paths):
            if self.collides_with(node.paths[a], j):
                res = self.plan(a, node.released, kept, node.cons.get(a, NO_CONSTRAINTS), p2)
                if not res.ok:
                    return None
                p2[a], s2[a] = res.plan, res.soft
        return self.make(node.released, kept, dict(node.cons), p2, s2)

    def release_child(self, node: _Node, j: int) -> _Node | None:
        released = node.released | {j}
        res = self.plan(j, released, node.kept, NO_CONSTRAINTS, node.paths)
        if not res.ok:
            return None
        p2, s2 = dict(node.paths), dict(node.soft)
        p2[j], s2[j] = res.plan, res.soft
        return self.make(released, node.kept, dict(node.cons), p2, s2)

    # ------------------------------------------------------------------ main
    def run(self, coalition: tuple[int, ...], max_nodes: int) -> CBSOutcome:
        released = frozenset(coalition)
        paths, soft = {}, {}
        for a in sorted(released):
            res = self.plan(a, released, frozenset(), NO_CONSTRAINTS, paths)
            if not res.ok:
                return CBSOutcome(None, BUDGET if res.status == BUDGET else INFEASIBLE,
                                  0, self.exact and res.status != BUDGET)
            paths[a], soft[a] = res.plan, res.soft
        root = self.make(released, frozenset(), {}, paths, soft)
        heap = [(root.key(self.greedy), root)]
        expanded = 0
        max_total = self.ctx.config.max_total_expansions
        exp0 = self.s.ll_expansions
        while heap:
            if expanded >= max_nodes or self.s.ll_expansions - exp0 > max_total:
                return CBSOutcome(None, BUDGET, expanded, False)
            _, node = heapq.heappop(heap)
            expanded += 1
            conflict = self.internal_conflict(node)
            children: list[_Node | None] = []
            if conflict is not None:
                t, kind, j, i, where = conflict
                nc = self.nc
                if kind == "v":
                    key = t * nc + where
                    for a in (i, j):
                        children.append(self.constrained_child(
                            node, a, node.cons.get(a, NO_CONSTRAINTS).with_vertex(key, t + 0)))
                else:
                    c, prev = where             # j moved c->prev, i moved prev->c at t
                    children.append(self.constrained_child(
                        node, i, node.cons.get(i, NO_CONSTRAINTS).with_edge((t * nc + prev) * nc + c, t)))
                    children.append(self.constrained_child(
                        node, j, node.cons.get(j, NO_CONSTRAINTS).with_edge((t * nc + c) * nc + prev, t)))
            elif self.allow_release and (sc := self.soft_conflict(node)) is not None:
                _, i, j = sc
                children.append(self.keep_child(node, j))
                children.append(self.release_child(node, j))
            else:
                return CBSOutcome(node.paths, OK, expanded, self.exact)
            for ch in children:
                if ch is not None:
                    heapq.heappush(heap, (ch.key(self.greedy), ch))
        return CBSOutcome(None, INFEASIBLE, expanded, self.exact)


def coalition_cbs(ctx: RepairContext, session: Session, coalition: tuple[int, ...],
                  max_nodes: int) -> CBSOutcome:
    """Optimal (min-SOC) plans for ``coalition`` with every other plan fixed."""
    return _Search(ctx, session, allow_release=False).run(tuple(coalition), max_nodes)


def kr_cbs(ctx: RepairContext, session: Session, max_nodes: int,
           greedy: bool = False) -> CBSOutcome:
    """Minimum-change repair starting from the forced agents."""
    return _Search(ctx, session, allow_release=True, greedy=greedy).run(ctx.forced, max_nodes)


def brute_force_min_change(ctx: RepairContext, max_nodes: int = 50_000):
    """Exactness oracle: try every coalition (forced agents + extras), smallest
    first, each solved optimally by :func:`coalition_cbs`.

    Returns ``(min number of agents changed, min SOC at that size)`` or ``None``
    if no coalition works.  Raises if a sub-search is cut by its budget, since
    the answer would then not be a proof.
    """
    forced = set(ctx.forced)
    others = sorted(ctx.active - forced)
    for extra in range(len(others) + 1):
        best = None
        for combo in itertools.combinations(others, extra):
            coalition = tuple(sorted(forced | set(combo)))
            out = coalition_cbs(ctx, ctx.session(), coalition, max_nodes)
            if out.status == BUDGET or not out.exact:
                raise RuntimeError(f"brute force: budget hit for coalition {coalition}")
            if out.ok:
                soc = ctx.base_soc + sum(p.end - ctx.plans[a].end for a, p in out.plans.items())
                best = soc if best is None else min(best, soc)
        if best is not None:
            return len(forced) + extra, best
    return None
