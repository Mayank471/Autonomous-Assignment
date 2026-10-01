"""ADOPT -- Asynchronous Distributed OPTimization (Weiss Ch 12 section 4.1).

A faithful implementation of Modi et al.'s algorithm as presented in the
textbook: a distributed best-first backtrack search over a DFS pseudo-tree in
which every agent asynchronously assigns its variable the value minimising a
*local lower bound*, and those bounds are refined by messages passed between
neighbours.  Its three key components, in the textbook's words, are
"(i) local lower-bound estimates, (ii) backtrack thresholds, and (iii)
termination conditions".

Four message types are exchanged:

``VALUE``
    sent to every *lower* neighbour, reporting this agent's current value.
``COST``
    sent to the parent, carrying the current context and the minimum lower and
    upper bounds computed under it.
``THRESHOLD``
    sent to children, restoring a previously established lower bound when a
    context is revisited, so the subtree does not re-derive it from scratch.
``TERMINATE``
    propagated from the root once ``LB == UB``.

Agents are stepped in *synchronization cycles* -- every agent consumes the
messages delivered to it and emits the next round simultaneously.  That is the
textbook's own progress measure, "less sensitive to variations in agents'
computation speed and communication delays than the wall clock", and it is what
this module reports as :attr:`AdoptResult.cycles`.

Two deliberate deviations from the printed pseudocode, both behaviour
preserving, are marked in the code: the allocation invariant is restored by a
single proportional pass rather than unit increments (costs here are floats, so
``t := t + 1`` would not terminate), and ``initialize`` establishes the
threshold invariant so that a variable with no children terminates immediately
rather than idling with ``threshold = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .dfstree import DFSTree, build_dfs_forest
from .model import ConstraintNetwork, Value, Variable

INF = float("inf")

#: Tolerance for comparing accumulated float costs.
EPS = 1e-9

VALUE, COST, THRESHOLD, TERMINATE = "VALUE", "COST", "THRESHOLD", "TERMINATE"


@dataclass(frozen=True)
class AdoptResult:
    """Outcome of one ADOPT run, with the coordination overhead it incurred."""

    assignment: dict[Variable, Value]
    cost: float
    lower_bound: float
    upper_bound: float
    cycles: int
    messages: int
    messages_by_type: dict[str, int]
    terminated: bool
    n_trees: int

    @property
    def gap(self) -> float:
        """Remaining bound interval; ``0`` when the optimum was proved."""
        return self.upper_bound - self.lower_bound


class _AdoptAgent:
    """One agent, controlling one variable, per the textbook's assumption."""

    __slots__ = (
        "var", "network", "tree", "domain", "parent", "children",
        "lower_neighbors", "epsilon", "threshold", "current_context",
        "lb", "ub", "t", "ctx", "value", "terminated", "terminate_received",
        "inbox", "outbox", "_dirty",
    )

    def __init__(
        self,
        var: Variable,
        network: ConstraintNetwork,
        tree: DFSTree,
        epsilon: float,
    ) -> None:
        self.var = var
        self.network = network
        self.tree = tree
        self.domain = network.domain(var)
        self.parent = tree.parent.get(var)
        self.children = tree.children.get(var, ())
        self.lower_neighbors = tree.lower_neighbors(var)
        self.epsilon = epsilon

        self.threshold: float = 0.0
        self.current_context: dict[Variable, Value] = {}
        self.lb: dict[tuple[Value, Variable], float] = {}
        self.ub: dict[tuple[Value, Variable], float] = {}
        self.t: dict[tuple[Value, Variable], float] = {}
        self.ctx: dict[tuple[Value, Variable], dict[Variable, Value]] = {}
        self.value: Value = self.domain[0]
        self.terminated = False
        self.terminate_received = False

        self.inbox: list[tuple] = []
        self.outbox: list[tuple[Variable, tuple]] = []
        self._dirty = False

    # --------------------------------------------------------------- bounds

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def delta(self, d: Value) -> float:
        """Local cost of ``d``: unary plus every constraint with a higher neighbour.

        ``current_context`` only ever holds higher neighbours' values (VALUE
        messages travel downward) plus non-neighbours picked up from COST
        contexts, and :meth:`ConstraintNetwork.local_cost` ignores the latter.
        """
        return self.network.local_cost(self.var, d, self.current_context)

    def _bound(self, d: Value, table: dict[tuple[Value, Variable], float]) -> float:
        total = self.delta(d)
        for child in self.children:
            total += table[(d, child)]
        return total

    def LB(self, d: Value) -> float:
        return self._bound(d, self.lb)

    def UB(self, d: Value) -> float:
        return self._bound(d, self.ub)

    def min_LB(self) -> float:
        return min(self.LB(d) for d in self.domain)

    def min_UB(self) -> float:
        return min(self.UB(d) for d in self.domain)

    def _argmin(self, bound) -> Value:
        """Value minimising ``bound``, ties broken by domain order.

        Domain order is meaningful in the repair formulation -- ``keep`` is
        always index 0 -- so a stable tie-break biases the solver toward leaving
        plans untouched at equal cost.
        """
        best, best_cost = self.domain[0], bound(self.domain[0])
        for d in self.domain[1:]:
            cost = bound(d)
            if cost < best_cost - EPS:
                best, best_cost = d, cost
        return best

    def _converged(self) -> bool:
        """The termination condition: ``threshold == UB``.

        With a bounded error the relaxation lives in the root's threshold
        invariant, not here, so this test is the same for every agent.
        """
        return self.min_UB() - self.threshold <= EPS

    # ----------------------------------------------------------- invariants

    def maintain_threshold_invariant(self) -> None:
        """``LB <= threshold <= UB``, with the root's bounded-error variant.

        Modi et al. section 6 obtains a solution within ``b`` of optimal by
        letting *only the root* overestimate:

            ThresholdInvariantForRoot(BoundedError): min(LB + b, UB) = threshold

        The root then stops as soon as ``UB - LB <= b``, and because the cost it
        returns is its own upper bound, that solution is provably within ``b``
        of the optimum.  Relaxing the test at every agent instead would stop the
        search earlier but forfeit the guarantee.
        """
        lb, ub = self.min_LB(), self.min_UB()

        if self.is_root and self.epsilon > 0.0:
            self.threshold = min(lb + self.epsilon, ub)
            return

        if self.threshold < lb:
            self.threshold = lb
        if self.threshold > ub:
            self.threshold = ub

    def maintain_child_threshold_invariant(self) -> None:
        for d in self.domain:
            for child in self.children:
                key = (d, child)
                if self.t[key] < self.lb[key]:
                    self.t[key] = self.lb[key]
                if self.t[key] > self.ub[key]:
                    self.t[key] = self.ub[key]

    def maintain_allocation_invariant(self) -> None:
        """Subdivide ``threshold`` across children and tell them their shares.

        The printed pseudocode nudges one child's allocation by 1 at a time
        until the shares sum to ``threshold - delta(value)``.  Costs here are
        floats, so this does the same redistribution in a single pass; the
        fixpoint is identical.
        """
        d = self.value
        for child in self.children:
            key = (d, child)
            if self.t[key] < self.lb[key]:
                self.t[key] = self.lb[key]
            if self.t[key] > self.ub[key]:
                self.t[key] = self.ub[key]

        target = self.threshold - self.delta(d)
        current = sum(self.t[(d, child)] for child in self.children)
        remaining = target - current

        if remaining > EPS:
            for child in self.children:
                if remaining <= EPS:
                    break
                key = (d, child)
                headroom = self.ub[key] - self.t[key]
                give = remaining if headroom == INF else min(remaining, headroom)
                if give > 0:
                    self.t[key] += give
                    remaining -= give
        elif remaining < -EPS:
            deficit = -remaining
            for child in self.children:
                if deficit <= EPS:
                    break
                key = (d, child)
                slack = self.t[key] - self.lb[key]
                take = min(deficit, slack)
                if take > 0:
                    self.t[key] -= take
                    deficit -= take

        for child in self.children:
            self._send(child, (THRESHOLD, self.var, self.t[(d, child)], dict(self.current_context)))

    # -------------------------------------------------------------- protocol

    def initialize(self) -> None:
        self.threshold = 0.0
        self.current_context = {}
        for d in self.domain:
            for child in self.children:
                key = (d, child)
                self.lb[key] = 0.0
                self.t[key] = 0.0
                self.ub[key] = INF
                self.ctx[key] = {}
        self.value = self._argmin(self.LB)
        # Deviation from the printed pseudocode: establishing the invariant here
        # lets a childless root finish immediately instead of waiting for a
        # message that will never arrive.
        self.maintain_threshold_invariant()
        self.backtrack()

    def backtrack(self) -> None:
        if self.terminated:
            return

        if self._converged():
            self.value = self._argmin(self.UB)
        elif self.LB(self.value) > self.threshold + EPS:
            self.value = self._argmin(self.LB)

        for neighbor in self.lower_neighbors:
            self._send(neighbor, (VALUE, self.var, self.value))

        self.maintain_allocation_invariant()

        if self._converged() and (self.terminate_received or self.is_root):
            context = dict(self.current_context)
            context[self.var] = self.value
            for child in self.children:
                self._send(child, (TERMINATE, self.var, context))
            self.terminated = True
            return

        if self.parent is not None:
            self._send(
                self.parent,
                (COST, self.var, dict(self.current_context), self.min_LB(), self.min_UB()),
            )

    # -------------------------------------------------------------- handlers

    def on_value(self, sender: Variable, value: Value) -> None:
        if self.terminate_received:
            return
        self.current_context[sender] = value
        self._discard_incompatible()
        self.maintain_threshold_invariant()
        self._dirty = True

    def on_cost(
        self,
        sender: Variable,
        context: dict[Variable, Value],
        lb: float,
        ub: float,
    ) -> None:
        context = dict(context)
        d = context.pop(self.var, None)

        if not self.terminate_received:
            neighbours = set(self.network.neighbors(self.var))
            for var, val in context.items():
                if var not in neighbours:
                    self.current_context[var] = val
            self._discard_incompatible()

        if d is not None and _compatible(context, self.current_context):
            key = (d, sender)
            if key in self.lb:
                self.lb[key] = lb
                self.ub[key] = ub
                self.ctx[key] = context
                self.maintain_child_threshold_invariant()
                self.maintain_threshold_invariant()

        self._dirty = True

    def on_threshold(self, threshold: float, context: dict[Variable, Value]) -> None:
        if _compatible(context, self.current_context):
            self.threshold = threshold
            self.maintain_threshold_invariant()
            self._dirty = True

    def on_terminate(self, context: dict[Variable, Value]) -> None:
        self.terminate_received = True
        self.current_context = dict(context)
        self.current_context.pop(self.var, None)
        self._discard_incompatible()
        self.maintain_threshold_invariant()
        self._dirty = True

    def _discard_incompatible(self) -> None:
        """Forget child bounds recorded under a context that no longer holds.

        The textbook notes this is why backtrack thresholds exist at all: an
        agent "stores cost information only for the current context and deletes
        previously stored information as soon as the context changes", because
        keeping every context would need exponential memory.
        """
        for key, stored in self.ctx.items():
            if stored and not _compatible(stored, self.current_context):
                self.lb[key] = 0.0
                self.t[key] = 0.0
                self.ub[key] = INF
                self.ctx[key] = {}

    # ---------------------------------------------------------------- mailbox

    def _send(self, dest: Variable, message: tuple) -> None:
        self.outbox.append((dest, message))

    def drain_outbox(self) -> list[tuple[Variable, tuple]]:
        out, self.outbox = self.outbox, []
        return out

    def process_inbox(self) -> None:
        """Consume everything delivered this cycle, then act once.

        This is the textbook's synchronization cycle: "all agents receiving
        incoming messages and sending outgoing messages simultaneously".
        Backtracking after *every* message instead would make an agent
        re-broadcast once per message received, which multiplies traffic
        without changing the fixpoint the algorithm converges to.
        """
        messages, self.inbox = self.inbox, []
        if self.terminated:
            return

        self._dirty = False
        for message in messages:
            kind = message[0]
            if kind == VALUE:
                self.on_value(message[1], message[2])
            elif kind == COST:
                self.on_cost(message[1], message[2], message[3], message[4])
            elif kind == THRESHOLD:
                self.on_threshold(message[2], message[3])
            elif kind == TERMINATE:
                self.on_terminate(message[2])
            if self.terminated:
                return

        if self._dirty:
            self.backtrack()


def _compatible(a: Mapping[Variable, Value], b: Mapping[Variable, Value]) -> bool:
    """Whether two partial assignments agree wherever they overlap."""
    if len(a) > len(b):
        a, b = b, a
    for var, val in a.items():
        other = b.get(var, _MISSING)
        if other is not _MISSING and other != val:
            return False
    return True


_MISSING = object()


def check_non_negative(network: ConstraintNetwork) -> None:
    """Raise if any cost is negative.

    ADOPT initialises each child's lower bound to 0 and refines it upward.  That
    is only an admissible bound when no subtree can cost less than 0, so a
    negative cost silently breaks optimality: the search closes ``LB == UB`` on
    a solution that is not the minimum.  A maximisation problem must therefore
    be re-baselined (cost of *not* achieving the reward) rather than negated.
    """
    for var in network.variables:
        for d in network.domain(var):
            if network.unary_cost(var, d) < -EPS:
                raise ValueError(
                    f"negative unary cost on {var!r}={d!r}; ADOPT requires "
                    f"non-negative costs -- re-baseline the objective"
                )
    for u, v in network.edges():
        for da in network.domain(u):
            for db in network.domain(v):
                if network.binary_cost(u, v, da, db) < -EPS:
                    raise ValueError(
                        f"negative binary cost on ({u!r}={da!r}, {v!r}={db!r}); "
                        f"ADOPT requires non-negative costs -- re-baseline the objective"
                    )


def solve_adopt(
    network: ConstraintNetwork,
    *,
    root: Variable | None = None,
    epsilon: float = 0.0,
    max_cycles: int = 20_000,
    validate: bool = True,
) -> AdoptResult:
    """Minimise ``network`` with ADOPT.

    Args:
        root: force the DFS root of the component containing it.
        epsilon: ADOPT's error bound.  With ``0`` the returned assignment is
            optimal; with ``e > 0`` the search stops once the bound interval
            narrows to ``e``, trading optimality for cycles -- the textbook's
            "the user can specify a valid error bound ... and as soon as the
            bound interval becomes less than this value the search process can
            be stopped".
        max_cycles: safety bound on synchronization cycles.  Reaching it
            returns the best assignment so far with ``terminated=False``.

        validate: check the network has no negative costs first.  Cheap relative
            to the search, and turns a silently suboptimal answer into an error.

    A disconnected network is solved one component at a time; components share
    no constraints, so their costs simply add.
    """
    if validate:
        check_non_negative(network)

    forest = build_dfs_forest(network, root=root)

    assignment: dict[Variable, Value] = {}
    total_cycles = 0
    total_messages = 0
    by_type: dict[str, int] = {VALUE: 0, COST: 0, THRESHOLD: 0, TERMINATE: 0}
    lower = 0.0
    upper = 0.0
    all_terminated = True

    for tree in forest:
        agents = {var: _AdoptAgent(var, network, tree, epsilon) for var in tree.order}

        for agent in agents.values():
            agent.initialize()

        cycles = 0
        while cycles < max_cycles:
            pending: list[tuple[Variable, tuple]] = []
            for agent in agents.values():
                pending.extend(agent.drain_outbox())
            if not pending or all(a.terminated for a in agents.values()):
                break

            cycles += 1
            for dest, message in pending:
                total_messages += 1
                by_type[message[0]] = by_type.get(message[0], 0) + 1
                agents[dest].inbox.append(message)
            for agent in agents.values():
                agent.process_inbox()

        root_agent = agents[tree.root]
        total_cycles = max(total_cycles, cycles)
        lower += root_agent.min_LB()
        upper += root_agent.min_UB()
        all_terminated &= all(a.terminated for a in agents.values())
        for var, agent in agents.items():
            assignment[var] = agent.value

    return AdoptResult(
        assignment=assignment,
        cost=network.evaluate(assignment) if assignment else 0.0,
        lower_bound=lower,
        upper_bound=upper,
        cycles=total_cycles,
        messages=total_messages,
        messages_by_type=by_type,
        terminated=all_terminated,
        n_trees=len(forest),
    )
