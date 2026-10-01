"""DFS pseudo-tree construction -- ADOPT's prerequisite ordering.

Weiss p556: "Before executing the ADOPT algorithm, agents must be arranged in a
depth-first search (DFS) tree", which is useful because "(i) agents in different
branches of the tree do not share any constraints, and (ii) every constraint
network can be ordered in a DFS tree ... in polynomial time".

In an undirected DFS every non-tree edge joins a node to one of its ancestors,
so each variable's constraints split cleanly into:

* **parent** and **pseudo-parents** -- its *higher* neighbours, whose values
  form its context and whose constraints it evaluates locally;
* **children** and **pseudo-children** -- its *lower* neighbours, which receive
  its ``VALUE`` messages.

A disconnected network yields a forest; its trees are independent subproblems
whose costs simply add, so the solver handles each separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import ConstraintNetwork, Variable


@dataclass(frozen=True)
class DFSTree:
    """One tree of the pseudo-forest."""

    root: Variable
    parent: dict[Variable, Variable | None]
    children: dict[Variable, tuple[Variable, ...]]
    pseudo_parents: dict[Variable, tuple[Variable, ...]]
    pseudo_children: dict[Variable, tuple[Variable, ...]]
    order: tuple[Variable, ...] = field(default=())
    depth: dict[Variable, int] = field(default_factory=dict)

    @property
    def variables(self) -> tuple[Variable, ...]:
        return self.order

    def higher_neighbors(self, var: Variable) -> tuple[Variable, ...]:
        """Parent plus pseudo-parents: the neighbours above ``var``."""
        parent = self.parent.get(var)
        pseudo = self.pseudo_parents.get(var, ())
        return (parent, *pseudo) if parent is not None else tuple(pseudo)

    def lower_neighbors(self, var: Variable) -> tuple[Variable, ...]:
        """Children plus pseudo-children: every neighbour that receives VALUE messages."""
        return (*self.children.get(var, ()), *self.pseudo_children.get(var, ()))

    def is_leaf(self, var: Variable) -> bool:
        return not self.children.get(var)

    def height(self) -> int:
        return max(self.depth.values(), default=0)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"DFSTree(root={self.root!r}, n={len(self.order)}, height={self.height()})"


def build_dfs_forest(
    network: ConstraintNetwork, root: Variable | None = None
) -> list[DFSTree]:
    """Arrange ``network`` into a DFS pseudo-forest, one tree per component.

    Roots are chosen by highest degree (a cheap heuristic for a shallow tree --
    the textbook notes finding the *optimal* ordering is itself hard and that
    ADOPT does not address it), with declaration order breaking ties so the
    result is deterministic.  An explicit ``root`` overrides the choice for the
    component containing it.
    """
    adj = {v: set(network.neighbors(v)) for v in network.variables}
    position = network.position

    def sort_key(v: Variable) -> tuple[int, int]:
        return (-len(adj[v]), position(v))

    unvisited = list(network.variables)
    visited: set[Variable] = set()
    trees: list[DFSTree] = []

    while unvisited:
        component_roots = [v for v in unvisited if v not in visited]
        if not component_roots:
            break
        if root is not None and root in component_roots:
            start = root
            root = None  # only honour the override once
        else:
            start = min(component_roots, key=sort_key)
        trees.append(_dfs_from(start, adj, visited, sort_key))
        unvisited = [v for v in unvisited if v not in visited]

    return trees


def _dfs_from(
    start: Variable,
    adj: dict[Variable, set[Variable]],
    visited: set[Variable],
    sort_key,
) -> DFSTree:
    parent: dict[Variable, Variable | None] = {start: None}
    children: dict[Variable, list[Variable]] = {}
    pseudo_parents: dict[Variable, list[Variable]] = {start: []}
    pseudo_children: dict[Variable, list[Variable]] = {}
    depth: dict[Variable, int] = {start: 0}
    preorder: list[Variable] = [start]

    visited.add(start)
    on_path: set[Variable] = {start}
    stack: list[tuple[Variable, object]] = [
        (start, iter(sorted(adj[start], key=sort_key)))
    ]

    while stack:
        node, neighbours = stack[-1]
        descended = False
        for nxt in neighbours:  # type: ignore[union-attr]
            if nxt in visited:
                continue
            visited.add(nxt)
            parent[nxt] = node
            children.setdefault(node, []).append(nxt)
            depth[nxt] = depth[node] + 1
            preorder.append(nxt)

            # Every already-seen neighbour still on the DFS path is an ancestor,
            # so a back edge to it makes it a pseudo-parent.
            pps = [a for a in sorted(adj[nxt], key=sort_key) if a in on_path and a != node]
            pseudo_parents[nxt] = pps
            for ancestor in pps:
                pseudo_children.setdefault(ancestor, []).append(nxt)

            on_path.add(nxt)
            stack.append((nxt, iter(sorted(adj[nxt], key=sort_key))))
            descended = True
            break

        if not descended:
            stack.pop()
            on_path.discard(node)

    return DFSTree(
        root=start,
        parent=parent,
        children={k: tuple(v) for k, v in children.items()},
        pseudo_parents={k: tuple(v) for k, v in pseudo_parents.items()},
        pseudo_children={k: tuple(v) for k, v in pseudo_children.items()},
        order=tuple(preorder),
        depth=depth,
    )
