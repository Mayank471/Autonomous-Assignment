"""ADOPT checked against Modi et al. (2005), the paper the algorithm was taught from.

The paper uses a *different* running example from Weiss Figure 12.1 -- different
edges and a different cost function -- so reproducing both is a stronger check
than either alone.  Every number here is stated outright in the paper:

* Section 2, on Fig. 2(a): "x1 and x3 are neighbors but x1 and x4 are not",
  "all four constraints are identical", ``F({all 0}) = 4``, ``F({all 1}) = 0``,
  and "A* = {(x1,1), (x2,1), (x3,1), (x4,1)}".
* Section 4.1, on Fig. 2(b): "x1 is the root, x1 is the parent of x2, and x2 is
  the parent of both x3 and x4".
* Section 6: bounded-error approximation via the root's threshold invariant,
  ``min(LB + b, UB) = threshold``.
"""

from __future__ import annotations

import pytest

from warehouse.dcop import solve_adopt, solve_bruteforce
from warehouse.dcop.dfstree import build_dfs_forest
from warehouse.dcop.examples import (
    ADOPT_PAPER_COSTS,
    ADOPT_PAPER_EDGES,
    adopt_paper_network,
)

ALL_ZERO = {"x1": 0, "x2": 0, "x3": 0, "x4": 0}
ALL_ONE = {"x1": 1, "x2": 1, "x3": 1, "x4": 1}


# ----------------------------------------------------------------- the graph


def test_neighbour_structure_matches_the_paper():
    """"x1 and x3 are neighbors but x1 and x4 are not"."""
    net = adopt_paper_network()

    assert "x3" in net.neighbors("x1")
    assert "x4" not in net.neighbors("x1")
    assert len(ADOPT_PAPER_EDGES) == 4, "the paper's example has four constraints"


def test_all_four_constraints_are_identical():
    net = adopt_paper_network()
    for u, v in ADOPT_PAPER_EDGES:
        for a in (0, 1):
            for b in (0, 1):
                assert net.binary_cost(u, v, a, b) == ADOPT_PAPER_COSTS[(a, b)]


def test_the_two_totals_the_paper_quotes():
    """F({all 0}) = 4 and F({all 1}) = 0 -- which is what pins the cost table."""
    net = adopt_paper_network()
    assert net.evaluate(ALL_ZERO) == 4.0
    assert net.evaluate(ALL_ONE) == 0.0


# ------------------------------------------------------------- the DFS tree


def test_dfs_tree_matches_figure_2b():
    """"x1 is the root, x1 is the parent of x2, and x2 is the parent of both x3 and x4"."""
    tree = build_dfs_forest(adopt_paper_network(), root="x1")[0]

    assert tree.root == "x1"
    assert tree.parent["x2"] == "x1"
    assert tree.parent["x3"] == "x2"
    assert tree.parent["x4"] == "x2"
    assert tree.children["x1"] == ("x2",)
    assert set(tree.children["x2"]) == {"x3", "x4"}

    # x1-x3 is a constraint but not a tree edge, so x1 is x3's pseudo-parent.
    assert tree.pseudo_parents["x3"] == ("x1",)
    assert "x3" in tree.lower_neighbors("x1")


def test_dfs_tree_is_valid_in_the_paper_sense():
    """"a DFS tree is valid if there are no constraints between agents in
    different subtrees" -- every edge joins an agent to an ancestor."""
    net = adopt_paper_network()
    tree = build_dfs_forest(net, root="x1")[0]

    def ancestors(var):
        out, node = set(), tree.parent.get(var)
        while node is not None:
            out.add(node)
            node = tree.parent.get(node)
        return out

    for u, v in net.edges():
        assert u in ancestors(v) or v in ancestors(u), (
            f"{u}-{v} crosses two subtrees, so the tree is not a valid DFS tree"
        )


# ---------------------------------------------------------------- the solve


def test_adopt_finds_the_optimum_the_paper_states():
    """A* = {(x1,1), (x2,1), (x3,1), (x4,1)} at cost 0."""
    net = adopt_paper_network()
    result = solve_adopt(net, root="x1")

    assert result.terminated
    assert result.assignment == ALL_ONE
    assert result.cost == 0.0


def test_bruteforce_agrees_and_the_optimum_is_unique():
    net = adopt_paper_network()
    exact = solve_bruteforce(net)

    assert exact.cost == 0.0
    assert len(exact.optima) == 1
    assert exact.optima[0] == ALL_ONE


def test_optimum_is_found_whatever_root_is_chosen():
    """DFS orderings are not unique; the answer must not depend on the choice."""
    net = adopt_paper_network()
    for root in ("x1", "x2", "x3", "x4"):
        result = solve_adopt(net, root=root)
        assert result.terminated, f"root={root} did not terminate"
        assert result.cost == 0.0, f"root={root} was suboptimal"


def test_adopt_terminates_when_the_bound_interval_closes():
    net = adopt_paper_network()
    result = solve_adopt(net, root="x1")
    assert result.lower_bound == pytest.approx(result.upper_bound)
    assert result.gap == pytest.approx(0.0)


# ------------------------------------------------------- bounded error, s6


@pytest.mark.parametrize("bound", [0.0, 1.0, 4.0, 10.0])
def test_bounded_error_stays_within_its_promise(bound):
    """Section 6: the result must be within ``b`` of the optimum."""
    net = adopt_paper_network()
    optimum = solve_bruteforce(net).cost
    result = solve_adopt(net, root="x1", epsilon=bound)

    assert result.cost <= optimum + bound + 1e-9


def test_a_loose_error_bound_finishes_sooner():
    """The point of the error bound is to stop early.

    Not monotone in ``b``, and the paper never claims it is: raising the root's
    threshold changes which branches get explored, so a small bound can cost an
    extra cycle before it saves any.  What must hold is that a bound large
    enough to cover the whole cost range finishes no later than proving
    optimality outright.
    """
    net = adopt_paper_network()
    exact = solve_adopt(net, root="x1", epsilon=0.0)
    loose = solve_adopt(net, root="x1", epsilon=10.0)

    assert loose.cycles <= exact.cycles
    assert loose.messages <= exact.messages


def test_root_threshold_invariant_uses_the_error_bound():
    """The relaxation belongs to the root alone, per ThresholdInvariantForRoot.

    A non-root agent keeps the strict invariant, so the guarantee survives.
    """
    from warehouse.dcop.adopt import _AdoptAgent

    net = adopt_paper_network()
    tree = build_dfs_forest(net, root="x1")[0]

    root = _AdoptAgent("x1", net, tree, epsilon=4.0)
    root.initialize()
    # LB is 0 and UB infinite at start, so min(LB + b, UB) = b.
    assert root.threshold == pytest.approx(4.0)

    child = _AdoptAgent("x2", net, tree, epsilon=4.0)
    child.initialize()
    assert child.threshold <= child.min_UB() + 1e-9
    assert child.threshold >= child.min_LB() - 1e-9
