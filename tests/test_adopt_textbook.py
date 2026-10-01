"""ADOPT checked against the worked example in Weiss Ch 12.

Every number asserted here is read from the textbook (Figures 12.1-12.3 and the
prose on pp556-560) or derived from its cost table, so these tests pin the
implementation to the algorithm as taught rather than to itself.
"""

from __future__ import annotations

import pytest

from warehouse.dcop import solve_adopt, solve_bruteforce
from warehouse.dcop.dfstree import build_dfs_forest
from warehouse.dcop.examples import (
    TEXTBOOK_COSTS,
    textbook_constraint_network,
)


# --------------------------------------------------------------------- network


def test_cost_table_matches_the_figure():
    """Figure 12.1's table, confirmed by the worked trace on p558."""
    assert TEXTBOOK_COSTS[(0, 0)] == 2.0
    assert TEXTBOOK_COSTS[(0, 1)] == 0.0
    assert TEXTBOOK_COSTS[(1, 0)] == 0.0
    assert TEXTBOOK_COSTS[(1, 1)] == 1.0


def test_network_shape():
    net = textbook_constraint_network()
    assert net.variables == ("x1", "x2", "x3", "x4")
    assert net.neighbors("x1") == ("x2", "x3", "x4")
    assert net.neighbors("x2") == ("x1", "x4")
    assert net.neighbors("x3") == ("x1",)
    assert net.neighbors("x4") == ("x1", "x2")


# -------------------------------------------------------------- local costs


def test_local_cost_of_x4_matches_p558():
    """p558: under context {x1=0, x2=0}, delta(x4=0)=4 and delta(x4=1)=0.

    The textbook uses these two numbers to explain how A4 computes the minimum
    lower bound it reports to its parent.
    """
    net = textbook_constraint_network()
    context = {"x1": 0, "x2": 0}
    assert net.local_cost("x4", 0, context) == 4.0
    assert net.local_cost("x4", 1, context) == 0.0


def test_lower_bounds_of_x2_match_p558():
    """p558: LB(x2=0) = delta(x2=0) + lb(x2=0, x4) = 2, and LB(x2=1) = 0.

    With A4's reported lower bound still 0, the LB is just A2's local cost, so
    A2 switches from 0 to 1 -- the value change the figure shows.
    """
    net = textbook_constraint_network()
    context = {"x1": 0}
    assert net.local_cost("x2", 0, context) == 2.0
    assert net.local_cost("x2", 1, context) == 0.0


# ---------------------------------------------------------------- DFS tree


def test_dfs_tree_matches_figure_12_2():
    """The pseudo-tree of Figure 12.2: A1 root, A2 and A3 its children, A4 under A2.

    The figure's point is that A1 still sends a VALUE message to A4 "even though
    there is no parent/child relationship between A1 and A4, because A4 is a
    neighbor of A1 who is lower in the DFS order" -- i.e. A1 is A4's
    pseudo-parent.
    """
    net = textbook_constraint_network()
    forest = build_dfs_forest(net)
    assert len(forest) == 1
    tree = forest[0]

    assert tree.root == "x1"
    assert tree.parent["x2"] == "x1"
    assert tree.parent["x3"] == "x1"
    assert tree.parent["x4"] == "x2"
    assert set(tree.children["x1"]) == {"x2", "x3"}
    assert tree.children["x2"] == ("x4",)

    assert tree.pseudo_parents["x4"] == ("x1",)
    assert tree.pseudo_children["x1"] == ("x4",)
    assert "x4" in tree.lower_neighbors("x1")
    assert set(tree.higher_neighbors("x4")) == {"x1", "x2"}


# ------------------------------------------------------------------- solving


def test_bruteforce_optimum_is_one():
    net = textbook_constraint_network()
    result = solve_bruteforce(net)
    assert result.cost == 1.0
    assert result.n_evaluated == 16

    optima = {tuple(a[v] for v in net.variables) for a in result.optima}
    assert optima == {(0, 1, 1, 1), (1, 0, 0, 1), (1, 1, 0, 0)}


def test_adopt_finds_the_optimum():
    net = textbook_constraint_network()
    result = solve_adopt(net)

    assert result.terminated, "ADOPT should prove optimality on a four-variable network"
    assert result.cost == 1.0
    assert tuple(result.assignment[v] for v in net.variables) in {
        (0, 1, 1, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 0),
    }


def test_adopt_proves_optimality_by_closing_the_bound_interval():
    """Termination condition: the search stops "when these two values agree"."""
    net = textbook_constraint_network()
    result = solve_adopt(net)

    assert result.lower_bound == pytest.approx(result.upper_bound)
    assert result.lower_bound == pytest.approx(1.0)
    assert result.gap == pytest.approx(0.0)


def test_adopt_exchanges_all_four_message_types():
    net = textbook_constraint_network()
    result = solve_adopt(net)

    assert result.messages > 0
    for kind in ("VALUE", "COST", "THRESHOLD", "TERMINATE"):
        assert result.messages_by_type[kind] > 0, f"no {kind} messages were sent"


def test_error_bound_stops_early_within_tolerance():
    """The bound interval doubles as a quality guarantee, per p556."""
    net = textbook_constraint_network()
    exact = solve_adopt(net, epsilon=0.0)
    loose = solve_adopt(net, epsilon=2.0)

    assert loose.cycles <= exact.cycles
    assert loose.cost <= exact.cost + 2.0
