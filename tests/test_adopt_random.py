"""ADOPT cross-checked against exhaustive search.

The textbook example proves the implementation reproduces the taught trace;
these tests prove it is actually a complete optimal solver, on networks with
varied density, domain sizes, disconnected components and hard constraints.
"""

from __future__ import annotations

import pytest

from warehouse.dcop import solve_adopt, solve_bruteforce
from warehouse.dcop.dfstree import build_dfs_forest
from warehouse.dcop.examples import random_network, target_tracking_network
from warehouse.dcop.model import HARD, ConstraintNetwork


@pytest.mark.parametrize("seed", range(25))
def test_adopt_matches_bruteforce_on_random_networks(seed):
    net = random_network(n_variables=5, domain_size=3, edge_probability=0.6, seed=seed)
    expected = solve_bruteforce(net)
    result = solve_adopt(net)

    assert result.terminated, f"seed {seed}: ADOPT did not terminate"
    assert result.cost == pytest.approx(expected.cost), f"seed {seed}: suboptimal"
    assert result.lower_bound == pytest.approx(result.upper_bound)


@pytest.mark.parametrize("seed", range(10))
def test_adopt_matches_bruteforce_on_sparse_networks(seed):
    """Sparse networks are often disconnected, exercising the pseudo-forest path."""
    net = random_network(n_variables=6, domain_size=2, edge_probability=0.25, seed=seed)
    expected = solve_bruteforce(net)
    result = solve_adopt(net)

    assert result.terminated
    assert result.cost == pytest.approx(expected.cost)


@pytest.mark.parametrize("seed", range(8))
def test_adopt_matches_bruteforce_on_dense_networks(seed):
    net = random_network(n_variables=5, domain_size=4, edge_probability=1.0, seed=seed)
    expected = solve_bruteforce(net)
    result = solve_adopt(net)

    assert result.terminated
    assert result.cost == pytest.approx(expected.cost)


def test_disconnected_network_is_solved_per_component():
    net = ConstraintNetwork(("a", "b", "c", "d"), {v: (0, 1) for v in "abcd"})
    net.add_binary("a", "b", {(0, 0): 5.0, (0, 1): 1.0, (1, 0): 3.0, (1, 1): 7.0})
    net.add_binary("c", "d", {(0, 0): 2.0, (0, 1): 9.0, (1, 0): 4.0, (1, 1): 0.0})

    assert len(build_dfs_forest(net)) == 2
    result = solve_adopt(net)
    assert result.n_trees == 2
    assert result.terminated
    assert result.cost == pytest.approx(solve_bruteforce(net).cost)
    assert result.cost == pytest.approx(1.0)  # a=0,b=1 gives 1; c=1,d=1 gives 0


def test_isolated_variable_picks_its_cheapest_value():
    """A variable with no constraints still has to settle on its best value."""
    net = ConstraintNetwork(("lonely",), {"lonely": (0, 1, 2)})
    net.add_unary("lonely", {0: 7.0, 1: 2.0, 2: 5.0})

    result = solve_adopt(net)
    assert result.terminated
    assert result.assignment["lonely"] == 1
    assert result.cost == pytest.approx(2.0)


def test_hard_constraints_are_respected_when_satisfiable():
    """Hard constraints priced at HARD, per the encoding described on p550."""
    net = ConstraintNetwork(("p", "q"), {"p": (0, 1), "q": (0, 1)})
    # Forbid agreement outright; make p=1 mildly preferred.
    net.add_binary("p", "q", {(0, 0): HARD, (1, 1): HARD, (0, 1): 0.0, (1, 0): 0.0})
    net.add_unary("p", {0: 3.0, 1: 0.0})

    result = solve_adopt(net)
    assert result.terminated
    assert result.assignment["p"] != result.assignment["q"]
    assert net.is_feasible(result.assignment)
    assert result.cost == pytest.approx(0.0)


def test_target_tracking_instance_pairs_sensors_onto_targets():
    """Ch 12 section 3.1.2: two cameras cooperating to identify a target."""
    coverage = {
        "s1": ("t1",),
        "s2": ("t1", "t2"),
        "s3": ("t2",),
    }
    net = target_tracking_network(coverage, reward=10.0, energy=1.0)
    expected = solve_bruteforce(net)
    result = solve_adopt(net)

    assert result.terminated
    assert result.cost == pytest.approx(expected.cost)
    # Exactly one target can be jointly covered, since s2 must pick a side.
    aimed = [v for v in result.assignment.values() if v != "idle"]
    assert len(aimed) == 2 and len(set(aimed)) == 1
    # One of the two edges stays uncovered (10) plus two active sensors (2).
    assert result.cost == pytest.approx(12.0)


@pytest.mark.parametrize("epsilon", [0.0, 1.0, 3.0, 10.0])
def test_error_bound_is_honoured(epsilon):
    """With bound e, the returned solution is within e of optimal."""
    net = random_network(n_variables=5, domain_size=3, edge_probability=0.7, seed=99)
    optimum = solve_bruteforce(net).cost
    result = solve_adopt(net, epsilon=epsilon)

    assert result.cost <= optimum + epsilon + 1e-9


def test_negative_costs_are_rejected():
    """A negated maximisation would break ADOPT's bounds, so it is refused."""
    net = ConstraintNetwork(("m", "n"), {"m": (0, 1), "n": (0, 1)})
    net.add_binary("m", "n", {(0, 0): -5.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0})

    with pytest.raises(ValueError, match="non-negative"):
        solve_adopt(net)


def test_a_loose_error_bound_finishes_sooner():
    """A bound wide enough to span the cost range short-circuits the search.

    Deliberately not asserted for *every* bound: the relaxation changes the
    search trajectory, so a narrow bound can occasionally cost an extra cycle.
    """
    net = random_network(n_variables=6, domain_size=3, edge_probability=0.7, seed=7)
    exact = solve_adopt(net, epsilon=0.0)
    loose = solve_adopt(net, epsilon=50.0)

    assert loose.cycles <= exact.cycles
