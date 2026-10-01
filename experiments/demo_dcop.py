"""Runnable walkthrough of the DCOP material from Weiss Ch 12 and Modi et al.

Four parts:

1. **The exemplar constraint network of Weiss Figure 12.1** -- prints the DFS
   pseudo-tree it is arranged into, the local costs quoted on p558, the message
   counts by type, and checks the answer against exhaustive search.
2. **The constraint network of Modi et al. (2005) Figure 2** -- the example
   ADOPT was actually taught from.  It is a *different* network from the
   textbook's: different edges and a different cost table.
3. **ADOPT's bounded-error approximation** (paper section 6) -- the root's
   threshold invariant relaxed to ``min(LB + b, UB)``, trading optimality for
   synchronization cycles while keeping a provable guarantee.
4. **The target-tracking problem of Weiss section 3.1.2** -- sensors with
   overlapping ranges deciding jointly which targets to cover, solved by the
   same engine the warehouse uses for emergency response.

Usage::

    python experiments/demo_dcop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warehouse.dcop import solve_adopt, solve_bruteforce  # noqa: E402
from warehouse.dcop.dfstree import build_dfs_forest  # noqa: E402
from warehouse.dcop.examples import (  # noqa: E402
    ADOPT_PAPER_COSTS,
    ADOPT_PAPER_EDGES,
    TEXTBOOK_COSTS,
    adopt_paper_network,
    target_tracking_network,
    textbook_constraint_network,
)

RULE = "=" * 74


def heading(text: str) -> None:
    print()
    print(RULE)
    print(text)
    print(RULE)


def show_tree(tree) -> None:
    for var in tree.order:
        children = ", ".join(tree.children.get(var, ())) or "-"
        pseudo = tree.pseudo_parents.get(var, ())
        extra = f", pseudo-parent(s)={', '.join(pseudo)}" if pseudo else ""
        print(f"    {var}: parent={tree.parent[var]}, children={children}{extra}")


def demo_textbook_network() -> None:
    heading("1. Exemplar constraint network (Weiss Figure 12.1)")

    network = textbook_constraint_network()
    print("Variables :", ", ".join(map(str, network.variables)), "over domain {0, 1}")
    print("Edges     :", ", ".join(f"{u}-{v}" for u, v in network.edges()))
    print("Cost F(xi, xj), shared by every edge:")
    for (a, b), cost in sorted(TEXTBOOK_COSTS.items()):
        print(f"    F({a}, {b}) = {cost:g}")

    tree = build_dfs_forest(network)[0]
    print()
    print(f"DFS pseudo-tree (root {tree.root}):")
    show_tree(tree)
    print()
    print("  Note x1 sends VALUE to x4 even though it is not x4's parent: it is")
    print("  a neighbour higher in the DFS order, exactly as Figure 12.2 shows.")

    print()
    print("Local costs quoted on p558, under the context {x1=0, x2=0}:")
    for value in (0, 1):
        delta = network.local_cost("x4", value, {"x1": 0, "x2": 0})
        print(f"    delta(x4={value}) = {delta:g}")
    print("  and under {x1=0}:")
    for value in (0, 1):
        delta = network.local_cost("x2", value, {"x1": 0})
        print(f"    delta(x2={value}) = {delta:g}")

    exact = solve_bruteforce(network)
    result = solve_adopt(network)

    print()
    print(f"Exhaustive search: optimum {exact.cost:g} over {exact.n_evaluated} assignments")
    print("  attained at:", ", ".join(
        "(" + ", ".join(f"{v}={a[v]}" for v in network.variables) + ")"
        for a in exact.optima
    ))
    print()
    print("ADOPT: " + ", ".join(f"{v}={result.assignment[v]}" for v in network.variables)
          + f"  cost {result.cost:g}")
    print(f"  bounds LB={result.lower_bound:g} UB={result.upper_bound:g} "
          f"-> gap {result.gap:g} (optimality proved: {result.terminated})")
    print(f"  {result.cycles} synchronization cycles, {result.messages} messages:")
    for kind, count in result.messages_by_type.items():
        print(f"      {kind:<10s} {count}")
    verdict = "matches" if abs(result.cost - exact.cost) < 1e-9 else "DIFFERS FROM"
    print(f"  ADOPT {verdict} the exhaustive optimum.")


def demo_paper_network() -> None:
    heading("2. Constraint network of Modi et al. (2005), Figure 2")

    network = adopt_paper_network()
    print("The paper's running example is NOT the textbook's.  Its edges are:")
    print("    " + ", ".join(f"{u}-{v}" for u, v in ADOPT_PAPER_EDGES))
    print("  so x1 and x3 are neighbours but x1 and x4 are not.  Its costs:")
    for (a, b), cost in sorted(ADOPT_PAPER_COSTS.items()):
        print(f"    f({a}, {b}) = {cost:g}")

    all_zero = {v: 0 for v in network.variables}
    all_one = {v: 1 for v in network.variables}
    print()
    print(f"  F(all zeros) = {network.evaluate(all_zero):g}    (paper states 4)")
    print(f"  F(all ones)  = {network.evaluate(all_one):g}    (paper states 0)")

    tree = build_dfs_forest(network, root="x1")[0]
    print()
    print("DFS tree of Fig. 2(b): x1 root, x1 parent of x2, x2 parent of x3 and x4:")
    show_tree(tree)

    result = solve_adopt(network, root="x1")
    exact = solve_bruteforce(network)
    print()
    print("ADOPT: " + ", ".join(f"{v}={result.assignment[v]}" for v in network.variables)
          + f"  cost {result.cost:g}")
    print("  paper states A* = {(x1,1), (x2,1), (x3,1), (x4,1)} at cost 0")
    print(f"  exhaustive optimum {exact.cost:g}; optimality proved: {result.terminated}; "
          f"{result.cycles} cycles, {result.messages} messages")


def demo_error_bound() -> None:
    heading("3. Bounded-error approximation (Modi et al. section 6)")

    print("The paper relaxes the ROOT's threshold invariant to")
    print("    min(LB + b, UB) = threshold")
    print("so the root stops once UB - LB <= b.  The cost it returns is its own")
    print("upper bound, so that solution is provably within b of the optimum.")
    print("Non-root agents keep the strict invariant, which is what preserves")
    print("the guarantee.")
    print()

    network = adopt_paper_network()
    optimum = solve_bruteforce(network).cost
    print(f"{'b':>5s} {'cost':>7s} {'excess':>8s} {'within b?':>10s} "
          f"{'cycles':>8s} {'messages':>9s}")
    for bound in (0.0, 1.0, 2.0, 4.0, 10.0):
        result = solve_adopt(network, root="x1", epsilon=bound)
        excess = result.cost - optimum
        ok = "yes" if excess <= bound + 1e-9 else "NO"
        print(f"{bound:5.1f} {result.cost:7.1f} {excess:8.1f} {ok:>10s} "
              f"{result.cycles:8d} {result.messages:9d}")


def demo_target_tracking() -> None:
    heading("4. Target tracking (Weiss section 3.1.2)")

    coverage = {
        "sensor1": ("target_A",),
        "sensor2": ("target_A", "target_B"),
        "sensor3": ("target_B",),
        "sensor4": ("target_B", "target_C"),
        "sensor5": ("target_C",),
    }
    print("Sensing ranges:")
    for sensor, targets in coverage.items():
        print(f"    {sensor}: {', '.join(targets)}")
    print()
    print("A target counts as identified when two sensors aim at it together, so")
    print("the pairwise constraint charges the full reward when a pair fails to")
    print("agree and nothing when it does.  Aiming costs energy, which stops")
    print("every sensor piling onto the same target.")

    network = target_tracking_network(coverage, reward=10.0, energy=1.0)
    exact = solve_bruteforce(network)
    result = solve_adopt(network)

    print()
    print(f"ADOPT assignment (cost {result.cost:g}, optimum {exact.cost:g}):")
    for sensor in coverage:
        print(f"    {sensor} -> {result.assignment[sensor]}")

    covered: dict[str, list[str]] = {}
    for sensor, choice in result.assignment.items():
        if choice != "idle":
            covered.setdefault(choice, []).append(sensor)

    print()
    print("Targets covered:")
    for target, sensors in sorted(covered.items()):
        mark = "identified" if len(sensors) >= 2 else "single sensor only"
        print(f"    {target}: {', '.join(sorted(sensors))} -- {mark}")
    print()
    print(f"  {result.cycles} cycles, {result.messages} messages, "
          f"optimality proved: {result.terminated}")
    print()
    print("  The warehouse uses this same builder and solver to decide which")
    print("  robots divert to an emergency -- see warehouse/emergency.py.")


def main() -> None:
    demo_textbook_network()
    demo_paper_network()
    demo_error_bound()
    demo_target_tracking()
    print()


if __name__ == "__main__":
    main()
