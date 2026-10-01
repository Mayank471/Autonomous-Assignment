"""Turn the experiment CSVs into the report's figures.

Reads ``results/runs.csv`` and writes PNGs to ``figures/``.  Each figure answers
one question from the assignment; the docstring of each ``figure_*`` function
says which.

Usage::

    python experiments/make_figures.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from warehouse.metrics import confidence_interval  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# One colour and marker per strategy, used consistently across every figure so
# a reader learns the legend once.
STYLE: dict[str, dict[str, Any]] = {
    "selfish": dict(color="#B3573F", marker="o", label="selfish (no negotiation)"),
    "greedy_pgp": dict(color="#C99A2E", marker="s", label="greedy PGP"),
    "adopt": dict(color="#2E6FA7", marker="^", label="ADOPT (optimal DCOP)"),
    "cbs": dict(color="#3F7A5E", marker="v", label="CBS (beyond syllabus)"),
    "full_replan": dict(color="#5C5C5C", marker="D", label="full replan (baseline)"),
}
ORDER = ("selfish", "greedy_pgp", "adopt", "cbs", "full_replan")

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.autolayout": True,
    }
)


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"{path} not found -- run experiments/run_experiments.py first")
    with path.open(newline="", encoding="utf-8") as handle:
        return [_coerce(row) for row in csv.DictReader(handle)]


def _coerce(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value in ("", None):
            out[key] = None
            continue
        if value in ("True", "False"):
            out[key] = value == "True"
            continue
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def series(
    rows: Iterable[dict[str, Any]], x_key: str, y_key: str
) -> tuple[list[float], list[float], list[float]]:
    """Mean and 95% CI of ``y_key`` grouped by ``x_key``."""
    buckets: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        x, y = row.get(x_key), row.get(y_key)
        if x is not None and y is not None:
            buckets[float(x)].append(float(y))
    xs = sorted(buckets)
    return (
        xs,
        [sum(buckets[x]) / len(buckets[x]) for x in xs],
        [confidence_interval(buckets[x]) for x in xs],
    )


def plot_by_strategy(
    ax,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    strategies: Sequence[str] = ORDER,
) -> None:
    for strategy in strategies:
        subset = [r for r in rows if r.get("strategy") == strategy]
        if not subset:
            continue
        xs, ys, cis = series(subset, x_key, y_key)
        if not xs:
            continue
        style = STYLE[strategy]
        ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=1.6, markersize=4.5)
        ax.fill_between(
            xs,
            [y - c for y, c in zip(ys, cis)],
            [y + c for y, c in zip(ys, cis)],
            color=style["color"], alpha=0.15, linewidth=0,
        )


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------- the figures


def figure_scaling_with_agents(rows: list[dict[str, Any]]) -> None:
    """Assignment question 3a: performance as the number of agents changes."""
    main = [r for r in rows if r["sweep"] == "main" and r["density"] > 0]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    plot_by_strategy(axes[0], main, "n_agents", "total_timesteps")
    axes[0].set(xlabel="agents", ylabel="total time steps (all agents)",
                title="Cost of completing the work")

    plot_by_strategy(axes[1], main, "n_agents", "mean_agents_changed")
    axes[1].set(xlabel="agents", ylabel="agents replanned per disruption",
                title="Impact of a single disruption")

    plot_by_strategy(axes[2], main, "n_agents", "mean_repair_ms")
    axes[2].set(xlabel="agents", ylabel="repair time per disruption (ms)",
                title="Cost of deciding", yscale="log")

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Scaling with fleet size", fontsize=11, y=1.04)
    save(fig, "fig1_scaling_agents.png")


def figure_scaling_with_density(rows: list[dict[str, Any]]) -> None:
    """Assignment question 3b: performance as dynamic-obstacle density varies."""
    main = [r for r in rows if r["sweep"] == "main"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    plot_by_strategy(axes[0], main, "density", "total_timesteps")
    axes[0].set(xlabel="dynamic obstacle density", ylabel="total time steps",
                title="Cost of completing the work")

    plot_by_strategy(axes[1], main, "density", "mean_agents_changed")
    axes[1].set(xlabel="dynamic obstacle density", ylabel="agents replanned per disruption",
                title="Impact of a single disruption")

    plot_by_strategy(axes[2], main, "density", "completion_rate")
    axes[2].set(xlabel="dynamic obstacle density", ylabel="tasks completed",
                title="Did the work get done?")

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Scaling with disruption density", fontsize=11, y=1.04)
    save(fig, "fig2_scaling_density.png")


def figure_impact_breakdown(rows: list[dict[str, Any]]) -> None:
    """The headline impact metric, and how much of it gets broadcast."""
    main = [r for r in rows if r["sweep"] == "main" and r["density"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

    labels, changed, involved, errs = [], [], [], []
    for strategy in ORDER:
        subset = [r for r in main if r["strategy"] == strategy]
        if not subset:
            continue
        values = [float(r["mean_agents_changed"]) for r in subset]
        labels.append(STYLE[strategy]["label"].split(" (")[0])
        changed.append(sum(values) / len(values))
        errs.append(confidence_interval(values))
        inv = [float(r["mean_agents_involved"]) for r in subset]
        involved.append(sum(inv) / len(inv))

    positions = range(len(labels))
    axes[0].bar(positions, involved, color="#D8D8D8", label="consulted")
    axes[0].bar(positions, changed, yerr=errs, capsize=3,
                color=[STYLE[s]["color"] for s in ORDER[: len(labels)]],
                label="actually replanned")
    axes[0].set_xticks(list(positions))
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set(ylabel="agents per disruption",
                title="Consulted vs replanned")
    axes[0].legend(fontsize=8)

    plot_by_strategy(axes[1], main, "n_agents", "mean_episode_size")
    axes[1].set(xlabel="agents", ylabel="agents per negotiation group",
                title="Negotiation group size (k_max bounds this)")
    axes[1].legend(fontsize=8)

    fig.suptitle("How far a disruption propagates", fontsize=11, y=1.04)
    save(fig, "fig3_impact.png")


def figure_beta_tradeoff(rows: list[dict[str, Any]]) -> None:
    """The ``beta`` knob: buying fewer altered plans with extra travel time."""
    subset = [r for r in rows if r["sweep"] == "beta"]
    if not subset:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

    xs, ys, cis = series(subset, "beta", "mean_agents_changed")
    axes[0].errorbar(xs, ys, yerr=cis, marker="o", color="#2E6FA7",
                     capsize=3, linewidth=1.6, markersize=4.5)
    axes[0].set(xlabel=r"change penalty $\beta$", ylabel="agents replanned per disruption",
                title="Raising the penalty disturbs fewer agents")

    # The Pareto view: churn against cost, with beta as the parameter.
    by_beta: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        by_beta[float(row["beta"])].append(row)
    churn, cost, betas = [], [], []
    for beta in sorted(by_beta):
        group = by_beta[beta]
        churn.append(sum(float(r["mean_agents_changed"]) for r in group) / len(group))
        cost.append(sum(float(r["total_timesteps"]) for r in group) / len(group))
        betas.append(beta)
    axes[1].plot(churn, cost, "-o", color="#B3573F", linewidth=1.6, markersize=5)
    for x, y, beta in zip(churn, cost, betas):
        axes[1].annotate(f"{beta:g}", (x, y), fontsize=7.5,
                         textcoords="offset points", xytext=(5, 4))
    axes[1].set(xlabel="agents replanned per disruption", ylabel="total time steps",
                title=r"Trade-off frontier (labels: $\beta$)")

    fig.suptitle(r"Minimising altered plans, as an objective", fontsize=11, y=1.04)
    save(fig, "fig4_beta_tradeoff.png")


def figure_solver_comparison(rows: list[dict[str, Any]]) -> None:
    """Greedy PGP against optimal ADOPT, and ADOPT's error bound."""
    main = [r for r in rows if r["sweep"] == "main" and r["density"] > 0]
    eps = [r for r in rows if r["sweep"] == "epsilon"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    plot_by_strategy(axes[0], main, "n_agents", "mean_repair_messages",
                     strategies=("greedy_pgp", "adopt"))
    axes[0].set(xlabel="agents", ylabel="messages per disruption",
                title="Coordination traffic", yscale="log")
    axes[0].legend(fontsize=8)

    plot_by_strategy(axes[1], main, "n_agents", "mean_agents_changed",
                     strategies=("greedy_pgp", "adopt", "cbs"))
    axes[1].set(xlabel="agents", ylabel="agents replanned per disruption",
                title="Does searching the plan space pay off?")
    axes[1].legend(fontsize=8)

    if eps:
        xs, ys, cis = series(eps, "epsilon", "mean_repair_ms")
        axes[2].errorbar(xs, ys, yerr=cis, marker="^", color="#2E6FA7",
                         capsize=3, linewidth=1.6)
        axes[2].set(xlabel=r"ADOPT error bound $\epsilon$",
                    ylabel="repair time per disruption (ms)",
                    title="Bounded error buys speed")
    else:
        axes[2].axis("off")

    fig.suptitle("Greedy hill-climbing vs complete DCOP search", fontsize=11, y=1.04)
    save(fig, "fig5_solver_comparison.png")


def figure_parameter_sweeps(rows: list[dict[str, Any]]) -> None:
    """PGP's dampening threshold, the neighbourhood cap, and social laws."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    tau = [r for r in rows if r["sweep"] == "tau"]
    if tau:
        xs, ys, cis = series(tau, "tau", "mean_reported_changes")
        axes[0].errorbar(xs, ys, yerr=cis, marker="o", color="#2E6FA7",
                         capsize=3, linewidth=1.6, label="broadcast")
        xs2, ys2, _ = series(tau, "tau", "mean_agents_changed")
        axes[0].plot(xs2, ys2, "--", color="#999999", label="actually changed")
        axes[0].set(xlabel=r"dampening threshold $\tau$",
                    ylabel="agents per disruption",
                    title="PGP mechanism 6: slack cuts chatter")
        axes[0].legend(fontsize=8)

    k = [r for r in rows if r["sweep"] == "k_max"]
    if k:
        xs, ys, cis = series(k, "k_max", "total_timesteps")
        axes[1].errorbar(xs, ys, yerr=cis, marker="s", color="#B3573F",
                         capsize=3, linewidth=1.6)
        axes[1].set(xlabel=r"neighbourhood cap $k_{max}$", ylabel="total time steps",
                    title="How wide should a huddle be?")

    laws = [r for r in rows if r["sweep"] == "social_laws"]
    if laws:
        labels, off, on = [], [], []
        for strategy in ORDER:
            subset = [r for r in laws if r["strategy"] == strategy]
            if not subset:
                continue
            a = [float(r["total_timesteps"]) for r in subset if not r["social_laws"]]
            b = [float(r["total_timesteps"]) for r in subset if r["social_laws"]]
            if not a or not b:
                continue
            labels.append(strategy.replace("_", " "))
            off.append(sum(a) / len(a))
            on.append(sum(b) / len(b))
        positions = range(len(labels))
        width = 0.38
        axes[2].bar([p - width / 2 for p in positions], off, width,
                    color="#9AA7B1", label="no convention")
        axes[2].bar([p + width / 2 for p in positions], on, width,
                    color="#2E6FA7", label="one-way aisles")
        axes[2].set_xticks(list(positions))
        axes[2].set_xticklabels(labels, rotation=18, ha="right")
        axes[2].set(ylabel="total time steps", title="Social laws (Ch 11 s3.1)")
        axes[2].legend(fontsize=8)

    fig.suptitle("Parameter sweeps", fontsize=11, y=1.04)
    save(fig, "fig6_parameters.png")


FIGURES_TO_MAKE: tuple[Callable[[list[dict[str, Any]]], None], ...] = (
    figure_scaling_with_agents,
    figure_scaling_with_density,
    figure_impact_breakdown,
    figure_beta_tradeoff,
    figure_solver_comparison,
    figure_parameter_sweeps,
)


def main() -> None:
    rows = load(RESULTS / "runs.csv")
    print(f"Loaded {len(rows)} runs from {RESULTS / 'runs.csv'}")
    for make in FIGURES_TO_MAKE:
        try:
            make(rows)
        except Exception as error:  # keep going; a missing sweep is not fatal
            print(f"  !! {make.__name__} failed: {error}")
    print(f"Figures in {FIGURES}")


if __name__ == "__main__":
    main()
