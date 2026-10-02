"""Turn the experiment JSONL files into the report's figures (PNG).

    python experiments/make_figures.py           # results/  -> figures/
    python experiments/make_figures.py --quick   # results/quick/ -> figures/quick/

Each figure is one chart with one y-axis.  Strategies keep the same colour and
marker everywhere (colour follows the strategy, never its rank) and every line
is labelled directly, so identity never rests on colour alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.summarise_results import NAMES, load  # noqa: E402
from warehouse.metrics import bootstrap_ci  # noqa: E402
from warehouse.repair import STRATEGIES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# Reference categorical palette, slots 1-4 in fixed order (validated adjacent pairs).
COLOR = {"krcbs": "#2a78d6", "local": "#eb6834", "cascade": "#1baf7a", "global": "#eda100"}
MARKER = {"krcbs": "o", "local": "s", "cascade": "^", "global": "D"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
    })


def lines(ax, xs, series: dict, xlabels=None) -> None:
    """series: strategy -> list of (mean, lo, hi) aligned with xs."""
    ends = []
    for s in STRATEGIES:
        if s not in series:
            continue
        pts = [(x, m, lo, hi) for x, (m, lo, hi) in zip(xs, series[s]) if m == m]
        if not pts:
            continue
        x, m, lo, hi = zip(*pts)
        ax.fill_between(x, lo, hi, color=COLOR[s], alpha=0.12, linewidth=0)
        ax.plot(x, m, color=COLOR[s], marker=MARKER[s], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAMES[s])
        ends.append([m[-1], x[-1], NAMES[s]])
    if xlabels:
        ax.set_xticks(xs, xlabels)
    ax.legend(loc="upper left")
    ax.margins(x=0.12)
    # direct end labels, pushed apart so they never overlap
    y0, y1 = ax.get_ylim()
    gap = 0.055 * (y1 - y0)
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, x, name in ends:
        ax.annotate(name, (x, y), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    print("wrote", out / name)


# ---------------------------------------------------------------- figures
def fig_impact(rows, out):
    kinds = ["blockage", "breakdown", "emergency"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.2
    for j, s in enumerate(STRATEGIES):
        ms, errs = [], [[], []]
        for k in kinds:
            m, lo, hi = bootstrap_ci([r["changed"] for r in rows
                                      if r["kind"] == k and r["strategy"] == s and r.get("ok")])
            ms.append(m)
            errs[0].append(m - lo)
            errs[1].append(hi - m)
        xs = [i + (j - 1.5) * width for i in range(len(kinds))]
        ax.bar(xs, ms, width * 0.9, color=COLOR[s], label=NAMES[s], yerr=errs,
               error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    forced = [sum(r["forced"] for r in rows if r["kind"] == k and r["strategy"] == "krcbs")
              / max(1, sum(1 for r in rows if r["kind"] == k and r["strategy"] == "krcbs"))
              for k in kinds]
    for i, f in enumerate(forced):
        ax.hlines(f, i - 2 * width, i + 2 * width, colors=INK, linestyles=(0, (3, 2)), linewidth=1.2)
    ax.set_xticks(range(len(kinds)), kinds)
    ax.set_ylabel("agents whose plan changed")
    ax.set_title("Agents changed to handle a single disruption (E1)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK, linestyle=(0, (3, 2)), linewidth=1.2))
    labels.append("forced (lower bound)")
    ax.legend(handles, labels, ncols=5, loc="upper left", fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    save(fig, out, "fig3_impact.png")


def fig_changed_vs_agents(rows, out):
    ns = sorted({r["n_agents"] for r in rows})
    series = {s: [bootstrap_ci([r["changed"] for r in rows
                                if r["n_agents"] == n and r["strategy"] == s and r.get("ok")])
                  for n in ns] for s in STRATEGIES}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lines(ax, ns, series)
    ax.set_xlabel("number of agents")
    ax.set_ylabel("agents changed per disruption")
    ax.set_title("Impact of one disruption as the fleet grows (E1)")
    save(fig, out, "fig1_scaling_agents.png")


def _episode_series(rows, xkey, metric):
    xs = sorted({r[xkey] for r in rows})
    series = {}
    for s in STRATEGIES:
        series[s] = [bootstrap_ci(metric([r for r in rows if r[xkey] == x and r["strategy"] == s]))
                     for x in xs]
    return xs, series


def per_repair_changed(rs):
    return [rep[2] for r in rs for rep in r["repairs"]]


def soc_increase_pct(rs):
    return [100 * (r["soc"] - r["soc0"]) / r["soc0"] for r in rs if r["completed"]]


def fig_density(rows, out):
    xs, series = _episode_series(rows, "rho", per_repair_changed)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lines(ax, xs, series, [f"{100 * x:g}%" for x in xs])
    ax.set_xlabel("dynamic obstacle density ρ (share of aisle cells blocked)")
    ax.set_ylabel("agents changed per repair")
    ax.set_title(f"Impact per repair vs obstacle density (E2b, {rows[0]['n_agents']} agents)")
    save(fig, out, "fig2_scaling_density.png")

    xs, series = _episode_series(rows, "rho", soc_increase_pct)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lines(ax, xs, series, [f"{100 * x:g}%" for x in xs])
    ax.set_xlabel("dynamic obstacle density ρ")
    ax.set_ylabel("total time increase over undisrupted plan (%)")
    ax.set_title("Total time steps vs obstacle density (E2b)")
    save(fig, out, "fig5_total_time_density.png")


def fig_agents_episodes(rows, out):
    xs, series = _episode_series(rows, "n_agents", soc_increase_pct)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lines(ax, xs, series)
    ax.set_xlabel("number of agents")
    ax.set_ylabel("total time increase over undisrupted plan (%)")
    ax.set_title(f"Total time steps vs fleet size (E2a, ρ = {100 * rows[0]['rho']:g}%)")
    save(fig, out, "fig4_total_time_agents.png")

    xs, series = _episode_series(
        rows, "n_agents", lambda rs: [sum(rep[9] for rep in r["repairs"]) / max(1, len(r["repairs"]))
                                      for r in rs])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lines(ax, xs, series)
    ax.set_yscale("log")
    ax.set_xlabel("number of agents")
    ax.set_ylabel("mean wall-clock time per repair (s, log scale)")
    ax.set_title("Compute cost of a repair (E2a)")
    save(fig, out, "fig7_runtime.png")


def fig_slack(rows, out):
    slacks = sorted({r["slack"] for r in rows}, key=lambda d: (d is None, d or 0))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s in ("krcbs", "local", "global"):
        pts = []
        for d in slacks:
            rs = [r for r in rows if r["slack"] == d and r["strategy"] == s and r.get("ok")]
            if rs:
                pts.append((d, sum(r["emergency_lateness"] for r in rs) / len(rs),
                            sum(r["changed"] for r in rs) / len(rs)))
        if not pts:
            continue
        _, x, y = zip(*pts)
        ax.plot(x, y, color=COLOR[s], marker=MARKER[s], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAMES[s])
        if s == "krcbs":
            for d, xi, yi in pts:
                ax.annotate("δ=∞" if d is None else f"δ={d}", (xi, yi), xytext=(6, 6),
                            textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlabel("emergency delivery lateness (steps after earliest possible)")
    ax.set_ylabel("agents changed per emergency")
    ax.set_title("Emergency deadline slack: urgency vs disruption (E3)")
    ax.legend()
    save(fig, out, "fig6_emergency_slack.png")


def fig_repair_example(out):
    """Before/after picture of one KR-CBS repair that needed a neighbour to move."""
    from warehouse.disruptions import sample_single
    from warehouse.repair import repair
    from warehouse.simulator import SimConfig, apply_event, make_context
    from experiments.run_experiments import _world_at

    for sample in range(200):
        world, rng = _world_at(20, 40_000 + sample)
        ev = sample_single(world, "emergency", rng)
        if ev is None:
            continue
        w = world.copy()
        cfg = SimConfig()
        status, forced, cell = apply_event(w, ev, cfg)
        ctx = make_context(w, forced, cell, cfg)
        res = repair("krcbs", ctx)
        if not res.ok or not res.exact:
            continue
        changed = res.session.close(ctx.plans, res.plans, ctx.t)
        if 2 <= len(changed) <= 4:
            break
    else:
        return
    g = w.grid
    fig, ax = plt.subplots(figsize=(9, 6))
    img = [[0.0] * g.width for _ in range(g.height)]
    for c in range(g.n_cells):
        r, col = g.rc(c)
        img[r][col] = 1.0 if not g.free[c] else 0.0
    ax.imshow(img, cmap="Greys", vmin=0, vmax=3.2, origin="upper")
    for c in g.dock_set:
        r, col = g.rc(c)
        ax.add_patch(plt.Rectangle((col - .5, r - .5), 1, 1, color="#d8d7d2", lw=0))
    for c in g.station_set:
        r, col = g.rc(c)
        ax.add_patch(plt.Rectangle((col - .5, r - .5), 1, 1, color="#b9b8b2", lw=0))
    horizon = ctx.t + 30
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for k, a in enumerate(sorted(changed, key=lambda a: (a not in forced, a))):
        col = colors[k % 4]
        for plan, ls, lw in ((ctx.plans[a], (0, (2, 2)), 1.6), (res.plans[a], "-", 2.4)):
            pts = [g.rc(plan.at(t)) for t in range(ctx.t, min(horizon, max(plan.end, ctx.t)) + 1)]
            ax.plot([p[1] for p in pts], [p[0] for p in pts], color=col, linestyle=ls, lw=lw)
        r, c0 = g.rc(ctx.plans[a].at(ctx.t))
        ax.plot(c0, r, marker="o", markersize=9, color=col, markeredgecolor=INK)
        role = "emergency agent" if a in forced else "released neighbour"
        ax.annotate(f"agent {a} ({role})", (c0, r), xytext=(8, -10), textcoords="offset points",
                    fontsize=8, color=INK)
    for a in sorted(set(ctx.active) - changed):
        r, c0 = g.rc(ctx.plans[a].at(ctx.t))
        ax.plot(c0, r, marker="o", markersize=6, color="#a3a29c")
    pr, pc = g.rc(ev.pickup)
    ax.plot(pc, pr, marker="*", markersize=14, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"One KR-CBS repair: emergency at t={ctx.t}, {len(changed)} of "
                 f"{len(ctx.active)} plans changed")
    ax.text(0, -0.02, "dashed = old plan, solid = repaired plan (next 30 steps), ★ = emergency pickup",
            transform=ax.transAxes, fontsize=9, color=INK2, va="top")
    ax.legend(handles=[Patch(color="#d8d7d2", label="docks"), Patch(color="#b9b8b2", label="stations"),
                       plt.Line2D([], [], marker="o", color="#a3a29c", lw=0, label="unchanged agents")],
              loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.08))
    save(fig, out, "fig8_repair_example.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    style()
    out = ROOT / "figures" / ("quick" if args.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    e1 = load("e1_single", args.quick)
    if e1:
        fig_impact(e1, out)
        fig_changed_vs_agents(e1, out)
    e2a = load("e2a_agents", args.quick)
    if e2a:
        fig_agents_episodes(e2a, out)
    e2b = load("e2b_density", args.quick)
    if e2b:
        fig_density(e2b, out)
    e3 = load("e3_slack", args.quick)
    if e3:
        fig_slack(e3, out)
    fig_repair_example(out)


if __name__ == "__main__":
    main()
