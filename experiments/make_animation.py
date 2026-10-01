"""Render a warehouse run as an animated GIF.

Shows a single scenario executing: agents moving along their routes, cells
going dark as they are blocked, and agents changing colour for a few frames
when a repair alters their plan.  The point is to make the *locality* of repair
visible -- when a disruption lands, only a handful of nearby agents light up
while the rest of the fleet carries on undisturbed.

Usage::

    python experiments/make_animation.py                       # default scenario
    python experiments/make_animation.py --strategy full_replan
    python experiments/make_animation.py --agents 25 --density 0.02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from warehouse import scenario  # noqa: E402
from warehouse.repair import RepairConfig  # noqa: E402
from warehouse.simulator import simulate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

FLOOR = "#F5F3EF"
SHELF = "#C3BDB4"
BLOCKED = "#6E2A2A"
AGENT = "#2E6FA7"
REPAIRED = "#D4761E"
DOCK = "#DCD6CC"


def build_frames(args) -> tuple:
    grid = scenario.make_grid(n_rows=args.rows, n_cols=args.cols)
    sc = scenario.build(grid, args.agents, args.seed)
    disruptions = sc.disruptions(density=args.density)

    result = simulate(
        grid,
        sc.fresh_tasks(),
        sc.solution,
        disruptions,
        strategy=args.strategy,
        config=RepairConfig(),
        seed=args.seed,
        move_filter=sc.move_filter,
        record_history=True,
    )
    return grid, sc, result, disruptions


def render(args) -> Path:
    grid, sc, result, disruptions = build_frames(args)
    history = result.history
    if not history:
        raise SystemExit("simulation produced no frames")

    # When each cell becomes blocked, so the animation reveals them in time.
    blocked_at: dict[int, int] = {}
    for when, kind, cell in result.disruption_log:
        if kind in ("CellBlockage", "AgentBreakdown"):
            blocked_at.setdefault(cell, when)

    base = np.full((grid.n_rows, grid.n_cols, 3), _rgb(FLOOR))
    for r in range(grid.n_rows):
        for c in range(grid.n_cols):
            if not grid.passable[r, c]:
                base[r, c] = _rgb(SHELF)
    for dock in grid.dock_points:
        r, c = grid.rc(dock)
        base[r, c] = _rgb(DOCK)

    step = max(1, args.every)
    frames = list(range(0, len(history), step))

    fig, ax = plt.subplots(figsize=(args.cols / 9, args.rows / 9))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    image = ax.imshow(base, interpolation="nearest")
    scatter = ax.scatter([], [], s=args.dot, c=AGENT, edgecolors="white", linewidths=0.4, zorder=3)
    title = ax.set_title("", fontsize=9, loc="left")

    def draw(index: int):
        now = frames[index]
        canvas = base.copy()
        for cell, when in blocked_at.items():
            if when <= now:
                r, c = grid.rc(cell)
                canvas[r, c] = _rgb(BLOCKED)
        image.set_data(canvas)

        positions = history[now]
        rows, cols, colours = [], [], []
        for agent, cell in positions.items():
            r, c = grid.rc(cell)
            rows.append(r)
            cols.append(c)
            colours.append(AGENT)
        scatter.set_offsets(np.column_stack([cols, rows]))
        scatter.set_color(colours)

        events = sum(1 for when, _, _ in result.disruption_log if when <= now)
        title.set_text(
            f"{args.strategy}   t={now}   disruptions so far: {events}   "
            f"blocked cells: {sum(1 for w in blocked_at.values() if w <= now)}"
        )
        return image, scatter, title

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=args.interval, blit=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / args.out
    animation.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    metrics = result.metrics
    print(
        f"  {len(frames)} frames | {metrics.n_disruptions} disruptions | "
        f"{metrics.repairs_performed} repairs | "
        f"{metrics.mean_agents_changed:.2f} agents changed per disruption"
    )
    return out


def _rgb(hex_colour: str) -> tuple[float, float, float]:
    hex_colour = hex_colour.lstrip("#")
    return tuple(int(hex_colour[i : i + 2], 16) / 255 for i in (0, 2, 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="adopt")
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--density", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--cols", type=int, default=36)
    parser.add_argument("--every", type=int, default=2, help="keep every Nth step")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--interval", type=int, default=80)
    parser.add_argument("--dot", type=float, default=22)
    parser.add_argument("--out", default="warehouse_run.gif")
    args = parser.parse_args()

    print(f"Rendering {args.strategy} with {args.agents} agents...")
    out = render(args)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
