"""Render a warehouse run as an animated GIF (figures/warehouse_run.gif).

    python experiments/make_animation.py --agents 15 --rho 0.03 --seed 1

Agents are dots (ring = broken down); blocked cells are dark squares; an agent
whose plan was changed by the most recent repair flashes with a halo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warehouse.disruptions import EventRates, make_schedule  # noqa: E402
from warehouse.scenario import make_instance  # noqa: E402
from warehouse.simulator import SimConfig, World, handle_event, initial_plans  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=15)
    ap.add_argument("--rho", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--strategy", default="krcbs")
    args = ap.parse_args()

    inst = make_instance(args.agents, args.seed)
    plans = initial_plans(inst)
    horizon = max(p.end for p in plans.values())
    events = sorted(make_schedule(inst, horizon, EventRates(rho=args.rho), args.seed),
                    key=lambda e: (e.t, e.eid))
    cfg = SimConfig(strategy=args.strategy)
    world = World(inst, plans)
    frames = []
    pending = list(events)
    halo: dict[int, int] = {}
    while not world.all_done() and world.t < 1000:
        due = [e for e in pending if e.t <= world.t]
        pending = [e for e in pending if e.t > world.t]
        for ev in due:
            old = dict(world.plans)
            status, rec = handle_event(world, ev, cfg)
            if status == "deferred":
                pending.append(type(ev)(ev.kind, world.t + 1, ev.eid, ev.cell, ev.agent,
                                        ev.duration, ev.pickup, ev.station))
            for a in range(world.n_agents):
                if world.plans[a] is not old[a]:
                    halo[a] = world.t + 4
        blocked = [c for c, s, e, o in world.obstacles.items() if s <= world.t < e and o < 0]
        frames.append((world.t, [world.position(a) for a in range(world.n_agents)],
                       [world.is_broken(a) for a in range(world.n_agents)], blocked,
                       {a for a, until in halo.items() if until >= world.t}))
        world.step()

    g = inst.grid
    fig, ax = plt.subplots(figsize=(8, 5.4))
    img = [[1.0 if not g.free[r * g.width + c] else 0.0 for c in range(g.width)]
           for r in range(g.height)]
    ax.imshow(img, cmap="Greys", vmin=0, vmax=3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    dots = ax.scatter([], [], s=60, c="#2a78d6", edgecolors="#fcfcfb", linewidths=1, zorder=3)
    rings = ax.scatter([], [], s=160, facecolors="none", edgecolors="#e34948", linewidths=2, zorder=4)
    halos = ax.scatter([], [], s=260, facecolors="none", edgecolors="#eda100", linewidths=2, zorder=2)
    blocks = ax.scatter([], [], s=120, marker="s", c="#0b0b0b", zorder=1)
    title = ax.set_title("")

    def xy(cells):
        return [(g.rc(c)[1], g.rc(c)[0]) for c in cells] or [(float("nan"), float("nan"))]

    def update(i):
        t, pos, broken, blocked, changed = frames[i]
        dots.set_offsets(xy(pos))
        rings.set_offsets(xy([p for p, b in zip(pos, broken) if b]))
        halos.set_offsets(xy([pos[a] for a in changed]))
        blocks.set_offsets(xy(blocked))
        title.set_text(f"t = {t}   ({args.strategy}; orange halo = plan just changed, "
                       f"red ring = broken down)")
        return dots, rings, halos, blocks, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=120, blit=False)
    out = ROOT / "figures" / "warehouse_run.gif"
    anim.save(out, writer=PillowWriter(fps=8))
    print("wrote", out)


if __name__ == "__main__":
    main()
