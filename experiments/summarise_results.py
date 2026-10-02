"""Summarise the experiment JSONL files into the report's tables.

    python experiments/summarise_results.py            # full results
    python experiments/summarise_results.py --quick    # results/quick/

Writes ``results/summary.md`` (or ``results/quick/summary.md``) and prints it.
Means are reported with 95% bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from warehouse.metrics import bootstrap_ci, fmt_ci  # noqa: E402
from warehouse.repair import STRATEGIES  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
FALLBACK_LEVELS = {"neighbourhood", "global", "failed"}
NAMES = {"krcbs": "KR-CBS", "local": "Local", "cascade": "Cascade", "global": "Global"}


def load(name: str, quick: bool) -> list[dict]:
    path = RESULTS / ("quick" if quick else "") / f"{name}.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if not r.get("skipped")]


def is_fallback(strategy: str, level: str | None) -> bool:
    return strategy != "global" and level in FALLBACK_LEVELS


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ----------------------------------------------------------------------- E1
def summarise_single(rows: list[dict]) -> str:
    if not rows:
        return ""
    parts = ["## E1 - Agents changed to handle a single disruption\n"]
    n_dis = len({(r["n_agents"], r["kind"], r["sample"]) for r in rows})
    parts.append(f"{n_dis} disruptions; every strategy repaired the identical state. "
                 "`changed` includes the disrupted (forced) agents; `collateral` = changed - forced.\n")
    by = defaultdict(list)
    for r in rows:
        by[(r["kind"], r["strategy"])].append(r)
        by[("all", r["strategy"])].append(r)
    hdr = ["disruption", "strategy", "forced", "changed", "collateral", "ΔSOC (steps)",
           "messages", "fallback", "exact"]
    body = []
    for kind in ("blockage", "breakdown", "emergency", "all"):
        for s in STRATEGIES:
            rs = [r for r in by[(kind, s)] if r.get("ok")]
            if not rs:
                continue
            body.append([kind, NAMES[s],
                         f"{sum(r['forced'] for r in rs) / len(rs):.2f}",
                         fmt_ci(*bootstrap_ci([r["changed"] for r in rs])),
                         fmt_ci(*bootstrap_ci([r["collateral"] for r in rs])),
                         fmt_ci(*bootstrap_ci([r["soc_delta"] for r in rs]), 1),
                         f"{sum(r['messages'] for r in rs) / len(rs):.0f}",
                         pct(sum(is_fallback(s, r["level"]) for r in rs) / len(rs)),
                         pct(sum(bool(r["exact"]) for r in rs) / len(rs)) if s == "krcbs" else "-"])
    parts.append(table(hdr, body))
    # by number of agents
    ns = sorted({r["n_agents"] for r in rows})
    hdr = ["agents"] + [NAMES[s] for s in STRATEGIES] + ["forced"]
    body = []
    for n in ns:
        line = [str(n)]
        for s in STRATEGIES:
            rs = [r for r in rows if r["n_agents"] == n and r["strategy"] == s and r.get("ok")]
            line.append(fmt_ci(*bootstrap_ci([r["changed"] for r in rs])))
        rs = [r for r in rows if r["n_agents"] == n and r["strategy"] == "krcbs"]
        line.append(f"{sum(r['forced'] for r in rs) / max(1, len(rs)):.2f}")
        body.append(line)
    parts.append("\n**Mean agents changed per disruption, by fleet size (all disruption types)**\n")
    parts.append(table(hdr, body))
    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------- E2
def summarise_episodes(rows: list[dict], xkey: str, title: str, xfmt) -> str:
    if not rows:
        return ""
    parts = [f"## {title}\n"]
    xs = sorted({r[xkey] for r in rows})
    hdr = [xkey, "strategy", "completed", "SOC", "ΔSOC vs undisrupted", "makespan",
           "repairs/episode", "changed/repair", "forced/repair", "fallback", "runtime/episode (s)"]
    body = []
    for x in xs:
        for s in STRATEGIES:
            rs = [r for r in rows if r[xkey] == x and r["strategy"] == s]
            if not rs:
                continue
            reps = [rep for r in rs for rep in r["repairs"]]
            done = [r for r in rs if r["completed"]]
            body.append([
                xfmt(x), NAMES[s], f"{len(done)}/{len(rs)}",
                fmt_ci(*bootstrap_ci([r["soc"] for r in done]), 0),
                fmt_ci(*bootstrap_ci([r["soc"] - r["soc0"] for r in done]), 0),
                f"{sum(r['makespan'] for r in done) / max(1, len(done)):.0f}",
                f"{len(reps) / len(rs):.1f}",
                fmt_ci(*bootstrap_ci([rep[2] for rep in reps])),
                f"{sum(rep[1] for rep in reps) / max(1, len(reps)):.2f}",
                pct(sum(is_fallback(s, rep[4]) for rep in reps) / max(1, len(reps))),
                f"{sum(r['runtime_s'] for r in rs) / len(rs):.1f}",
            ])
    parts.append(table(hdr, body))
    dens = [r["realized_density"] for r in rows]
    parts.append(f"\nRealised time-averaged blocked fraction: {min(dens):.3f}-{max(dens):.3f}.\n")
    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------- E3
def summarise_slack(rows: list[dict]) -> str:
    if not rows:
        return ""
    parts = ["## E3 - Emergency deadline slack δ\n",
             "Lateness = emergency delivery time minus the earliest possible delivery time.\n"]
    slacks = sorted({r["slack"] for r in rows}, key=lambda d: (d is None, d or 0))
    hdr = ["δ", "strategy", "changed", "collateral", "lateness (steps)", "deadline relaxed"]
    body = []
    for d in slacks:
        for s in ("krcbs", "local", "global"):
            rs = [r for r in rows if r["slack"] == d and r["strategy"] == s and r.get("ok")]
            if not rs:
                continue
            body.append(["∞ (none)" if d is None else str(d), NAMES[s],
                         fmt_ci(*bootstrap_ci([r["changed"] for r in rs])),
                         fmt_ci(*bootstrap_ci([r["collateral"] for r in rs])),
                         fmt_ci(*bootstrap_ci([r["emergency_lateness"] for r in rs]), 1),
                         pct(sum(bool(r["deadline_relaxed"]) for r in rs) / len(rs))])
    parts.append(table(hdr, body))
    return "\n".join(parts) + "\n"


# ----------------------------------------------------------------------- E4
def summarise_exactness(rows: list[dict]) -> str:
    if not rows:
        return ""
    parts = ["## E4 - Exactness of KR-CBS against brute force\n"]
    hdr = ["agents", "disruptions", "KR-CBS exact", "changed = minimum", "SOC = minimum",
           "oracle errors", "mean changed", "KR-CBS time (s)", "brute force time (s)"]
    body = []
    for n in sorted({r["n_agents"] for r in rows}):
        rs = [r for r in rows if r["n_agents"] == n]
        ex = [r for r in rs if r["kr_exact"] and r["bf_changed"] is not None]
        body.append([str(n), str(len(rs)), f"{sum(r['kr_exact'] for r in rs)}/{len(rs)}",
                     f"{sum(r['kr_changed'] == r['bf_changed'] for r in ex)}/{len(ex)}",
                     f"{sum(r['kr_soc'] == r['bf_soc'] for r in ex)}/{len(ex)}",
                     str(sum(r["bf_error"] is not None for r in rs)),
                     f"{sum(r['kr_changed'] for r in rs) / len(rs):.2f}",
                     f"{sum(r['kr_runtime_s'] for r in rs) / len(rs):.2f}",
                     f"{sum(r['bf_runtime_s'] for r in rs) / len(rs):.2f}"])
    parts.append(table(hdr, body))
    nonexact = [r for r in rows if not r["kr_exact"] and r["bf_changed"] is not None]
    if nonexact:
        gap = [r["kr_changed"] - r["bf_changed"] for r in nonexact]
        parts.append(f"\nWhen KR-CBS ran out of budget ({len(nonexact)} cases) its answer "
                     f"exceeded the minimum by {sum(gap) / len(gap):.2f} agents on average.\n")
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    q = args.quick
    md = ["# Results summary\n"]
    md.append(summarise_single(load("e1_single", q)))
    e2a, e2b = load("e2a_agents", q), load("e2b_density", q)
    if e2a:
        md.append(summarise_episodes(e2a, "n_agents", "E2a - Scaling with the number of agents "
                                     f"(ρ = {100 * e2a[0]['rho']:g}%)", str))
    if e2b:
        md.append(summarise_episodes(e2b, "rho", "E2b - Scaling with obstacle density "
                                     f"({e2b[0]['n_agents']} agents)", lambda x: f"{100 * x:g}%"))
    md.append(summarise_slack(load("e3_slack", q)))
    md.append(summarise_exactness(load("e4_exactness", q)))
    text = "\n".join(m for m in md if m)
    out = RESULTS / ("quick" if q else "") / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
