"""Re-run only the rows of ``results/runs.csv`` a code change can have affected.

A full sweep costs over an hour, which is too slow to repeat for a fix that
touches one rarely-taken path.  When a change is confined to a branch that most
runs never enter, the rest of the CSV is still valid and only the affected rows
need recomputing.

That reasoning has to be justified per change, not assumed.  The change this was
written for is inside
:func:`~warehouse.simulator.enforce_no_conflicts`, which begins by checking
whether any conflict exists at all and returns immediately when none does.  A
run that never hit a conflict therefore executed identical code before and
after, so only runs that recorded a safety-net activation (or a surviving
collision) can differ.

Usage::

    python experiments/refresh_runs.py --where "safety_truncations>0 or collisions_detected>0"
    python experiments/refresh_runs.py --where "..." --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_experiments import (  # noqa: E402
    Job,
    _as_metrics,
    run_job,
    write_csv,
)
from warehouse.metrics import aggregate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def coerce(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value in ("", None):
            out[key] = None
        elif value in ("True", "False"):
            out[key] = value == "True"
        else:
            try:
                out[key] = int(value)
            except ValueError:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
    return out


def job_for(row: dict[str, Any]) -> Job:
    return Job(
        strategy=row["strategy"],
        n_agents=int(row["n_agents"]),
        density=float(row["density"]),
        seed=int(row["seed"]),
        beta=float(row["beta"]),
        tau=int(row["tau"]),
        k_max=int(row["k_max"]),
        epsilon=float(row["epsilon"]),
        social_laws=bool(row["social_laws"]),
        sweep=row["sweep"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--where",
        required=True,
        help="Python expression over the row's columns; true means re-run.",
    )
    parser.add_argument("--csv", type=Path, default=RESULTS / "runs.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = [coerce(r) for r in csv.DictReader(handle)]

    stale = [
        i
        for i, row in enumerate(rows)
        if eval(args.where, {"__builtins__": {}}, {k: (v or 0) for k, v in row.items()})
    ]
    print(f"{len(stale)} of {len(rows)} rows match and will be re-run")
    if args.dry_run or not stale:
        return

    started = time.perf_counter()
    for count, index in enumerate(stale, 1):
        old = rows[index]
        fresh = run_job(job_for(old))
        fresh["sweep"] = old["sweep"]
        rows[index] = fresh
        if count % 5 == 0 or count == len(stale):
            elapsed = time.perf_counter() - started
            print(
                f"  {count}/{len(stale)}  {elapsed/60:.1f} min elapsed",
                flush=True,
            )

    write_csv(args.csv, rows)
    print(f"Rewrote {args.csv.relative_to(ROOT)} with {len(stale)} refreshed rows")

    # The per-sweep summaries are derived, so they go stale the moment any row
    # changes.  Regenerate them here rather than leaving artefacts that disagree
    # with the data they came from.
    out = args.csv.parent
    main_rows = [r for r in rows if r["sweep"] == "main"]
    if main_rows:
        write_csv(
            out / "summary_main.csv",
            aggregate(
                [_as_metrics(r) for r in main_rows], "strategy", "n_agents", "density"
            ),
        )
    for sweep in ("beta", "tau", "epsilon", "k_max", "social_laws"):
        subset = [r for r in rows if r["sweep"] == sweep]
        if subset:
            write_csv(out / f"summary_{sweep}.csv", subset)
    print("Regenerated the derived summary CSVs")


if __name__ == "__main__":
    main()
