"""Metric collection.

The assignment asks for three things, and they map onto fields here directly:

* *"the total time steps required by all the agents to accomplish their
  respective tasks"* -> :attr:`RunMetrics.total_timesteps`;
* *"the number of agents whose original plans are to be changed to handle a
  single disruption"* -> :attr:`RunMetrics.mean_agents_changed`, with the full
  per-episode distribution kept in :attr:`RunMetrics.agents_changed`;
* *"how the performance of the system changes as the number of agents changes
  and the density of the dynamic obstacles is varied"* -> the sweep in
  :mod:`experiments.run_experiments`, which records one :class:`RunMetrics` per
  configuration.

Everything else here exists to make those three numbers interpretable: repair
time against replan time, ADOPT's message and cycle counts, how often the
fallback ladder was needed, and how many tasks actually got done.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass
class RunMetrics:
    """Everything measured about one simulation run."""

    # --- configuration --------------------------------------------------
    strategy: str = ""
    n_agents: int = 0
    density: float = 0.0
    seed: int = 0
    beta: float = 0.0
    tau: int = 0
    k_max: int = 0
    epsilon: float = 0.0
    social_laws: bool = False

    # --- headline outcome ------------------------------------------------
    total_timesteps: int = 0
    makespan: int = 0
    tasks_completed: int = 0
    tasks_total: int = 0
    agents_finished: int = 0
    agents_stranded: int = 0

    # --- disruption impact ----------------------------------------------
    n_disruptions: int = 0
    disruptions_by_kind: dict[str, int] = field(default_factory=dict)
    agents_changed: list[int] = field(default_factory=list)
    agents_involved: list[int] = field(default_factory=list)
    repair_hops: list[int] = field(default_factory=list)
    reported_changes: list[int] = field(default_factory=list)
    #: Size of each individual negotiation group (``k_max`` bounds these).
    episode_sizes: list[int] = field(default_factory=list)

    # --- coordination cost ----------------------------------------------
    repair_ms: list[float] = field(default_factory=list)
    repair_messages: list[int] = field(default_factory=list)
    repair_cycles: list[int] = field(default_factory=list)
    low_level_searches: int = 0

    # --- robustness -------------------------------------------------------
    repair_failures: int = 0
    abandoned_agents: int = 0
    #: Episodes where ADOPT ran out of cycles before proving optimality.
    solver_timeouts: int = 0
    #: Agents the final safety pass had to trim.  Should be ~0; a non-zero count
    #: means a repair produced an unsafe plan that the net had to catch.
    safety_truncations: int = 0
    initial_plan_ms: float = 0.0
    initial_sum_of_costs: int = 0
    collisions_detected: int = 0
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- derived

    @property
    def mean_agents_changed(self) -> float:
        """Agents altered per disruption -- the assignment's impact metric."""
        return _mean(self.agents_changed)

    @property
    def max_agents_changed(self) -> int:
        return max(self.agents_changed, default=0)

    @property
    def mean_agents_involved(self) -> float:
        """Distinct agents drawn into repairing one disruption, across all groups."""
        return _mean(self.agents_involved)

    @property
    def max_episode_size(self) -> int:
        """Largest single negotiation group -- what ``k_max`` actually bounds."""
        return max(self.episode_sizes, default=0)

    @property
    def mean_episode_size(self) -> float:
        return _mean(self.episode_sizes)

    @property
    def mean_reported_changes(self) -> float:
        """Changes that exceeded the dampening threshold and were broadcast."""
        return _mean(self.reported_changes)

    @property
    def mean_repair_ms(self) -> float:
        return _mean(self.repair_ms)

    @property
    def total_repair_ms(self) -> float:
        return sum(self.repair_ms)

    @property
    def mean_repair_messages(self) -> float:
        return _mean(self.repair_messages)

    @property
    def mean_repair_cycles(self) -> float:
        return _mean(self.repair_cycles)

    @property
    def completion_rate(self) -> float:
        return self.tasks_completed / self.tasks_total if self.tasks_total else 0.0

    @property
    def repairs_performed(self) -> int:
        return len(self.agents_changed)

    @property
    def cost_over_initial(self) -> float:
        """How much the disruptions cost, relative to the undisrupted plan."""
        if not self.initial_sum_of_costs:
            return 0.0
        return self.total_timesteps / self.initial_sum_of_costs

    def record_repair(self, outcome) -> None:
        """Fold one repair episode's result into the run totals."""
        self.agents_changed.append(outcome.n_changed)
        self.agents_involved.append(outcome.n_involved)
        self.reported_changes.append(len(outcome.reported))
        self.repair_ms.append(outcome.wall_ms)
        self.repair_messages.append(outcome.messages)
        self.repair_cycles.append(outcome.cycles)
        self.repair_hops.append(outcome.hops)
        self.episode_sizes.extend(outcome.episode_sizes or (outcome.n_involved,))
        self.low_level_searches += outcome.searches
        if not outcome.success:
            self.repair_failures += 1
        self.abandoned_agents += len(outcome.abandoned)
        if outcome.timed_out:
            self.solver_timeouts += 1
        if outcome.note:
            self.notes.append(outcome.note)

    def as_row(self) -> dict[str, Any]:
        """Flatten to one CSV row."""
        return {
            "strategy": self.strategy,
            "n_agents": self.n_agents,
            "density": self.density,
            "seed": self.seed,
            "beta": self.beta,
            "tau": self.tau,
            "k_max": self.k_max,
            "epsilon": self.epsilon,
            "social_laws": self.social_laws,
            "total_timesteps": self.total_timesteps,
            "makespan": self.makespan,
            "tasks_completed": self.tasks_completed,
            "tasks_total": self.tasks_total,
            "completion_rate": round(self.completion_rate, 4),
            "agents_finished": self.agents_finished,
            "agents_stranded": self.agents_stranded,
            "n_disruptions": self.n_disruptions,
            "repairs_performed": self.repairs_performed,
            "mean_agents_changed": round(self.mean_agents_changed, 4),
            "max_agents_changed": self.max_agents_changed,
            "mean_agents_involved": round(self.mean_agents_involved, 4),
            "mean_episode_size": round(self.mean_episode_size, 4),
            "max_episode_size": self.max_episode_size,
            "mean_reported_changes": round(self.mean_reported_changes, 4),
            "mean_repair_ms": round(self.mean_repair_ms, 4),
            "total_repair_ms": round(self.total_repair_ms, 3),
            "mean_repair_messages": round(self.mean_repair_messages, 2),
            "mean_repair_cycles": round(self.mean_repair_cycles, 2),
            "mean_repair_hops": round(_mean(self.repair_hops), 3),
            "low_level_searches": self.low_level_searches,
            "repair_failures": self.repair_failures,
            "abandoned_agents": self.abandoned_agents,
            "solver_timeouts": self.solver_timeouts,
            "safety_truncations": self.safety_truncations,
            "initial_plan_ms": round(self.initial_plan_ms, 3),
            "initial_sum_of_costs": self.initial_sum_of_costs,
            "cost_over_initial": round(self.cost_over_initial, 4),
            "collisions_detected": self.collisions_detected,
            **{f"disruption_{k}": v for k, v in self.disruptions_by_kind.items()},
        }


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def confidence_interval(values: Sequence[float], confidence: float = 0.95) -> float:
    """Half-width of the normal-approximation CI for the mean.

    Used for the error bands on the figures.  With 30 seeds per configuration
    the normal approximation is adequate and avoids a scipy dependency in the
    plotting path.
    """
    n = len(values)
    if n < 2:
        return 0.0
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    return z * statistics.stdev(values) / (n**0.5)


def aggregate(runs: Iterable[RunMetrics], *keys: str) -> list[dict[str, Any]]:
    """Group runs by ``keys`` and summarise the headline metrics with CIs."""
    buckets: dict[tuple, list[RunMetrics]] = {}
    for run in runs:
        buckets.setdefault(tuple(getattr(run, k) for k in keys), []).append(run)

    out: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        row: dict[str, Any] = dict(zip(keys, key))
        row["n_runs"] = len(group)
        for metric in (
            "total_timesteps",
            "makespan",
            "mean_agents_changed",
            "mean_agents_involved",
            "mean_repair_ms",
            "mean_repair_messages",
            "completion_rate",
            "repair_failures",
        ):
            values = [float(getattr(r, metric)) for r in group]
            row[f"{metric}_mean"] = round(_mean(values), 4)
            row[f"{metric}_ci"] = round(confidence_interval(values), 4)
        out.append(row)
    return out
