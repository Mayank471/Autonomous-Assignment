"""A committed plan: where one agent will be at every future time step.

``cells[k]`` is the agent's cell at time ``t0 + k``.  A plan always ends at the
moment the agent arrives at its dock with every goal done; from then on the
agent stays parked there forever, so ``at(t)`` returns the last cell for any
``t`` past the end.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    t0: int
    cells: tuple[int, ...]

    @property
    def end(self) -> int:
        """Time of the last step (arrival at the dock, i.e. completion)."""
        return self.t0 + len(self.cells) - 1

    def at(self, t: int) -> int:
        k = t - self.t0
        if k < 0:
            raise ValueError(f"plan starts at {self.t0}, asked for t={t}")
        return self.cells[k] if k < len(self.cells) else self.cells[-1]

    def suffix(self, t: int) -> "Plan":
        """The same trajectory, re-based to start at time ``t``."""
        if t <= self.t0:
            return self
        k = t - self.t0
        if k >= len(self.cells):
            return Plan(t, (self.cells[-1],))
        return Plan(t, self.cells[k:])


def plans_differ(a: Plan, b: Plan, t_from: int) -> bool:
    end = max(a.end, b.end)
    return any(a.at(t) != b.at(t) for t in range(t_from, end + 1))


def plan_distance(a: Plan, b: Plan, t_from: int) -> int:
    """Number of time steps (from ``t_from``) at which the two plans put the
    agent in different cells -- a plan-stability measure."""
    end = max(a.end, b.end)
    return sum(1 for t in range(t_from, end + 1) if a.at(t) != b.at(t))
