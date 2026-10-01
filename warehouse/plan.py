"""The plan object exchanged between agents and stored in the reservation table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    """A single agent's space-time route.

    ``cells[i]`` is the agent's location at absolute time ``start_time + i``.
    Consecutive entries are either equal (a wait) or 4-connected neighbours.
    An agent is understood to remain at ``cells[-1]`` for all later times,
    which is what makes the final cell a *permanent* reservation.
    """

    agent: int
    start_time: int
    cells: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("a plan must contain at least one cell")

    @property
    def end_time(self) -> int:
        """Absolute time at which the agent reaches its final cell."""
        return self.start_time + len(self.cells) - 1

    @property
    def goal(self) -> int:
        return self.cells[-1]

    @property
    def duration(self) -> int:
        """Number of time steps spent moving or waiting (``end_time - start_time``)."""
        return len(self.cells) - 1

    def at(self, t: int) -> int:
        """Location at absolute time ``t``, clamped to the plan's endpoints."""
        if t <= self.start_time:
            return self.cells[0]
        if t >= self.end_time:
            return self.cells[-1]
        return self.cells[t - self.start_time]

    def moves(self):
        """Yield ``(from_cell, to_cell, t)`` for each transition, ``t`` its start."""
        for i in range(len(self.cells) - 1):
            yield self.cells[i], self.cells[i + 1], self.start_time + i

    def suffix_from(self, t: int) -> "Plan":
        """The portion of this plan from absolute time ``t`` onward."""
        if t <= self.start_time:
            return self
        if t >= self.end_time:
            return Plan(self.agent, t, (self.cells[-1],))
        return Plan(self.agent, t, self.cells[t - self.start_time :])

    def concat(self, other: "Plan") -> "Plan":
        """Append ``other`` to this plan; ``other`` must start where this one ends."""
        if other.start_time != self.end_time:
            raise ValueError(
                f"cannot concatenate: ends at {self.end_time}, next starts at {other.start_time}"
            )
        if other.cells[0] != self.cells[-1]:
            raise ValueError("cannot concatenate: plans do not meet at the same cell")
        return Plan(self.agent, self.start_time, self.cells + other.cells[1:])

    def with_agent(self, agent: int) -> "Plan":
        return Plan(agent, self.start_time, self.cells)

    def __len__(self) -> int:
        return len(self.cells)
