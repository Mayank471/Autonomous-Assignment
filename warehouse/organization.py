"""The meta-level organization -- Weiss Ch 11 section 3.2 and PGP mechanism 2.

Section 3.2 introduces organizational structuring as coordination fixed *before*
planning: "an organizational structure defines, for example, a set of roles and
relationships among them".  Section 6.3 then names the coordination-level
counterpart explicitly:

    "PGP assumes that responsibility for coordination decisions can be
    distributed among agents in any of a variety of ways, as expressed in a
    meta-level organization (MLO).  The MLO defines roles, protocols, and
    authority structures that the agents use when solving the coordination
    problem ... the MLO can distribute coordination responsibility such that
    each agent has authority to modify its own plans based on its current
    partial view of the plans of (some of) the other agents."

Two things are settled here, both needed by the repair layer:

* **Authority.**  A total order over agents.  A repair episode orders its
  participants by it, which is Algorithm 11.2 step 5(a): "agents form a total
  order.  Top agent is the current superior."  Inferiors adapt to superiors and
  double-check against previous superiors.
* **Roles.**  Which agent mediates an episode, and which agent is answerable for
  each zone of the floor.

Authority is inherited from the initial planning priority, so the agent that
picked its route first also speaks first when the route has to change.  That
keeps the initial plan and its repairs consistent with one another instead of
imposing two unrelated orderings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .grid import WarehouseGrid


@dataclass
class MetaLevelOrganization:
    """Roles, authority and zone responsibility for the fleet."""

    grid: WarehouseGrid
    authority: dict[int, int] = field(default_factory=dict)
    zone_coordinators: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_priority(
        cls, grid: WarehouseGrid, priority_order: Sequence[int]
    ) -> "MetaLevelOrganization":
        """Build an MLO whose authority ranking is the planning priority order.

        Rank 0 is the most senior agent -- the one that planned first and whose
        route the others had to accommodate.
        """
        authority = {agent: rank for rank, agent in enumerate(priority_order)}
        return cls(grid=grid, authority=authority)

    # ---------------------------------------------------------------- authority

    def rank(self, agent: int) -> int:
        """Authority rank; lower is more senior.  Unknown agents sort last."""
        return self.authority.get(agent, len(self.authority) + agent)

    def total_order(self, agents: Iterable[int]) -> tuple[int, ...]:
        """Order ``agents`` most senior first (Algorithm 11.2 step 5a)."""
        return tuple(sorted(agents, key=self.rank))

    def superior(self, agents: Iterable[int]) -> int:
        """The most senior of ``agents`` -- the current superior."""
        return min(agents, key=self.rank)

    def is_senior(self, a: int, b: int) -> bool:
        return self.rank(a) < self.rank(b)

    # -------------------------------------------------------------------- roles

    def mediator(self, involved: Iterable[int], disrupted: int | None = None) -> int:
        """Which agent runs a repair episode.

        The disrupted agent mediates its own episode when it is a participant --
        it is the one that detected the problem and holds the most information
        about it.  Otherwise the most senior participant takes over.
        """
        involved = list(involved)
        if disrupted is not None and disrupted in involved:
            return disrupted
        return self.superior(involved)

    def assign_zone_coordinators(self, plans: Mapping[int, object]) -> None:
        """Make the most senior agent routed through each zone answerable for it.

        Zone coordinators are what let a disruption be announced to a bounded
        set of agents rather than broadcast to the fleet.
        """
        from .abstractplan import AbstractPlan  # local import: avoids a cycle

        self.zone_coordinators.clear()
        for agent, abstract in plans.items():
            if not isinstance(abstract, AbstractPlan):
                continue
            for interval in abstract.intervals:
                current = self.zone_coordinators.get(interval.zone)
                if current is None or self.is_senior(agent, current):
                    self.zone_coordinators[interval.zone] = agent

    def coordinator_of(self, cell: int) -> int | None:
        """The agent answerable for the zone containing ``cell``."""
        return self.zone_coordinators.get(self.grid.zone_of(cell))

    def __len__(self) -> int:
        return len(self.authority)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"MetaLevelOrganization(agents={len(self.authority)}, "
            f"zones_covered={len(self.zone_coordinators)})"
        )
