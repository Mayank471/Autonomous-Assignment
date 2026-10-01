"""Dynamic obstacles: cells that become permanently impassable during execution.

A blockage is *permanent from a known time* rather than transient.  That is the
harder case for plan repair -- an agent cannot simply wait it out -- and it is
what the assignment's "sudden blockage of some grid cell" describes.  A broken
down agent is modelled the same way: its cell blocks from the moment it fails.
"""

from __future__ import annotations

from typing import Iterator

INF = float("inf")


class DynamicObstacles:
    """Cells blocked from a given time onward, with a version stamp.

    The version increments on every change so that caches keyed on the obstacle
    set (notably :class:`~warehouse.stastar.HeuristicCache`) can tell when they
    have gone stale.
    """

    __slots__ = ("_from", "_version")

    def __init__(self) -> None:
        self._from: dict[int, int] = {}
        self._version: int = 0

    @property
    def version(self) -> int:
        return self._version

    def __len__(self) -> int:
        return len(self._from)

    def __iter__(self) -> Iterator[int]:
        return iter(self._from)

    def __contains__(self, cell: int) -> bool:
        return cell in self._from

    def block(self, cell: int, from_time: int) -> bool:
        """Block ``cell`` from ``from_time``.

        Returns ``True`` if this tightened the constraint (new cell, or an
        earlier start time than previously recorded).
        """
        prev = self._from.get(cell)
        if prev is not None and prev <= from_time:
            return False
        self._from[cell] = from_time
        self._version += 1
        return True

    def unblock(self, cell: int) -> bool:
        """Remove the blockage on ``cell``; returns whether anything changed."""
        if self._from.pop(cell, None) is None:
            return False
        self._version += 1
        return True

    def blocked_from(self, cell: int) -> float:
        """Time from which ``cell`` is impassable, or ``inf`` if never."""
        return self._from.get(cell, INF)

    def is_blocked(self, cell: int, t: int) -> bool:
        """Whether ``cell`` is impassable at time ``t``."""
        start = self._from.get(cell)
        return start is not None and t >= start

    def blocked_by(self, t: int) -> set[int]:
        """Cells already blocked at time ``t``."""
        return {cell for cell, start in self._from.items() if start <= t}

    def all_cells(self) -> set[int]:
        """Every cell that is or will become blocked."""
        return set(self._from)

    def items(self):
        return self._from.items()

    def copy(self) -> "DynamicObstacles":
        other = DynamicObstacles()
        other._from = dict(self._from)
        other._version = self._version
        return other

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"DynamicObstacles({len(self._from)} cells)"
