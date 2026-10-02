"""Dynamic obstacles as time windows on cells.

Two disruption types produce windows:

* a sudden **blockage** of cell ``c`` for ``d`` steps -> window ``[t, t+d)``
  with ``owner = -1``;
* a **breakdown** of agent ``b`` -> its cell is blocked while ``b`` is frozen,
  with ``owner = b`` (the broken agent itself is of course allowed to be there).

Durations are announced when the disruption starts, so planners can route
around a window or wait for it to clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObstacleWindows:
    windows: dict[int, list[tuple[int, int, int]]] = field(default_factory=dict)

    def add(self, cell: int, start: int, end: int, owner: int = -1) -> None:
        """Block ``cell`` for ``start <= t < end``."""
        self.windows.setdefault(cell, []).append((start, end, owner))

    def blocked(self, cell: int, t: int, agent: int = -2) -> bool:
        ws = self.windows.get(cell)
        if not ws:
            return False
        for s, e, owner in ws:
            if s <= t < e and owner != agent:
                return True
        return False

    def last_time(self) -> int:
        """Latest time at which anything is blocked (0 if nothing)."""
        return max((e - 1 for ws in self.windows.values() for _, e, _ in ws), default=0)

    def blocked_after(self, cell: int, t: int, agent: int = -2) -> bool:
        """Is ``cell`` blocked at any time strictly after ``t``?"""
        return any(e - 1 > t and owner != agent for _, e, owner in self.windows.get(cell, ()))

    def prune(self, t_now: int) -> None:
        for cell in list(self.windows):
            kept = [w for w in self.windows[cell] if w[1] > t_now]
            if kept:
                self.windows[cell] = kept
            else:
                del self.windows[cell]

    def active_count(self, t: int, owner_filter: int | None = -1) -> int:
        """Number of cells blocked at time ``t`` (by blockages only, by default)."""
        n = 0
        for ws in self.windows.values():
            if any(s <= t < e and (owner_filter is None or o == owner_filter) for s, e, o in ws):
                n += 1
        return n

    def items(self):
        for cell, ws in self.windows.items():
            for s, e, o in ws:
                yield cell, s, e, o

    def copy(self) -> "ObstacleWindows":
        return ObstacleWindows({c: list(ws) for c, ws in self.windows.items()})
