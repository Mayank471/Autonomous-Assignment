"""Social laws -- coordination *prior* to local planning (Weiss Ch 11 section 3.1).

The textbook's motivation: "the purpose of social laws is to relieve agents of
the burden of explicitly coordinating".  A convention fixed once at design time
removes a whole class of interactions before any planning happens, and costs
nothing at runtime.

The convention used here is the one warehouses actually use: alternating
one-way aisles.  Even-numbered aisle rows run east, odd-numbered rows run west.
Vertical movement, waiting, and the perimeter highway stay unrestricted, which
is what keeps the floor strongly connected -- a social law that disconnects the
graph would trade conflicts for infeasibility.  :func:`check_connectivity`
asserts that it does not.
"""

from __future__ import annotations

from collections import deque

from .grid import WarehouseGrid


class SocialLaws:
    """A movement convention, queried by the low-level search as a move filter."""

    __slots__ = ("_grid", "enabled", "_margin", "_oneway_east")

    def __init__(
        self, grid: WarehouseGrid, enabled: bool = True, margin: int = 2
    ) -> None:
        self._grid = grid
        self.enabled = enabled
        self._margin = margin
        self._oneway_east = self._classify_rows()

    def _classify_rows(self) -> dict[int, bool]:
        """Map each restricted row to ``True`` (eastbound) or ``False`` (westbound)."""
        grid = self._grid
        out: dict[int, bool] = {}
        aisle_index = 0
        for r in range(self._margin, grid.n_rows - self._margin):
            # Only rows that are genuinely a through-aisle get a direction.
            interior = grid.passable[r, self._margin : grid.n_cols - self._margin]
            if interior.all():
                out[r] = aisle_index % 2 == 0
                aisle_index += 1
        return out

    def in_highway(self, cell: int) -> bool:
        """Whether ``cell`` is in the unrestricted perimeter ring."""
        r, c = self._grid.rc(cell)
        m = self._margin
        return (
            r < m
            or r >= self._grid.n_rows - m
            or c < m
            or c >= self._grid.n_cols - m
        )

    def allows(self, u: int, v: int) -> bool:
        """Whether moving ``u -> v`` respects the convention."""
        if not self.enabled or u == v:
            return True

        grid = self._grid
        ur, uc = grid.rc(u)
        vr, vc = grid.rc(v)
        if ur != vr:
            return True  # vertical moves are always permitted
        if self.in_highway(u) or self.in_highway(v):
            return True

        eastbound = self._oneway_east.get(ur)
        if eastbound is None:
            return True
        return (vc > uc) if eastbound else (vc < uc)

    @property
    def restricted_rows(self) -> tuple[int, ...]:
        return tuple(sorted(self._oneway_east))

    def as_filter(self):
        """Return a ``(u, v) -> bool`` callable, or ``None`` when disabled.

        ``None`` lets the search skip the check entirely rather than calling a
        function that always returns ``True``.
        """
        return self.allows if self.enabled else None


def check_connectivity(grid: WarehouseGrid, laws: SocialLaws) -> bool:
    """Whether every free cell can still reach every other under ``laws``.

    One-way rules make the movement graph *directed*, so this checks strong
    connectivity: a forward BFS from a seed cell must reach everything, and a
    backward BFS must too.
    """
    cells = [c for c in grid.free_cells]
    if not cells:
        return True
    seed = cells[0]
    total = len(cells)

    def reach(forward: bool) -> int:
        seen = {seed}
        queue = deque([seed])
        while queue:
            cell = queue.popleft()
            for nxt in grid.neighbors(cell):
                ok = laws.allows(cell, nxt) if forward else laws.allows(nxt, cell)
                if ok and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return len(seen)

    return reach(True) == total and reach(False) == total
