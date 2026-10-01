"""Warehouse floor: a 2D grid of cells with shelves, aisles and stations.

Cells are addressed by integer id (``r * n_cols + c``) rather than ``(r, c)``
tuples.  The planner touches millions of cells across the experiment sweep and
integer keys keep the Space-Time A* inner loop cheap; :meth:`WarehouseGrid.rc`
and :meth:`WarehouseGrid.cell` convert when readability matters.

A ``WarehouseGrid`` is immutable once built, so one instance is safely shared
across every run that uses the same layout.  Dynamic obstacles live separately
in :class:`~warehouse.disruptions.DynamicObstacles`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

# Movement deltas in (row, col).  Order matters only for tie-breaking.
_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Directions usable as a social-law annotation (see :mod:`warehouse.sociallaws`).
NORTH, SOUTH, WEST, EAST = _DELTAS


@dataclass(frozen=True)
class WarehouseGrid:
    """A static warehouse floor plan.

    Attributes:
        n_rows, n_cols: floor dimensions in cells.
        passable: boolean array, ``True`` where a robot may stand.
        pickup_points: cell ids where objects can be collected (aisle cells
            adjacent to a shelf block).
        delivery_points: cell ids of the workstations objects are carried to.
        dock_points: cell ids of the docking bays agents start and finish on.
        zone_shape: ``(zone_rows, zone_cols)`` size of one organizational zone.
    """

    n_rows: int
    n_cols: int
    passable: np.ndarray
    pickup_points: tuple[int, ...]
    delivery_points: tuple[int, ...]
    dock_points: tuple[int, ...] = ()
    zone_shape: tuple[int, int] = (5, 6)

    # Derived, filled in __post_init__.  Excluded from eq/repr.
    _neighbors: tuple[tuple[int, ...], ...] = field(
        default=(), repr=False, compare=False
    )
    _free_cells: tuple[int, ...] = field(default=(), repr=False, compare=False)
    _zone_of: tuple[int, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_neighbors", self._build_neighbors())
        object.__setattr__(
            self,
            "_free_cells",
            tuple(int(i) for i in np.flatnonzero(self.passable.ravel())),
        )
        object.__setattr__(self, "_zone_of", self._build_zones())

    # ---------------------------------------------------------------- indexing

    def cell(self, r: int, c: int) -> int:
        """Cell id for row ``r``, column ``c``."""
        return r * self.n_cols + c

    def rc(self, cell: int) -> tuple[int, int]:
        """``(row, col)`` for a cell id."""
        return divmod(cell, self.n_cols)

    @property
    def n_cells(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def free_cells(self) -> tuple[int, ...]:
        """Every statically passable cell id."""
        return self._free_cells

    def is_free(self, cell: int) -> bool:
        """Whether ``cell`` is statically passable (ignores dynamic obstacles)."""
        r, c = self.rc(cell)
        return bool(self.passable[r, c])

    def neighbors(self, cell: int) -> tuple[int, ...]:
        """Passable 4-connected neighbours of ``cell``, precomputed."""
        return self._neighbors[cell]

    def manhattan(self, a: int, b: int) -> int:
        ar, ac = self.rc(a)
        br, bc = self.rc(b)
        return abs(ar - br) + abs(ac - bc)

    # ------------------------------------------------------------ organization

    def zone_of(self, cell: int) -> int:
        """Zone id containing ``cell`` (Ch 11 section 3.2 organizational structuring)."""
        return self._zone_of[cell]

    @property
    def n_zones(self) -> int:
        zr, zc = self.zone_shape
        return ((self.n_rows + zr - 1) // zr) * ((self.n_cols + zc - 1) // zc)

    # ------------------------------------------------------------ construction

    def _build_neighbors(self) -> tuple[tuple[int, ...], ...]:
        out: list[tuple[int, ...]] = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if not self.passable[r, c]:
                    out.append(())
                    continue
                adj = []
                for dr, dc in _DELTAS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
                        if self.passable[nr, nc]:
                            adj.append(nr * self.n_cols + nc)
                out.append(tuple(adj))
        return tuple(out)

    def _build_zones(self) -> tuple[int, ...]:
        zr, zc = self.zone_shape
        per_row = (self.n_cols + zc - 1) // zc
        return tuple(
            (r // zr) * per_row + (c // zc)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
        )

    # --------------------------------------------------------------- factories

    @classmethod
    def kiva(
        cls,
        n_rows: int = 50,
        n_cols: int = 60,
        shelf_h: int = 2,
        shelf_w: int = 5,
        aisle: int = 1,
        highway: int = 2,
        n_delivery: int = 12,
        zone_shape: tuple[int, int] = (5, 6),
    ) -> "WarehouseGrid":
        """Build a Kiva-style warehouse: shelf blocks in a field of aisles.

        A ``highway``-wide ring of open floor surrounds a field of
        ``shelf_h x shelf_w`` shelf blocks separated by ``aisle``-wide gaps.

        * **Pickup points** are the aisle cells touching a shelf.
        * **Docking bays** are the outermost ring cells.  Agents start and
          finish there, and a dock belongs to its agent for the whole run, so
          it is a wall to everyone else.  Putting them on the outer wall means
          that can never disconnect the floor: the *inner* highway row remains
          open and still joins every aisle.
        * **Delivery stations** sit on the inner highway ring, so they stay
          traversable and never collide with a dock.
        """
        passable = np.ones((n_rows, n_cols), dtype=bool)

        top, left = highway, highway
        bottom, right = n_rows - highway, n_cols - highway
        r = top
        while r + shelf_h <= bottom:
            c = left
            while c + shelf_w <= right:
                passable[r : r + shelf_h, c : c + shelf_w] = False
                c += shelf_w + aisle
            r += shelf_h + aisle

        pickups = cls._aisle_cells_touching_shelves(passable, n_rows, n_cols)

        # Docking bays: the outermost ring.
        docks: list[int] = []
        for c in range(n_cols):
            docks.append(c)
            docks.append((n_rows - 1) * n_cols + c)
        for r in range(1, n_rows - 1):
            docks.append(r * n_cols)
            docks.append(r * n_cols + n_cols - 1)
        dock_set = set(docks)

        # Delivery stations: evenly spaced along the inner highway ring.
        deliveries: list[int] = []
        inner_left, inner_right = 1, n_cols - 2
        inner_top, inner_bottom = 1, n_rows - 2
        half = max(1, n_delivery // 2)
        for rr in np.linspace(inner_top + 1, inner_bottom - 1, half).astype(int):
            deliveries.append(int(rr) * n_cols + inner_left)
        for cc in np.linspace(inner_left + 1, inner_right - 1, n_delivery - half).astype(int):
            deliveries.append(inner_bottom * n_cols + int(cc))

        flat = passable.ravel()
        seen: set[int] = set()
        delivery_points: list[int] = []
        for d in deliveries:
            if d not in seen and d not in dock_set and flat[d]:
                seen.add(d)
                delivery_points.append(d)

        return cls(
            n_rows=n_rows,
            n_cols=n_cols,
            passable=passable,
            pickup_points=tuple(c for c in pickups if c not in dock_set),
            delivery_points=tuple(delivery_points),
            dock_points=tuple(d for d in docks if flat[d]),
            zone_shape=zone_shape,
        )

    @classmethod
    def from_ascii(cls, art: str, zone_shape: tuple[int, int] = (5, 6)) -> "WarehouseGrid":
        """Build a grid from ASCII art -- ``#`` shelf, ``p`` pickup, ``d`` delivery.

        Used by the tests and the small demo scenarios, where a hand-drawn
        layout is clearer than a generated one.
        """
        lines = [ln for ln in art.strip("\n").splitlines() if ln.strip()]
        n_rows = len(lines)
        n_cols = max(len(ln) for ln in lines)
        passable = np.ones((n_rows, n_cols), dtype=bool)
        pickups: list[int] = []
        deliveries: list[int] = []
        for r, line in enumerate(lines):
            for c in range(n_cols):
                ch = line[c] if c < len(line) else "."
                if ch == "#":
                    passable[r, c] = False
                elif ch == "p":
                    pickups.append(r * n_cols + c)
                elif ch == "d":
                    deliveries.append(r * n_cols + c)
        return cls(
            n_rows=n_rows,
            n_cols=n_cols,
            passable=passable,
            pickup_points=tuple(pickups),
            delivery_points=tuple(deliveries),
            zone_shape=zone_shape,
        )

    @staticmethod
    def _aisle_cells_touching_shelves(
        passable: np.ndarray, n_rows: int, n_cols: int
    ) -> list[int]:
        """Free cells with at least one shelf neighbour -- the shelf access points."""
        out: list[int] = []
        for r in range(n_rows):
            for c in range(n_cols):
                if not passable[r, c]:
                    continue
                for dr, dc in _DELTAS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols and not passable[nr, nc]:
                        out.append(r * n_cols + c)
                        break
        return out

    # ------------------------------------------------------------------ pretty

    def render(
        self,
        occupied: dict[int, str] | None = None,
        blocked: Iterable[int] = (),
    ) -> str:
        """Render the floor as text -- used by the terminal demo and for debugging."""
        occupied = occupied or {}
        blocked = set(blocked)
        rows = []
        for r in range(self.n_rows):
            chars = []
            for c in range(self.n_cols):
                cell = r * self.n_cols + c
                if cell in occupied:
                    chars.append(occupied[cell])
                elif cell in blocked:
                    chars.append("X")
                elif not self.passable[r, c]:
                    chars.append("#")
                elif cell in self.delivery_points:
                    chars.append("d")
                else:
                    chars.append(".")
            rows.append("".join(chars))
        return "\n".join(rows)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"WarehouseGrid({self.n_rows}x{self.n_cols}, "
            f"free={len(self._free_cells)}, "
            f"pickups={len(self.pickup_points)}, "
            f"deliveries={len(self.delivery_points)}, "
            f"zones={self.n_zones})"
        )
