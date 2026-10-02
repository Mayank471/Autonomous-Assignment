"""Warehouse floor: a 4-connected grid with shelves, aisles, docks and stations.

Cells are plain ints, ``cell = row * width + col``, so they can be used as dict
keys and packed into larger integer keys cheaply.

Layout produced by :func:`make_warehouse` (``main`` preset shown schematically)::

    row 0      D D D D D D D D D D D      docks: private dead-end pockets
    row 1    . . . . . . . . . . . . .    ring corridor
    row 2  S . . p p p p . p p p p . . S  aisle (p = pickup point next to a shelf)
    row 3  # . . # # # # . # # # # . . #  shelf row, broken by cross-aisles
    ...
    row H-2  . . . . . . . . . . . . .    ring corridor
    row H-1    D D D D D D D D D D D      docks

``S`` cells are delivery stations: dead-end pockets on the left and right edges,
shared by every agent.  ``D`` cells are docks: each agent owns one and no other
agent may ever enter it.  Robots only park in docks, so a parked robot never
blocks anyone -- the layout is *well-formed* in the sense of Ma et al. (2017),
which is what lets prioritized planning always find a solution.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

UNREACHABLE = 1 << 30


@dataclass
class WarehouseGrid:
    width: int
    height: int
    free: list[bool]
    docks: list[int]
    stations: list[int]
    pickups: list[int]
    name: str = "custom"
    neighbors: list[tuple[int, ...]] = field(init=False, repr=False)
    dock_set: frozenset[int] = field(init=False, repr=False)
    station_set: frozenset[int] = field(init=False, repr=False)
    blockable: list[int] = field(init=False, repr=False)
    _dist_cache: dict[int, list[int]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        w, h = self.width, self.height
        self.dock_set = frozenset(self.docks)
        self.station_set = frozenset(self.stations)
        pockets = self.dock_set | self.station_set
        nbrs: list[tuple[int, ...]] = []
        for cell in range(w * h):
            if not self.free[cell]:
                nbrs.append(())
                continue
            r, c = divmod(cell, w)
            out = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and self.free[rr * w + cc]:
                    # Pockets (docks, stations) connect only to ordinary cells,
                    # never to each other, so they are true dead ends even
                    # when laid out side by side.
                    if cell in pockets and rr * w + cc in pockets:
                        continue
                    out.append(rr * w + cc)
            nbrs.append(tuple(out))
        self.neighbors = nbrs
        # Cells a sudden blockage may land on: anything traversable except the
        # pockets (docks and stations).
        self.blockable = [
            c for c in range(w * h)
            if self.free[c] and c not in self.dock_set and c not in self.station_set
        ]

    # ------------------------------------------------------------------ basics
    @property
    def n_cells(self) -> int:
        return self.width * self.height

    def rc(self, cell: int) -> tuple[int, int]:
        return divmod(cell, self.width)

    def cell(self, row: int, col: int) -> int:
        return row * self.width + col

    def manhattan(self, a: int, b: int) -> int:
        ar, ac = divmod(a, self.width)
        br, bc = divmod(b, self.width)
        return abs(ar - br) + abs(ac - bc)

    # ---------------------------------------------------------------- distances
    def dist_from(self, target: int) -> list[int]:
        """Exact static shortest-path distance from every cell to ``target``.

        Docks and stations are dead-end pockets, so no shortest path ever passes
        *through* one; plain BFS on the free cells is therefore exact even though
        other agents' docks are forbidden during planning.
        """
        d = self._dist_cache.get(target)
        if d is not None:
            return d
        d = [UNREACHABLE] * self.n_cells
        d[target] = 0
        q = deque([target])
        nbrs = self.neighbors
        while q:
            u = q.popleft()
            du = d[u] + 1
            for v in nbrs[u]:
                if d[v] == UNREACHABLE:
                    d[v] = du
                    q.append(v)
        self._dist_cache[target] = d
        return d

    def dist(self, a: int, b: int) -> int:
        return self.dist_from(b)[a]

    def tour_length(self, start: int, goals: list[int] | tuple[int, ...]) -> int:
        total, cur = 0, start
        for g in goals:
            total += self.dist(cur, g)
            cur = g
        return total

    def render(self, marks: dict[int, str] | None = None) -> str:
        marks = marks or {}
        rows = []
        for r in range(self.height):
            line = []
            for c in range(self.width):
                cell = r * self.width + c
                if cell in marks:
                    line.append(marks[cell])
                elif not self.free[cell]:
                    line.append("#")
                elif cell in self.dock_set:
                    line.append("D")
                elif cell in self.station_set:
                    line.append("S")
                else:
                    line.append(".")
            rows.append("".join(line))
        return "\n".join(rows)


PRESETS = {
    # n_blocks_x, block_len, n_shelf_rows
    "small": (2, 4, 3),   # 11 x 15
    "main": (4, 6, 8),    # 21 x 33
}


def make_warehouse(preset: str = "main", *, n_blocks_x: int | None = None,
                   block_len: int | None = None, n_shelf_rows: int | None = None) -> WarehouseGrid:
    """Build a Kiva-style warehouse (see module docstring)."""
    bx, bl, nr = PRESETS[preset] if preset in PRESETS else (4, 6, 8)
    bx = n_blocks_x or bx
    bl = block_len or bl
    nr = n_shelf_rows or nr

    interior_w = bx * (bl + 1) + 1
    interior_h = 2 * nr + 1
    w = interior_w + 4
    h = interior_h + 4
    free = [False] * (w * h)

    def setf(r: int, c: int) -> None:
        free[r * w + c] = True

    # ring corridor
    for c in range(1, w - 1):
        setf(1, c)
        setf(h - 2, c)
    for r in range(1, h - 1):
        setf(r, 1)
        setf(r, w - 2)
    # interior: aisle rows everywhere, shelf rows only on cross-aisle columns
    shelves: set[tuple[int, int]] = set()
    for ir in range(interior_h):
        r = ir + 2
        for ic in range(interior_w):
            c = ic + 2
            is_shelf_row = ir % 2 == 1
            is_cross = ic % (bl + 1) == 0
            if is_shelf_row and not is_cross:
                shelves.add((r, c))
            else:
                setf(r, c)
    # docks on top/bottom rows, above/below the ring (excluding corners)
    docks = []
    for c in range(2, w - 2):
        for r in (0, h - 1):
            setf(r, c)
            docks.append(r * w + c)
    docks.sort()
    # stations on the left/right edges, level with interior aisle rows
    stations = []
    for ir in range(0, interior_h, 2):
        r = ir + 2
        for c in (0, w - 1):
            setf(r, c)
            stations.append(r * w + c)
    stations.sort()
    # pickups: aisle cells vertically adjacent to a shelf
    pickups = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if free[r * w + c] and ((r - 1, c) in shelves or (r + 1, c) in shelves):
                pickups.append(r * w + c)
    return WarehouseGrid(width=w, height=h, free=free, docks=docks, stations=stations,
                         pickups=pickups, name=preset)


def grid_from_ascii(text: str, name: str = "ascii") -> WarehouseGrid:
    """Build a grid from an ASCII map (tests): ``#`` wall, ``D`` dock,
    ``S`` station, ``p`` pickup, anything else free."""
    lines = [ln for ln in text.strip("\n").splitlines()]
    h, w = len(lines), max(len(ln) for ln in lines)
    free = [False] * (w * h)
    docks, stations, pickups = [], [], []
    for r, ln in enumerate(lines):
        for c in range(w):
            ch = ln[c] if c < len(ln) else "#"
            cell = r * w + c
            if ch == "#":
                continue
            free[cell] = True
            if ch == "D":
                docks.append(cell)
            elif ch == "S":
                stations.append(cell)
            elif ch == "p":
                pickups.append(cell)
    return WarehouseGrid(width=w, height=h, free=free, docks=docks, stations=stations,
                         pickups=pickups, name=name)
