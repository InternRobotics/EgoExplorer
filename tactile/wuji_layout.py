"""Wuji glove 24x31 tactile matrix -> anatomical region layout (v2).

Derived directly from the per-taxel validity mask of the recordings (the union
of taxels that ever report a value >= 0), which draws a clear hand silhouette:

    col  0- 4  pinky          rows  5- 9
    col  5-10  ring           rows  1- 9      (palm continues below, rows 10-23)
    col 11-12  inter-finger gap (invalid above row 10)
    col 13-18  middle         rows  0- 9
    col 19-20  inter-finger gap (invalid above row 10)
    col 21-25  index          rows  1- 9
    col 26-30  thumb          rows  9-18
    rows 10-23 palm: ulnar (cols 0-10) / central (11-20) / radial-thenar (21-30)

Zones are rectangles over the full grid; the physical taxel set is obtained by
intersecting with the per-glove validity mask, so a zone's taxel count is a
per-recording quantity, not a constant.

Matrix convention (both gloves, as stored): row 0 = fingertips, row 23 = wrist;
col 0 = pinky side, col 30 = thumb side.  For a palmar view the left hand is
drawn as stored and the right hand mirrored horizontally.
"""
from __future__ import annotations

ROWS, COLS = 24, 31
NCH = ROWS * COLS

# (region, row_start, row_end_inclusive, col_start, col_end_inclusive)
# Earlier entries win where rectangles overlap.
ZONES = [
    ("thumb_distal",      9, 13, 26, 30),
    ("thumb_proximal",   14, 18, 26, 30),
    ("index_distal",      0,  3, 21, 25),
    ("index_middle",      4,  6, 21, 25),
    ("index_proximal",    7,  9, 21, 25),
    ("middle_distal",     0,  3, 13, 18),
    ("middle_middle",     4,  6, 13, 18),
    ("middle_proximal",   7,  9, 13, 18),
    ("ring_distal",       0,  3,  5, 10),
    ("ring_middle",       4,  6,  5, 10),
    ("ring_proximal",     7,  9,  5, 10),
    ("pinky_distal",      0,  6,  0,  4),
    ("pinky_middle",      7,  8,  0,  4),
    ("pinky_proximal",    9,  9,  0,  4),
    ("palm_radial",      10, 23, 21, 30),
    ("palm_central",     10, 23, 11, 20),
    ("palm_ulnar",       10, 23,  0, 10),
]

REGION_NAMES = [z[0] for z in ZONES]
FINGER_OF = {n: n.split("_")[0] for n in REGION_NAMES}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky", "palm"]
FINGER_COLORS = {
    "thumb": "#ff6b6b", "index": "#ffd166", "middle": "#06d6a0",
    "ring": "#4cc9f0", "pinky": "#b085f5", "palm": "#f0b49a",
}
# coarse 6-way grouping (the granularity the paper reports)
COARSE = ["thumb", "index", "middle", "ring", "pinky", "palm"]


def region_grid():
    grid = [[None] * COLS for _ in range(ROWS)]
    for name, r0, r1, c0, c1 in ZONES:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if grid[r][c] is None:
                    grid[r][c] = name
    return grid


def channel_regions():
    g = region_grid()
    return [g[r][c] for r in range(ROWS) for c in range(COLS)]


def region_index():
    """Flat 744 int array: index into REGION_NAMES, -1 if unassigned."""
    idx = {n: i for i, n in enumerate(REGION_NAMES)}
    return [idx[r] if r else -1 for r in channel_regions()]
