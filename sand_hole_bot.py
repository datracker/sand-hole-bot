#!/usr/bin/env python3
"""
Sand Hole bot, played through macOS iPhone Mirroring.

The puzzle is deterministic: a drag slides one tray along a free path (turning corners if need be).
Each half of a bucket pours into the tray right under it while the bottom of that half is the
tray's color, and a full tray vanishes. So the bot reads the board and the buckets' stripes once
and solves the whole level offline (the clock only starts with the first drag), keeping both halves
of a bucket draining evenly. The sand pictures aren't to scale, so how much sand there is comes
from the trays, never from the picture. Then it performs the drags. Like a finger pulling a tray,
it watches the screen as it drags, and turns a corner or lets go only once the tray is there. It
looks after each drag too: a tray that stopped short gets dragged on from where it is. A delivered
tray that stops filling slides under the bucket's other half if that half shows its color.

Setup:
  pip install pyautogui numpy pillow pyobjc-framework-Quartz
  System Settings > Privacy & Security: give your terminal Accessibility
  AND Screen Recording permission.

Usage:
  python sand_hole_bot.py snap              # saves window.png + grid_check.png -> tune CALIBRATION
  python sand_hole_bot.py read              # prints what the bot sees  (add -v for raw RGB values)
  python sand_hole_bot.py solve [img.png]   # solves the level on screen or in a screenshot and
                                            # saves it to solution.pkl and moves.txt (-v lists it)
  python sand_hole_bot.py follow [moves]    # plays moves.txt (or the file named) exactly as written
  python sand_hole_bot.py play [--blind]    # plays the saved solution if it fits the board, else
                                            # solves first, and solves again if the game differs.
                                            # --blind only looks after deliveries.
                                            # Abort: shove the mouse into a screen corner.
"""
import copy, heapq, itertools, math, os, pickle, re, subprocess, sys, tempfile, time
from collections import deque, namedtuple

import numpy as np
import pyautogui
import Quartz
from PIL import Image, ImageDraw

pyautogui.FAILSAFE = True

# ======================= CALIBRATION =======================
# All numbers are pixels in window.png (run `snap`, open it, measure). Screenshots of a bigger or
# smaller window are scaled to REF_W wide and shifted so the rim under the buckets sits at RIM_Y.
REF_W = 708                     # width of window.png
RIM_Y = 433                     # middle of the white rim under the buckets
GRID_X, GRID_Y = 74, 436        # top-left corner of the cell grid
CELL_W, CELL_H = 70, 68         # size of one cell
COLS, ROWS = 8, 11

# Each bucket: where to sample its BOTTOM stripe (just above the white rim, so a nearly drained
# stripe still reads right), the y range its stripes span (bottom, top), and the cells (row, col)
# that count as "under" it.
BUCKETS = {
    "left":  {"sample": (144, 427), "stripes": (430, 308), "targets": [(0, 0), (0, 1)]},
    "right": {"sample": (564, 427), "stripes": (430, 308), "targets": [(0, 6), (0, 7)]},
}

# Calibrated from window.png (tray hollow + bucket stripe shades).
PALETTE = {
    "empty":    [(60, 65, 97), (68, 75, 110), (72, 79, 117)],
    "wall":     [(210, 230, 253), (160, 177, 232), (130, 141, 246), (131, 159, 231)],
    "brown":    [(85, 32, 30), (110, 54, 50)],
    "pink":     [(204, 121, 212), (219, 130, 226), (238, 141, 247), (240, 173, 240)],
    "orange":   [(188, 86, 34), (238, 126, 68)],
    "yellow":   [(223, 161, 58), (246, 202, 74)],
    "cyan":     [(60, 135, 206), (97, 193, 249)],
    "red":      [(164, 42, 34), (216, 68, 59)],
    "purple":   [(101, 37, 221), (124, 56, 245)],
    "blue":     [(38, 68, 186), (69, 116, 246)],
    "magenta":  [(174, 44, 169), (184, 47, 179), (235, 90, 217)],
    "lavender": [(131, 131, 234), (164, 163, 248)],
}
NON_PIECE = {"empty", "wall"}
LABEL_MIN_WHITE = 20     # white pixels needed to count as a "0%" label

# ======================= TIMING =======================
# The tray trails the finger and snaps to the nearest cell when let go, so the finger lets go a
# third of a cell past the target: a tray that is still behind then snaps forward, not back.
# The tray also lags by however long iPhone Mirroring takes to pass the finger on, so the finger
# turns a corner or lets go only once the screen shows the tray there; a faster finger just leaves
# it further behind.
STEP_PX = 5          # drag interpolation step (screen points)
STEP_DELAY = 0.008   # seconds between drag steps (~600 points/s)
MIN_LEG = 0.15       # a straight stretch never takes less than this; shorter flicks get ignored
PRESS_HOLD = 0.12    # hold after pressing so the game picks the tray up before it moves
CORNER_PAUSE = 0.15  # let the tray reach the corner before the finger turns (when not watching)
RELEASE_HOLD = 0.08  # hold at the end before letting go (when not watching)
CORNER_WAIT = 0.6    # longest the finger waits at a corner for the screen to show the tray there
END_WAIT = 0.6       # longest it waits at the end for the tray before letting go anyway
CORNER_PUSH = 0.3    # how far past a corner (in cells) the finger goes when the tray can't go on
EDGE = 0.35          # how far from a cell's middle (in cells) to look for a tray clearing its edge
LIP = 0.25           # how far (in cells) a tray's front face hangs down over the cell below
OVERSHOOT = 0.34     # how far past the target's center the finger lets go, in cells
SETTLE = 0.12        # wait after letting go before looking at the result
RELOOK = 0.15        # wait between further looks while a tray may still be snapping into place
GAP = 0.15           # pause after a drag done without looking (--blind)
POLL = 0.25          # pause between looks while waiting for trays to vanish
FILL_WAIT = 20.0     # seconds to wait for the board to match before solving again
POUR_WAIT = 15.0     # follow: longest wait for a pouring tray that is in the next move's way
CLOCK = 180          # the level's countdown, which starts with the first drag
PLAN_TIME = 5.0      # seconds of search per delivery (6x that if it finds nothing)

SAVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.pkl")
MOVES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moves.txt")
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ======================= SCREEN =======================
_rect = None


def window_rect(refresh=False):
    """Window position/size. Asking System Events costs ~0.35 s, so it's cached."""
    global _rect
    if refresh or _rect is None:
        script = ('tell application "System Events" to tell process "iPhone Mirroring" '
                  'to get {position, size} of window 1')
        out = subprocess.check_output(["osascript", "-e", script]).decode()
        _rect = tuple(int(v) for v in out.replace(" ", "").split(","))
    return _rect


def grab(x, y, w, h):
    """RGB pixels of a screen rectangle straight from Quartz (~0.04 s), or None if unavailable."""
    try:
        cg = Quartz.CGWindowListCreateImage(Quartz.CGRectMake(x, y, w, h),
                                            Quartz.kCGWindowListOptionOnScreenOnly,
                                            Quartz.kCGNullWindowID, Quartz.kCGWindowImageBestResolution)
    except AttributeError:
        return None
    if cg is None:
        return None
    width, height = Quartz.CGImageGetWidth(cg), Quartz.CGImageGetHeight(cg)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(cg))
    px = np.frombuffer(data, np.uint8).reshape(height, Quartz.CGImageGetBytesPerRow(cg) // 4, 4)
    return px[:, :width, [2, 1, 0]].copy()          # BGRA -> RGB


_offset = {}   # window size -> rows the board sits below where window.png has it


def find_offset(img):
    """Resizing the window scales the picture but not quite its top margin, so the board can sit
    a little higher or lower; the white rim under the buckets gives the shift away."""
    x = BUCKETS["left"]["sample"][0]
    strip = img[RIM_Y - 40:RIM_Y + 41, x - 20:x + 21].astype(int)
    rows = [RIM_Y - 40 + i for i, v in enumerate((strip.min(axis=2) > 240).mean(axis=1)) if v > 0.9]
    return round(sum(rows) / len(rows)) - RIM_Y if rows else 0


def capture(refresh=False):
    """Screenshot of the window, scaled and shifted to line up with window.png, plus what
    to_screen() needs to turn its pixels back into screen points."""
    x, y, w, h = window_rect(refresh)
    img = grab(x, y, w, h)
    if img is None:
        path = os.path.join(tempfile.gettempdir(), "sand_hole_bot_cap.png")
        subprocess.run(["screencapture", "-x", "-R", f"{x},{y},{w},{h}", path], check=True)
        img = np.array(Image.open(path).convert("RGB"))
    if img.shape[1] != REF_W:
        img = np.array(Image.fromarray(img).resize((REF_W, round(img.shape[0] * REF_W / img.shape[1])),
                                                   Image.LANCZOS))
    if (w, h) not in _offset:
        _offset[(w, h)] = find_offset(img)
    dy = _offset[(w, h)]
    if dy:
        img = np.roll(img, -dy, axis=0)
    scale = REF_W / w                 # pixels per screen point
    return img, (x, y + dy / scale, scale)


def to_screen(px, py, win):
    x, y, scale = win
    return x + px / scale, y + py / scale


def _mouse(kind, x, y):
    e = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, e)


waits = []   # how long the finger waited for the tray at each stop it reached in time


def hold(pt, check, limit):
    """Keep the finger down at pt until check() passes or `limit` seconds go by, nudging it a
    point back and forth meanwhile, as a real finger is never quite still. Returns whether it
    passed."""
    t0 = time.time()
    k = 0
    while not check():
        if time.time() > t0 + limit:
            return False
        k += 1
        _mouse(Quartz.kCGEventLeftMouseDragged, pt[0] + k % 2, pt[1])
    waits.append(time.time() - t0)
    return True


def drag(points, checks=None):
    """Press at points[0], drag smoothly through the rest, release. Real drag events. With
    `checks` (one per straight stretch), the finger waits at the end of each stretch until its
    check passes, as a finger waits for the tray it pulls, and lets go early if the tray doesn't
    come; without, it waits fixed times. Returns False if a check never passed."""
    _mouse(Quartz.kCGEventMouseMoved, *points[0]); time.sleep(0.03)
    _mouse(Quartz.kCGEventLeftMouseDown, *points[0]); time.sleep(PRESS_HOLD)
    ok, here = True, points[0]
    for idx, ((x0, y0), (x1, y1)) in enumerate(zip(points, points[1:])):
        if idx > 0 and not checks:
            time.sleep(CORNER_PAUSE)
        steps = max(2, int(math.hypot(x1 - x0, y1 - y0) / STEP_PX))
        delay = max(STEP_DELAY, MIN_LEG / steps)
        for k in range(1, steps + 1):
            t = k / steps
            _mouse(Quartz.kCGEventLeftMouseDragged, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            time.sleep(delay)
        here = (x1, y1)
        if checks:
            ok = hold(here, checks[idx], END_WAIT if idx == len(points) - 2 else CORNER_WAIT)
            if not ok:
                break
    if not checks:
        time.sleep(RELEASE_HOLD)
    _mouse(Quartz.kCGEventLeftMouseUp, *here)
    return ok


# ======================= VISION =======================
_PAL = [(name, np.array(s, float)) for name, ss in PALETTE.items() for s in ss]


def classify(rgb):
    return min(_PAL, key=lambda p: float(((p[1] - rgb) ** 2).sum()))[0]


def cell_origin(r, c):
    return int(GRID_X + c * CELL_W), int(GRID_Y + r * CELL_H)


def cell_rgb(img, r, c):
    cx = int(GRID_X + (c + 0.5) * CELL_W)
    cy = int(GRID_Y + (r + 0.5) * CELL_H) + 6      # a bit low, away from the "0%" text
    return img[cy - 5:cy + 5, cx - 5:cx + 5].reshape(-1, 3).mean(axis=0)


class Look:
    """A quick screenshot (~20 ms) for watching a drag, read a cell at a time where cell_rgb()
    would, without the resizing and shifting capture() does."""

    def __init__(self, win):
        self.x, self.y, w, h = window_rect()
        self.px = grab(self.x, self.y, w, h)
        self.k = self.px.shape[1] / w if self.px is not None else 0   # pixels per screen point
        self.win = win

    def color(self, r, c, toward=(0, 0)):
        """The color where cell_rgb() reads, or, given a step, near that edge of the cell (below
        the lip of whatever sits above, for the top edge)."""
        dy = (EDGE * toward[0] + LIP * (toward[0] < 0)) * CELL_H if any(toward) else 6
        sx, sy = to_screen(GRID_X + (c + 0.5 + EDGE * toward[1]) * CELL_W,
                           GRID_Y + (r + 0.5) * CELL_H + dy, self.win)
        px, py = int((sx - self.x) * self.k), int((sy - self.y) * self.k)
        d = max(2, round(5 * self.k / self.win[2]))
        return classify(self.px[py - d:py + d, px - d:px + d].reshape(-1, 3).mean(axis=0))


def arrived(win, cells, color, step, full=False):
    """True once the screen shows a tray of `color` onto `cells`, having slid in moving `step`:
    its leading cells show its color and the ones it left behind don't. That is true from about
    halfway; with full, the cells left behind must be clear right up to the tray."""
    look = Look(win)
    if look.px is None:                           # can't look: give it the old fixed pause
        time.sleep(CORNER_PAUSE)
        return True
    back = shift(cells, (-step[0], -step[1]))
    if not all(look.color(*x) == color for x in cells - back):
        return False
    if full:
        return all(look.color(*x, step) == "empty" for x in back - cells)
    return not any(look.color(*x) == color for x in back - cells)


def has_label(img, r, c):
    """True if the white '0%' text sits in this cell's top-left = a tray's anchor cell."""
    x0, y0 = cell_origin(r, c)
    patch = img[y0 + 3:y0 + int(CELL_H * 0.45), x0 + 3:x0 + int(CELL_W * 0.6)]
    return int(np.all(patch > 230, axis=2).sum()) >= LABEL_MIN_WHITE


def same_tray(img, r0, c0, r1, c1):
    """True if adjacent cells (r0,c0) and (r1,c1) share a continuous tray interior (no border)."""
    if r0 == r1:
        x = GRID_X + max(c0, c1) * CELL_W
        y = int(GRID_Y + (r0 + 0.5) * CELL_H) + 6
        strip = img[y - 4:y + 4, x - 8:x + 8].astype(float)
        return float(strip.std(axis=1).mean()) < 5.0
    x = int(GRID_X + (c0 + 0.5) * CELL_W)
    y = GRID_Y + max(r0, r1) * CELL_H
    strip = img[y - 8:y + 8, x - 4:x + 4].astype(float)
    return float(strip.std(axis=0).mean()) < 5.0


def bucket_color(img, b):
    x, y = b["sample"]
    rgb = img[y - 2:y + 3, x - 24:x + 25].reshape(-1, 3).mean(axis=0)
    name = classify(rgb)
    return (None if name in NON_PIECE else name), rgb


def bucket_colors(img):
    return {n: bucket_color(img, b)[0] for n, b in BUCKETS.items()}


def half_colors(img):
    """The color at the bottom of each half of each bucket (what pours into the tray right under
    that half), keyed like HALVES. Only which color, never how much: the pictures aren't to scale."""
    out = {}
    for n, cell in HALVES:
        x, y = int(GRID_X + (cell[1] + 0.5) * CELL_W), BUCKETS[n]["sample"][1]
        name = classify(img[y - 2:y + 3, x - 15:x + 16].reshape(-1, 3).mean(axis=0))
        out[(n, cell)] = None if name in NON_PIECE else name
    return out


def read_stripes(img, b):
    """Colors of a bucket's sand stripes, bottom first, sampled up the bucket's middle."""
    x = b["sample"][0]
    bottom, top = b["stripes"]
    runs = []
    for y in range(bottom, top - 1, -2):
        name = classify(img[y - 1:y + 2, x - 24:x + 25].reshape(-1, 3).mean(axis=0))
        if runs and runs[-1][0] == name:
            runs[-1][1] += 1
        else:
            runs.append([name, 1])
    out = []
    for name, n in runs:              # short runs are the blends between two stripes
        if n >= 4 and name not in NON_PIECE and (not out or out[-1] != name):
            out.append(name)
    return out


def shift(cells, off):
    return frozenset((r + off[0], c + off[1]) for r, c in cells)


class Board:
    def __init__(self, pieces, walls):
        self.pieces = pieces      # list of {"color": str, "cells": frozenset}
        self.walls = walls        # set of blocked cells
        self.rebuild()

    def rebuild(self):
        self.occ = {cell: i for i, p in enumerate(self.pieces) for cell in p["cells"]}

    def open(self, cell):         # on the board and not a wall (ignores trays)
        r, c = cell
        return 0 <= r < ROWS and 0 <= c < COLS and cell not in self.walls

    def free_for(self, i, cell):  # open and not taken by another tray
        return self.open(cell) and self.occ.get(cell, i) == i

    def move(self, i, off):
        self.pieces[i]["cells"] = shift(self.pieces[i]["cells"], off)
        self.rebuild()


def parse(img):
    colors = [[classify(cell_rgb(img, r, c)) for c in range(COLS)] for r in range(ROWS)]
    anchors = {(r, c) for r in range(ROWS) for c in range(COLS)
               if colors[r][c] not in NON_PIECE and has_label(img, r, c)}
    owned, pieces = set(), []
    for a in sorted(anchors):                      # grow each tray right/down from its label
        col, cells, stack = colors[a[0]][a[1]], {a}, [a]
        owned.add(a)
        while stack:
            r, c = stack.pop()
            for n in ((r + 1, c), (r, c + 1)):
                if (0 <= n[0] < ROWS and 0 <= n[1] < COLS and n not in owned
                        and n not in anchors and colors[n[0]][n[1]] == col
                        and same_tray(img, r, c, n[0], n[1])):
                    owned.add(n); cells.add(n); stack.append(n)
        pieces.append({"color": col, "cells": frozenset(cells)})
    walls = {(r, c) for r in range(ROWS) for c in range(COLS)
             if colors[r][c] == "wall" or (colors[r][c] not in NON_PIECE and (r, c) not in owned)}
    return Board(pieces, walls), colors, anchors


def trays(board):
    return frozenset((p["color"], p["cells"]) for p in board.pieces)


UNDER = frozenset(c for b in BUCKETS.values() for c in b["targets"])


def away(ts):
    """The trays not under a bucket (a tray that is filling can be hard to read)."""
    return frozenset(t for t in ts if not t[1] & UNDER)


# ======================= LEVEL =======================
def sand_split(board, stripes):
    """How many tray cells each stripe fills: [[color, cells], ...] per bucket, bottom first.
    Every tray must be filled by a stripe of its color, so a color's trays are dealt out evenly
    over its stripes. When two buckets share a color (brown and pink here) the real split can't be
    seen; if it differs, a bucket changes color sooner or later than expected and play() re-solves."""
    sand = {n: [[c, 0] for c in s] for n, s in stripes.items()}
    slots = {}
    for s in sand.values():
        for stripe in s:
            slots.setdefault(stripe[0], []).append(stripe)
    for p in sorted(board.pieces, key=lambda p: -len(p["cells"])):
        if p["color"] in slots:
            min(slots[p["color"]], key=lambda s: s[1])[1] += len(p["cells"])
    return sand


class Level:
    """The puzzle as a model: the trays, plus the sand left in each bucket's stripes. A tray that
    comes to rest under a bucket of its color fills and vanishes at once (the game takes a few
    seconds; play() waits for it)."""

    def __init__(self, board, stripes):
        self.board = copy.deepcopy(board)
        self.sand = sand_split(board, stripes)
        for s in self.sand.values():
            while s and s[0][1] <= 0:
                s.pop(0)

    def bcol(self):
        return {n: (s[0][0] if s else None) for n, s in self.sand.items()}

    def settle(self):
        """Fill and remove every tray resting under a bucket of its color. Returns them."""
        gone = []
        while True:
            cols = self.bcol()
            hit = next(((n, i) for n, b in BUCKETS.items() for i, p in enumerate(self.board.pieces)
                        if p["color"] == cols[n] and p["cells"] & set(b["targets"])), None)
            if hit is None:
                return gone
            n, i = hit
            p = self.board.pieces.pop(i)
            self.board.rebuild()
            gone.append((p["color"], p["cells"]))
            s = self.sand[n]
            s[0][1] -= len(p["cells"])
            while s and s[0][1] <= 0:
                s.pop(0)


HALVES = [(n, tuple(cell)) for n, b in BUCKETS.items() for cell in b["targets"]]
EPS = 1e-6


def halves_of(stripes):
    """A bucket's stripes, given to each of its halves (the stripes start out flat)."""
    return {h: list(stripes[h[0]]) for h in HALVES}


def half_split(board, stripes):
    """Sand in each half of each bucket: {half: [[color, amount], ...]}, bottom first, in tray
    cells. The pictures aren't to scale, so the amounts come from the trays: a color's sand is
    exactly what its trays need, shared evenly by every half with a stripe of that color."""
    need, slots = {}, {}
    for p in board.pieces:
        need[p["color"]] = need.get(p["color"], 0) + p.get("need", len(p["cells"]))
    for s in stripes.values():
        for c in s:
            slots[c] = slots.get(c, 0) + 1
    return {h: [[c, need.get(c, 0) / slots[c]] for c in s] for h, s in stripes.items()}


class Halves:
    """The puzzle as a model, half by half: each half of a bucket pours only into the tray right
    under it, and only while its bottom stripe is that tray's color; a tray under both halves takes
    from both. A full tray vanishes (at once here; the game takes a few seconds). One whose halves
    run out of its color stays part filled until a half above it shows the color again."""

    def __init__(self, board, stripes):
        self.board = copy.deepcopy(board)
        for p in self.board.pieces:
            p.setdefault("need", float(len(p["cells"])))
        self.sand = half_split(self.board, stripes)
        self.drain()

    def drain(self):
        for s in self.sand.values():
            while s and s[0][1] <= EPS:
                s.pop(0)

    def bcol(self):
        return {h: (s[0][0] if s else None) for h, s in self.sand.items()}

    def settle(self):
        """Pour into every tray under a half showing its color; remove the full ones. Returns them."""
        gone = []
        while True:
            for i, p in enumerate(self.board.pieces):
                src = [s for h, s in self.sand.items() if h[1] in p["cells"] and s and s[0][0] == p["color"]]
                if src:
                    break
            else:
                return gone
            share = min([p["need"] / len(src)] + [s[0][1] for s in src])
            for s in src:                       # even shares, then the rest from whichever has more
                s[0][1] -= share
                p["need"] -= share
            for s in sorted(src, key=lambda s: -s[0][1]):
                take = min(s[0][1], p["need"])
                s[0][1] -= take
                p["need"] -= take
            self.drain()
            if p["need"] <= EPS:
                self.board.pieces.pop(i)
                self.board.rebuild()
                gone.append((p["color"], p["cells"]))


def room(cells):
    """For a tray on `cells` under just one half of a bucket: the cells it would slide into to get
    under the other half as well. The halves don't split a color evenly, so a half can run out
    early and leave the tray part filled; it then needs this slide. Empty for any other tray."""
    touch = [h for h in HALVES if h[1] in cells]
    if len(touch) != 1:
        return frozenset()
    (name, (_, c)), = touch
    other = next(h[1][1] for h in HALVES if h[0] == name and h != touch[0])
    return shift(cells, (0, other - c)) - cells


def deliverable(level, color, strict=True):
    """Where a tray of `color` can go to be filled now: the cells under halves showing that color,
    and a test for a tray's cells there. It passes if the halves the tray touches hold enough for
    all of it, and it isn't under just one half while the bucket's other half has more of the
    color (so the two halves drain evenly and neither runs dry early). With strict=False, any spot
    under a half showing the color passes."""
    amt = {h: s[0][1] for h, s in level.sand.items() if s and s[0][0] == color}

    def ok(cells):
        touch = [h for h in amt if h[1] in cells]
        if not touch or not strict:
            return bool(touch)
        if sum(amt[h] for h in touch) + EPS < len(cells):
            return False
        return len(touch) > 1 or all(amt[h] <= amt[touch[0]] + EPS for h in amt if h[0] == touch[0][0])

    return {h[1] for h in amt}, ok


# ======================= SEARCH MODEL =======================
# Cells are bits (r * COLS + c) of a Python int, so occupancy tests are single AND operations.
NCELL = ROWS * COLS
FULL = (1 << NCELL) - 1
FIRST_COL = sum(1 << (r * COLS) for r in range(ROWS))
LAST_COL = FIRST_COL << (COLS - 1)
INF = 99


def cell_bits(cells):
    return sum(1 << (r * COLS + c) for r, c in cells)


def bits_cells(mask):
    return frozenset(divmod(k, COLS) for k in range(NCELL) if mask >> k & 1)


def dilate(x):
    return (((x << 1) & ~FIRST_COL) | ((x >> 1) & ~LAST_COL) | (x << COLS) | (x >> COLS)) & FULL


def flood(x, passable):
    while True:
        y = x | (dilate(x) & passable)
        if y == x:
            return x
        x = y


class Shape:
    """A tray shape: its cell mask for every anchor (bounding-box top-left) it can occupy."""

    def __init__(self, rel, blocked):
        hgt, wid = 1 + max(r for r, _ in rel), 1 + max(c for _, c in rel)
        self.mask = {}
        for r in range(ROWS - hgt + 1):
            for c in range(COLS - wid + 1):
                m = cell_bits((r + dr, c + dc) for dr, dc in rel)
                if not m & blocked:
                    self.mask[r * COLS + c] = m
        self.nbrs = {p: [q for q in (p - COLS, p + COLS, p - 1, p + 1)
                         if q in self.mask and abs(q % COLS - p % COLS) <= 1]
                     for p in self.mask}
        # cells a one-step slide from p would newly cover; if all are taken the tray can't move
        self.edge = {}
        for p, qs in self.nbrs.items():
            m = 0
            for q in qs:
                m |= self.mask[q]
            self.edge[p] = m & ~self.mask[p]


class Model:
    """Search-side copy of the board. A state is a tuple of tray anchors. Trays with the same color
    and shape are interchangeable, so each such group's anchors are kept sorted and equivalent
    boards collapse into one state."""

    def __init__(self, board):
        self.blocked = cell_bits(board.walls)
        trays = []
        for p in board.pieces:
            r0 = min(r for r, _ in p["cells"]); c0 = min(c for _, c in p["cells"])
            rel = tuple(sorted((r - r0, c - c0) for r, c in p["cells"]))
            trays.append((p["color"], rel, r0 * COLS + c0))
        trays.sort()
        self.shapes = {}
        for _, rel, _ in trays:
            if rel not in self.shapes:
                self.shapes[rel] = Shape(rel, self.blocked)
        self.color = [t[0] for t in trays]
        self.rel = [t[1] for t in trays]
        self.shape = [self.shapes[t[1]] for t in trays]
        self.start = tuple(t[2] for t in trays)
        self.bounds, self.gid, self.span = [], [], []
        k = 0
        while k < len(trays):
            j = k
            while j < len(trays) and trays[j][:2] == trays[k][:2]:
                j += 1
            self.bounds.append((k, j))
            self.gid += [len(self.bounds) - 1] * (j - k)
            self.span += [(k, j)] * (j - k)
            k = j

    def occ(self, pos):
        o = self.blocked
        for t, p in enumerate(pos):
            o |= self.shape[t].mask[p]
        return o

    def reach(self, pos, occ, t):
        """Anchors tray t can be dragged to in one go (any free path), nearest first."""
        sh, p = self.shape[t], pos[t]
        rest = occ ^ sh.mask[p]
        seen, out, todo = {p}, [], deque([p])
        while todo:
            x = todo.popleft()
            for q in sh.nbrs[x]:
                if q not in seen and not sh.mask[q] & rest:
                    seen.add(q); out.append(q); todo.append(q)
        return out, rest

    def moves(self, pos, occ):
        out = []
        free = ~occ
        for t, p in enumerate(pos):
            sh = self.shape[t]
            if not sh.edge[p] & free:
                continue                  # boxed in: skip without searching
            dests, rest = self.reach(pos, occ, t)
            for q in dests:
                out.append((t, p, q, rest | sh.mask[q]))
        return out

    def apply(self, pos, t, q):
        s, e = self.span[t]
        lst = list(pos)
        lst[t] = q
        if e - s > 1:
            lst[s:e] = sorted(lst[s:e])
        return tuple(lst)

    def find(self, pos, g, p):
        s, e = self.bounds[g]
        return next((t for t in range(s, e) if pos[t] == p), None)


class Goal:
    """Get any tray of `color` (or just tray `only`) onto one of `targets`, at a spot whose cells
    pass `accept` if that is given, and with the cells `room(spot)` free if that is given.
    h() = 0 when done, else 1 + the fewest trays that must clear out of the best route there."""

    def __init__(self, model, color, targets, only=None, accept=None, room=None):
        self.m = model
        self.T = cell_bits(targets)
        self.cands = {}
        for t, col in enumerate(model.color):
            if col == color and (only is None or t == only):
                self.cands.setdefault(model.rel[t], []).append(t)
        self.info, self.room = {}, {}
        for rel in self.cands:
            hgt, wid = 1 + max(r for r, _ in rel), 1 + max(c for _, c in rel)
            valid = tp = 0
            for r in range(ROWS - hgt + 1):
                for c in range(COLS - wid + 1):
                    m = cell_bits((r + dr, c + dc) for dr, dc in rel)
                    if not m & model.blocked:
                        valid |= 1 << (r * COLS + c)
                        spot = {(r + dr, c + dc) for dr, dc in rel}
                        if m & self.T and (accept is None or accept(spot)):
                            tp |= 1 << (r * COLS + c)
                            need = cell_bits(room(spot)) if room else 0
                            if need:
                                self.room.setdefault(rel, []).append((r * COLS + c, need))
            self.info[rel] = (valid, tp, [dr * COLS + dc for dr, dc in rel])

    def spots(self, rel, free):
        """The target anchors for shape rel whose room is all within `free`."""
        tp = self.info[rel][1]
        for q, need in self.room.get(rel, ()):
            if need & ~free:
                tp &= ~(1 << q)
        return tp

    def h(self, pos, occ):
        m = self.m
        best = INF
        for rel, ts in self.cands.items():
            valid, tp, offs = self.info[rel]
            mask = m.shapes[rel].mask
            reach = own = 0
            for t in ts:
                reach |= 1 << pos[t]
                own |= mask[pos[t]]
            if reach & self.spots(rel, FULL & ~occ):
                return 0
            free = (FULL & ~occ) | own
            layers = 0
            while layers + 1 < best:      # each layer = one more tray pushed out of the way
                ok = valid
                for o in offs:
                    ok &= free >> o
                reach = flood(reach, ok)
                if reach & self.spots(rel, free):
                    best = layers + 1
                    break
                front = dilate(reach) & valid & ~reach
                cells = 0
                for o in offs:
                    cells |= front << o
                for q, need in self.room.get(rel, ()):     # a target reached but its room taken
                    if reach >> q & 1:
                        cells |= need
                cells &= ~free
                grew = False
                for t2, p2 in enumerate(pos):
                    mk = m.shape[t2].mask[p2]
                    if mk & cells:
                        free |= mk
                        grew = True
                if not grew:
                    break
                layers += 1
        return best


def iw(model, goal, pos, occ, h0, width, score):
    """Breadth-first search that only keeps boards putting some tray somewhere new (width 1), or
    some pair of trays in a new combination (width 2), until boards with h < h0 turn up. Of those
    found at that depth it returns the best by (h, score) as (moves, state, occ, h), or None.
    A generator: it yields now and then so several searches can share the time."""
    G = NCELL
    gid = model.gid

    def atoms(st):
        return {gid[t] * G + p for t, p in enumerate(st)}

    seen1 = atoms(pos)
    seen2 = {x: seen1 - {x} for x in seen1} if width > 1 else None   # atom -> atoms seen with it
    parent = {pos: None}
    todo = deque([(pos, occ, 0)])
    hits, hit_depth, n = [], None, 0
    while todo:
        cur, cocc, d = todo.popleft()
        if hit_depth is not None and d >= hit_depth:
            break
        n += 1
        if n % 40 == 0:
            yield
        here = atoms(cur) if width > 1 else None
        for t, p, q, nocc in model.moves(cur, cocc):
            a = gid[t] * G + q            # a novel board is never a repeat, so test novelty first
            if width > 1:
                others = here - {gid[t] * G + p}
                with_a = seen2.setdefault(a, set())
                if a in seen1 and with_a >= others:
                    continue
                with_a |= others
                for x in others:
                    seen2[x].add(a)
            elif a in seen1:
                continue
            seen1.add(a)
            nxt = model.apply(cur, t, q)
            parent[nxt] = (cur, (gid[t], p, q))
            hv = goal.h(nxt, nocc)
            if hv < h0:
                hits.append((hv, nxt, nocc))
                hit_depth = d + 1
            else:
                todo.append((nxt, nocc, d + 1))
    if not hits:
        return None
    hv, nxt, nocc = min(hits, key=lambda x: (x[0], score(x[1], x[2]))) if len(hits) > 1 else hits[0]
    steps, k = [], nxt
    while parent[k] is not None:
        k, mv = parent[k]
        steps.append(mv)
    return steps[::-1], nxt, nocc, hv


def replay(model, moves):
    """Play (group, from, to) drags from the start state; None if any drag is impossible."""
    pos = model.start
    occ = model.occ(pos)
    for g, p, q in moves:
        t = model.find(pos, g, p)
        if t is None:
            return None
        dests, rest = model.reach(pos, occ, t)
        if q not in dests:
            return None
        occ = rest | model.shape[t].mask[q]
        pos = model.apply(pos, t, q)
    return pos, occ


def shorten(model, goal, moves, deadline):
    """Merge back-to-back drags of the same tray and drop drags the plan can do without."""
    def works(ms):
        end = replay(model, ms)
        return end is not None and goal.h(*end) == 0

    moves = list(moves)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        k = 0
        while k + 1 < len(moves):                      # A->B then B->C  ==  A->C
            (g, p, q), (g2, p2, q2) = moves[k], moves[k + 1]
            if g == g2 and q == p2:
                moves[k:k + 2] = [] if p == q2 else [(g, p, q2)]
                changed = True
            else:
                k += 1
        k = 0
        while k < len(moves) and time.time() < deadline:
            g, p, q = moves[k]
            cand = moves[:k] + moves[k + 1:]
            for j in range(k, len(cand)):              # the tray stays at p, so its next drag starts there
                if cand[j][0] == g and cand[j][1] == q:
                    if cand[j][2] == p:
                        del cand[j]
                    else:
                        cand[j] = (g, p, cand[j][2])
                    break
            if works(cand):
                moves, changed = cand, True
            else:
                k += 1
    return moves


def solve(model, goal, score):
    """Serialized iterated width: repeatedly jump to the nearest board where fewer trays block the
    route, until a tray is there. Generator; returns [(group, from, to), ...] or None."""
    pos = model.start
    occ = model.occ(pos)
    h = goal.h(pos, occ)
    if h >= INF:
        return None
    moves = []
    while h > 0:
        res = yield from iw(model, goal, pos, occ, h, 1, score)
        if res is None:
            res = yield from iw(model, goal, pos, occ, h, 2, score)
        if res is None:
            return None
        steps, pos, occ, h = res
        moves += steps
    return moves


def race(jobs, budget):
    """Run solve() generators round-robin. Stops when all finish, the budget is spent, or shortly
    after the first one finds a plan. Returns {key: moves, or None if it failed / didn't finish}."""
    t0 = time.time()
    stop_at = t0 + budget
    live, done = dict(jobs), {}
    while live and time.time() < stop_at:
        for key, gen in list(live.items()):
            try:
                next(gen)
            except StopIteration as fin:
                done[key] = fin.value
                del live[key]
                if fin.value is not None:
                    now = time.time()
                    stop_at = min(stop_at, now + max(0.3, 0.5 * (now - t0)))
    return {key: done.get(key) for key in jobs}


# ======================= SOLVER =======================
def to_steps(model, moves):
    steps, pos = [], model.start
    for g, p, q in moves:
        t = model.find(pos, g, p)
        mask = model.shape[t].mask
        steps.append((model.color[t], bits_cells(mask[p]), bits_cells(mask[q])))
        pos = model.apply(pos, t, q)
    return steps


def blocking_score(model, bcol):
    """How badly a board blocks the trays the buckets want: summed over every tray of each
    bucket's color, the number of trays in its way. Breaks ties between equally short steps so
    a plan doesn't park trays across the next deliveries' routes."""
    goals = [Goal(model, bcol[n], b["targets"], only=t)
             for n, b in BUCKETS.items() if bcol[n]
             for t, col in enumerate(model.color) if col == bcol[n]]
    return lambda pos, occ: sum(min(g.h(pos, occ), 10) for g in goals)


def half_blocking_score(model, level):
    """blocking_score for the half-by-half model: trays in the way of every tray some half wants."""
    goals = []
    for color in sorted({c for c in level.bcol().values() if c}):
        cells, ok = deliverable(level, color)
        goals += [Goal(model, color, cells, only=t, accept=ok, room=room)
                  for t, col in enumerate(model.color) if col == color]
    return lambda pos, occ: sum(min(g.h(pos, occ), 10) for g in goals)


def next_delivery(level, budget=PLAN_TIME, strict=True):
    """Fewest drags that bring a tray to a spot under a bucket where it fills (see deliverable),
    the smaller tray when two take as many. Returns (color, [(color, from cells, to cells), ...])
    or None."""
    model = Model(level.board)
    score = half_blocking_score(model, level)
    goals = {}
    for color in sorted({c for c in level.bcol().values() if c}):
        cells, ok = deliverable(level, color, strict)
        g = Goal(model, color, cells, accept=ok, room=room)
        if g.cands:
            goals[color] = g
    found = race({c: solve(model, g, score) for c, g in goals.items()}, budget)
    best = None
    for c, moves in found.items():
        if moves:
            steps = to_steps(model, shorten(model, goals[c], moves, time.time() + 0.5))
            rank = (len(steps), len(steps[-1][1]))
            if best is None or rank < best[0]:
                best = (rank, c, steps)
    return best and best[1:]


# One drag of a solution. color/src: the tray and the cells it starts on; route: the offsets it
# slides through (the last is where it ends up); gone: trays that then fill and vanish; after: all
# trays left once they have; bcol: the color at the bottom of each half of each bucket then.
Drag = namedtuple("Drag", "color src route gone after bcol")


def solve_level(board, stripes, budget=PLAN_TIME, log=print):
    """Plan the whole level: deliver whichever tray is quickest to get under the bucket half (or
    halves) that will fill it, let it vanish, repeat until the board is empty. `stripes` are each
    half's, bottom first. Returns (start, drags), where start is a Drag-like record of trays
    already filling; None if it gets stuck."""
    level = Halves(board, stripes)
    gone = level.settle()
    start = Drag(None, None, None, tuple(gone), trays(level.board), level.bcol())
    drags, t0 = [], time.time()
    while level.board.pieces:
        t1 = time.time()
        found = next_delivery(level, budget) or next_delivery(level, 6 * budget)
        if not found:
            found = next_delivery(level, 6 * budget, strict=False)
            if found:
                log("  (no tray can be filled completely right now; this one only partly)")
        if not found:
            log(f"  stuck with {len(level.board.pieces)} trays left; the halves show {level.bcol()}")
            return None
        color, steps = found
        n0 = len(drags)
        for c, src, dst in steps:
            i = find_tray(level.board, c, src)
            route = drag_route(level.board, i, dst) if i is not None else None
            if route is None:         # an earlier drag already filled this tray; plan again
                break
            level.board.move(i, route[-1])
            gone = level.settle()
            drags.append(Drag(c, src, tuple(route), tuple(gone), trays(level.board), level.bcol()))
        if len(drags) == n0:
            log(f"  stuck: the plan for {color} goes nowhere")
            return None
        under = " + ".join(f"{h[0]} {h[1]}" for h in HALVES if h[1] in steps[-1][2])
        log(f"  {color:8s} -> {under:22s} {len(drags) - n0:2d} drags   "
            f"{len(level.board.pieces):2d} trays left   ({time.time() - t1:.1f}s)")
    log(f"Solved: {len(drags)} drags, {time.time() - t0:.0f}s of thinking")
    return start, drags


# ======================= ACTION =======================
def find_tray(board, color, cells):
    return next((k for k, p in enumerate(board.pieces) if p["cells"] == cells and p["color"] == color), None)


def drag_route(board, i, dst):
    """Slide of tray i onto cells `dst` with the fewest cells, then fewest turns (offsets from start)."""
    base = board.pieces[i]["cells"]
    (r0, c0), (r1, c1) = min(base), min(dst)
    goal = (r1 - r0, c1 - c0)
    tie = itertools.count()
    start = ((0, 0), None)
    best, prev = {start: 0}, {}
    pq = [(0, next(tie), start)]
    while pq:
        cost, _, node = heapq.heappop(pq)
        off, d = node
        if off == goal:
            path = [off]
            while node in prev:
                node = prev[node]
                path.append(node[0])
            return path[::-1]
        if cost > best[node]:
            continue
        for nd in DIRS:
            n = (off[0] + nd[0], off[1] + nd[1])
            if all(board.free_for(i, x) for x in shift(base, n)):
                nc = cost + 10 + (3 if d not in (None, nd) else 0)
                if nc < best.get((n, nd), 1 << 30):
                    best[(n, nd)] = nc
                    prev[(n, nd)] = node
                    heapq.heappush(pq, (nc, next(tie), (n, nd)))
    return None


def corners(path):
    if len(path) <= 2:
        return path
    out = [path[0]]
    for a, b, c in zip(path, path[1:], path[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            out.append(b)
    return out + [path[-1]]


def heading(a, b):
    """The unit step from offset a toward offset b along a straight stretch."""
    return (b[0] > a[0]) - (b[0] < a[0]), (b[1] > a[1]) - (b[1] < a[1])


def ready(win, cells, color, walls):
    """Whether the screen shows the board as it was (its walls still read as walls, so nothing
    such as the time-up panel is drawn over it) with a tray of `color` on `cells` to press on.
    Cells under a bucket aren't read: sand pouring in can hide a tray's color."""
    look = Look(win)
    if look.px is None:
        return True
    return (all(look.color(*x) == "wall" for x in walls)
            and all(look.color(*x) == color for x in cells - UNDER))


def execute(move, win, color=None, free=None, walls=()):
    """Drag a tray along a path. Given the tray's color, the finger only presses once the screen
    shows the tray on the board (see ready()), and waits at every corner and at the end until it
    shows the tray there, letting go early if it doesn't come. Given free(cell), it also makes
    sure the tray is round each corner before turning (one that turns early jams if the cells
    beside its tail are taken): where the tray can't go straight on, the finger goes a little past
    the corner and waits for the tray to be all the way in; where it could go on, the drag stops
    at the corner and a new one starts, as a tray let go snaps into line. Returns False if the
    tray didn't get where it was going."""
    cells, path = move
    ar, ac = min(cells)                                       # grab the tray's top-left cell
    turns = corners(path)
    pushes, tight, parts = set(), set(), [[turns[0]]]
    for a, b, c in zip(turns, turns[1:], turns[2:]):
        parts[-1].append(b)
        if free:
            d1, d2 = heading(a, b), heading(b, c)
            at = shift(cells, b)
            caught = not all(free(x) for x in shift(shift(at, (-d1[0], -d1[1])), d2) - at)
            if not all(free(x) for x in shift(at, d1) - at):
                pushes.add(b)
                if caught:
                    tight.add(b)
            elif caught:
                parts.append([b])
    parts[-1].append(turns[-1])
    for part in parts:
        if color and not ready(win, shift(cells, part[0]), color, walls):
            return False
        pts = [[GRID_X + (ac + dc + 0.5) * CELL_W, GRID_Y + (ar + dr + 0.5) * CELL_H] for dr, dc in part]
        for j in range(1, len(part)):
            push = OVERSHOOT if j == len(part) - 1 else CORNER_PUSH if part[j] in pushes else 0
            dr, dc = heading(part[j - 1], part[j])
            pts[j][0] += push * CELL_W * dc
            pts[j][1] += push * CELL_H * dr
        checks = None
        if color:
            checks = [lambda at=shift(cells, b), step=heading(a, b), full=b in tight:
                      arrived(win, at, color, step, full) for a, b in zip(part, part[1:])]
        if not drag([to_screen(x, y, win) for x, y in pts], checks):
            return False
    return True


def await_board(want, bcol):
    """Look until the screen shows exactly the trays `want` and bucket halves showing `bcol`. Trays
    under a bucket get FILL_WAIT seconds to fill and vanish, a half 2 s to show its next color, and
    anything else three looks to come right. Returns None once it matches, else the last screenshot."""
    misses = off_color = 0
    for _ in range(int(FILL_WAIT / POLL)):
        time.sleep(POLL)
        pyautogui.failSafeCheck()
        img, _ = capture()
        seen = trays(parse(img)[0])
        if seen == want:
            if not want or half_colors(img) == bcol:
                return None
            off_color += 1
            if off_color * POLL >= 2:
                return img
            continue
        misses = 0 if away(seen) == away(want) else misses + 1
        if misses >= 3:
            break
    return img if want else None          # an empty board can hide behind the "level done" screen


def carry_out(d, before, win, walls):
    """Do one planned drag and make sure the tray gets there: if it stops short (or doesn't move),
    drag it on from wherever it is. Then wait for any trays it fills to vanish. `before` is the
    board the drag starts from. Returns None once the board is what the solution expects, else
    the screenshot showing it isn't."""
    target = (d.color, shift(d.src, d.route[-1]))
    mid = (before - {(d.color, d.src)}) | {target}       # the board before anything fills
    board = Board([{"color": c, "cells": x} for c, x in before], set(walls))
    cells, route = d.src, d.route
    for attempt in range(4):
        i = find_tray(board, d.color, cells)
        there = execute((cells, route), win, d.color, lambda x: board.free_for(i, x), walls)
        for look in range(3):                            # a tray still snapping into place looks off
            if look or not there:
                time.sleep(SETTLE if look == 0 else RELOOK)
            img, _ = capture()
            board = parse(img)[0]
            seen = trays(board)
            if seen == mid or (d.gone and away(seen) == away(d.after)):
                return await_board(d.after, d.bcol) if d.gone else None
        extra, missing = seen - mid, mid - seen
        if missing != {target} or len(extra) != 1 or attempt == 3:
            return img                                   # something other than this tray is off
        (color, now), = extra
        if color != d.color or len(now) != len(cells):
            return img
        at_turn = any(now == shift(cells, off) for off in corners(route)[1:-1])
        note = " - it didn't move at all" if now == cells else ", the corner" if at_turn else ""
        cells, route = now, drag_route(board, find_tray(board, color, now), target[1])
        if route is None:
            return img
        print(f"        it stopped at {min(now)}{note}")
    return img


def land(d, before, win, walls, deadline=math.inf):
    """follow: do one drag and make sure the tray gets there, finishing the slide from wherever it
    stops short (but not after `deadline`). Trays under the buckets aren't compared, since they
    may be pouring. Returns None once it has landed, else the screenshot showing it hasn't."""
    target = (d.color, shift(d.src, d.route[-1]))
    want = away((before - {(d.color, d.src)}) | {target})
    board = Board([{"color": c, "cells": x} for c, x in before], set(walls))
    cells, route = d.src, d.route
    for attempt in range(4):
        i = find_tray(board, d.color, cells)
        there = execute((cells, route), win, d.color, lambda x: board.free_for(i, x), walls)
        for look in range(3):                            # a tray still snapping into place looks off
            if look or not there:
                time.sleep(SETTLE if look == 0 else RELOOK)
            img, _ = capture()
            board = parse(img)[0]
            seen = away(trays(board))
            if seen == want:
                return None
        extra, missing = seen - want, want - seen
        if len(extra) != 1 or missing - {target} or attempt == 3 or time.time() > deadline:
            return img
        (color, now), = extra
        if color != d.color or len(now) != len(cells):
            return img
        at_turn = any(now == shift(cells, off) for off in corners(route)[1:-1])
        note = " - it didn't move at all" if now == cells else ", the corner" if at_turn else ""
        cells, route = now, drag_route(board, find_tray(board, color, now), target[1])
        if route is None:
            return img
        print(f"        it stopped at {min(now)}{note}", flush=True)
    return img


def nudge(tray, board, img, win, walls):
    """follow: a tray under a bucket that has stopped filling (no half above it shows its color)
    slides under the bucket's other half if that half shows its color and there's room. Returns
    the tray as it is now."""
    color, cells = tray
    tops = half_colors(img)
    touch = [h for h in HALVES if h[1] in cells]
    i = find_tray(board, color, cells)
    if not touch or any(tops[h] == color for h in touch) or i is None:
        return tray
    for h in HALVES:
        if h[0] == touch[0][0] and h not in touch and tops[h] == color:
            spots = [shift(cells, (h[1][0] - r, h[1][1] - c)) for r, c in cells]
            routes = [r for r in (drag_route(board, i, s) for s in spots) if r]
            if routes:
                route = min(routes, key=len)
                print(f"        the {color} tray at {min(cells)} stopped filling - sliding it under "
                      f"the other half of the {h[0]} bucket", flush=True)
                execute((cells, route), win, color, lambda x: board.free_for(i, x), walls)
                return color, shift(cells, route[-1])
    return tray


def in_way(tray, path, dest, tops=None):
    """follow: whether a pouring tray is in the way of a drag through the cells `path` that ends on
    `dest`: it covers a cell of the path, or the drag would park on the room it may need to slide
    into to finish filling (see room()) while the half over that room shows its color. `tops` are
    the colors the halves show; without them, any half might."""
    color, cells = tray
    if cells & path:
        return True
    near = room(cells)
    return bool(near & dest) and (tops is None or any(tops[h] == color for h in HALVES if h[1] in near))


def track(tray, board, colors):
    """follow: where a pouring tray is now (a nudge may have slid it into its room), or None once
    it has filled and gone, which is when its cells read empty: a tray full of sand can misread."""
    color, cells = tray
    near = cells | room(cells)
    for p in board.pieces:
        if p["color"] == color and p["cells"] & near:
            return color, p["cells"]
    return None if all(colors[r][c] == "empty" for r, c in cells) else tray


def wait_poured(pouring, path, dest, deadline, win, walls):
    """follow: wait until no pouring tray is in the way of a drag through `path` to `dest` (see
    in_way), nudging any that stopped filling. Returns the trays still pouring, or None if one is
    still in the way after POUR_WAIT (or at the deadline)."""
    end = min(time.time() + POUR_WAIT, deadline)
    said = False
    while True:
        img, _ = capture()
        board, colors, _ = parse(img)
        tops = half_colors(img)
        pouring = [t for t in (track(t, board, colors) for t in pouring) if t]
        if not any(in_way(t, path, dest, tops) for t in pouring):
            return pouring
        if time.time() > end:
            return None
        if not said:
            print("        waiting for a pouring tray to get out of the way", flush=True)
            said = True
        pouring = [nudge(t, board, img, win, walls) if in_way(t, path, dest, tops) else t
                   for t in pouring]
        pyautogui.failSafeCheck()
        time.sleep(POLL)


started = None   # when the first drag began: the game's clock starts with it


def perform(sol, win, walls, blind=False):
    """Carry out a solution, checking every drag (with blind=True, only the deliveries). Returns
    None if the game went as planned, else the screenshot showing it didn't."""
    global started
    start, drags = sol
    if start.gone:
        print(f"  waiting for {len(start.gone)} filling tray(s)")
        bad = await_board(start.after, start.bcol)
        if bad is not None:
            return bad
    before = start.after
    for k, d in enumerate(drags, 1):
        pyautogui.failSafeCheck()
        print(f"  {k:3d}/{len(drags)}  {d.color:8s} {min(d.src)} -> {min(shift(d.src, d.route[-1]))}"
              + ("   fills" if d.gone else ""))
        t = time.time()
        started = started or t
        if blind and not d.gone:
            execute((d.src, d.route), win, d.color, walls=walls)
            time.sleep(GAP)
        else:
            bad = carry_out(d, before, win, walls)
            if bad is not None:
                return bad
        if d.gone:
            print(f"        filled and gone {time.time() - t:.1f}s after the drag began; "
                  f"{time.time() - started:.0f}s since the first drag")
        before = d.after
    return None


# ======================= COMMANDS =======================
def snap():
    img, _ = capture(refresh=True)
    Image.fromarray(img).save("window.png")
    over = Image.fromarray(img); d = ImageDraw.Draw(over)
    for r in range(ROWS):
        for c in range(COLS):
            x0, y0 = cell_origin(r, c)
            d.rectangle([x0, y0, x0 + CELL_W, y0 + CELL_H], outline=(255, 0, 0), width=2)
    for b in BUCKETS.values():
        x, y = b["sample"]; d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(0, 255, 0), width=3)
        d.line([x, b["stripes"][0], x, b["stripes"][1]], fill=(0, 255, 0), width=1)
    over.save("grid_check.png")
    print(f"Saved window.png ({img.shape[1]}x{img.shape[0]}) and grid_check.png")


def read():
    img, _ = capture(refresh=True)
    board, colors, anchors = parse(img)
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            n = colors[r][c]
            s = "  ." if n == "empty" else "###" if n == "wall" else n[:3]
            row.append(s.upper() if (r, c) in anchors else s)
        print(" ".join(row))
    for name, b in BUCKETS.items():
        col, rgb = bucket_color(img, b)
        print(f"{name} bucket: {col}  rgb={tuple(int(v) for v in rgb)}  "
              f"stripes (bottom first): {' '.join(read_stripes(img, b))}")
    print("bottom of each half: " + ", ".join(f"{h[0]} {h[1]} {c}" for h, c in half_colors(img).items()))
    print(f"{len(board.pieces)} trays (UPPERCASE = label/anchor cell)")
    if "-v" in sys.argv:
        for r in range(ROWS):
            print(" ".join(f"{tuple(int(v) for v in cell_rgb(img, r, c))!s:>15}" for c in range(COLS)))


def describe(board, stripes):
    print(f"{len(board.pieces)} trays. Sand in each bucket half, bottom first, in tray cells (worked "
          f"out from the trays, shared evenly where halves or buckets share a color):")
    for h, s in half_split(board, stripes).items():
        print(f"  {h[0]:5s} {h[1]}: " + ", ".join(f"{c} {k:.3g}" for c, k in s))
    homeless = {p["color"] for p in board.pieces} - {c for s in stripes.values() for c in s}
    if homeless:
        print(f"  !! no bucket has {', '.join(sorted(homeless))} - check the stripe reading")


def key_of(board, stripes):
    return "halves", trays(board), tuple(sorted((h, tuple(s)) for h, s in stripes.items()))


def solve_and_save(board, stripes):
    print("Solving the whole level (the clock hasn't started)...")
    sol = solve_level(board, stripes)
    if sol:
        start, drags = sol
        with open(SAVE, "wb") as f:          # plain tuples, so it loads however the bot is started
            pickle.dump((key_of(board, stripes), tuple(start), [tuple(d) for d in drags]), f)
        write_moves(drags)
        print(f"Saved to {os.path.basename(SAVE)} and {os.path.basename(MOVES)}")
    return sol


def load_solution(board, stripes):
    try:
        with open(SAVE, "rb") as f:
            key, start, drags = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError):
        return None
    return (Drag(*start), [Drag(*d) for d in drags]) if key == key_of(board, stripes) else None


def solve_cmd():
    path = next((a for a in sys.argv[2:] if not a.startswith("-")), None)
    img = np.array(Image.open(path).convert("RGB")) if path else capture(refresh=True)[0]
    board, _, _ = parse(img)
    stripes = halves_of({n: read_stripes(img, b) for n, b in BUCKETS.items()})
    describe(board, stripes)
    sol = solve_and_save(board, stripes)
    if sol and "-v" in sys.argv:
        for k, d in enumerate(sol[1], 1):
            print(f"  {k:3d}. {d.color:8s} {min(d.src)} -> {min(shift(d.src, d.route[-1]))}"
                  + (f"   fills {', '.join(c for c, _ in d.gone)}" if d.gone else ""))


STEPS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def write_moves(drags, path=MOVES):
    """The drags as a moves file that read_moves (and a person) can follow."""
    word = {v: k for k, v in STEPS.items()}
    lines = ["Level 125 moves. Each tray is named by its top-left cell (row, col): rows 0-10 top to",
             "bottom, cols 0-7 left to right. Slides go cell by cell, turning corners in one drag.", ""]
    done = 0
    for k, d in enumerate(drags, 1):
        legs = []
        turns = corners(d.route)
        for (r0, c0), (r1, c1) in zip(turns, turns[1:]):
            step = ((r1 > r0) - (r1 < r0), (c1 > c0) - (c1 < c0))
            legs.append(f"{word[step]} {abs(r1 - r0) + abs(c1 - c0)}")
        line = f"{k:3d}. {d.color:8s} at {min(d.src)}: {', '.join(legs)}"
        lines.append(line + (f"   -> fills {', '.join(c for c, _ in d.gone)}" if d.gone else ""))
        if d.gone:
            done += 1
            lines.append(f"     --- delivery {done} done ---")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def read_moves(path):
    """[(color, top-left cell, route)] from a file written like moves.txt."""
    out = []
    for line in open(path):
        m = re.match(r"\s*\d+\.\s+(\w+)\s+at \((\d+), (\d+)\):\s*([a-z0-9 ,]+)", line)
        if m:
            route = [(0, 0)]
            for leg in m.group(4).split(","):
                word, n = leg.split()
                for _ in range(int(n)):
                    route.append((route[-1][0] + STEPS[word][0], route[-1][1] + STEPS[word][1]))
            out.append((m.group(1), (int(m.group(2)), int(m.group(3))), route))
    return out


def script(board, stripes, moves):
    """The drags of a moves file played out on the model, so each one knows the board it leaves."""
    level = Halves(board, stripes)
    gone = level.settle()
    start = Drag(None, None, None, tuple(gone), trays(level.board), level.bcol())
    drags = []
    for k, (color, cell, route) in enumerate(moves, 1):
        i = next((j for j, p in enumerate(level.board.pieces)
                  if p["color"] == color and min(p["cells"]) == cell), None)
        if i is None:
            sys.exit(f"Move {k}: there is no {color} tray at {cell}.")
        src = level.board.pieces[i]["cells"]
        if not all(level.board.free_for(i, x) for off in route for x in shift(src, off)):
            sys.exit(f"Move {k}: the {color} tray at {cell} is blocked on the way.")
        level.board.move(i, route[-1])
        gone = level.settle()
        drags.append(Drag(color, src, tuple(route), tuple(gone), trays(level.board), level.bcol()))
    return start, drags


def follow():
    """Play the moves in moves.txt (or the file named) as written, with no planning. Each drag is
    checked and finished if the tray stops short. A delivered tray pours while the next moves go
    on; a move that needs its cells waits for it. Stops at the first board the file doesn't
    expect, and a few seconds before the clock runs out (so nothing drags on the time-up panel)."""
    path = next((a for a in sys.argv[2:] if not a.startswith("-")), MOVES)
    subprocess.run(["osascript", "-e", 'tell application "iPhone Mirroring" to activate'])
    time.sleep(0.5)
    img, win = capture(refresh=True)
    board, _, _ = parse(img)
    stripes = halves_of({n: read_stripes(img, b) for n, b in BUCKETS.items()})
    start, drags = script(board, stripes, read_moves(path))
    print(f"Following the {len(drags)} moves in {os.path.basename(path)}. "
          f"Abort: push the mouse into a screen corner.", flush=True)
    try:
        follow_moves(drags, start, board.walls, win, os.path.basename(path))
    finally:
        if waits:
            w = sorted(waits)
            print(f"The finger waited for the tray {len(w)} times: half the waits were under "
                  f"{w[len(w) // 2]:.2f}s, the longest {w[-1]:.2f}s.")


def follow_moves(drags, start, walls, win, name):
    """follow: the drags in turn, from the board `start` leaves; `name` is the moves file's."""
    pouring, before, t0, deadline = list(start.gone), start.after, None, math.inf
    for k, d in enumerate(drags, 1):
        pyautogui.failSafeCheck()
        path = frozenset().union(*(shift(d.src, off) for off in d.route))
        dest = shift(d.src, d.route[-1])
        if any(in_way(t, path, dest) for t in pouring):
            pouring = wait_poured(pouring, path, dest, deadline, win, walls)
            if pouring is None:
                print(f"Stopped before move {k}: a tray under a bucket is still in its way.")
                return
        if time.time() > deadline:
            print(f"Stopped before move {k}: the clock is about to run out.")
            return
        print(f"  {k:3d}/{len(drags)}  {d.color:8s} {min(d.src)} -> {min(shift(d.src, d.route[-1]))}"
              + ("   fills" if d.gone else ""), flush=True)
        if t0 is None:
            t0 = time.time()
            deadline = t0 + CLOCK - 6
        bad = land(d, before, win, walls, deadline)
        if bad is not None:
            Image.fromarray(bad).save(os.path.join(os.path.dirname(MOVES), "follow_stop.png"))
            why = ("the clock is about to run out" if time.time() > deadline
                   else f"the board isn't what {name} expects")
            print(f"Stopped at move {k}: {why} (screenshot in follow_stop.png).")
            return
        if d.gone:
            print(f"        {time.time() - t0:.0f}s since the first drag", flush=True)
        pouring += d.gone
        before = d.after
    print(f"All {len(drags)} moves done, {time.time() - t0:.0f}s after the first drag.")
    if pouring and wait_poured(pouring, UNDER, frozenset(), deadline, win, walls) is None:
        print("A tray under a bucket still hasn't filled.")


def trim(stripes, color):
    """A bucket's stripes still to come, now that it shows `color` at the bottom."""
    if color is None:
        return []
    return stripes[stripes.index(color):] if color in stripes else [color] + stripes


def play():
    blind = "--blind" in sys.argv
    subprocess.run(["osascript", "-e", 'tell application "iPhone Mirroring" to activate'])
    time.sleep(0.5)
    img, win = capture(refresh=True)
    board, _, _ = parse(img)
    stripes = halves_of({n: read_stripes(img, b) for n, b in BUCKETS.items()})
    describe(board, stripes)
    sol = load_solution(board, stripes)
    if sol:
        print(f"Using the saved solution ({len(sol[1])} drags).")
    else:
        sol = solve_and_save(board, stripes)
    walls = board.walls
    while sol:
        bad = perform(sol, win, walls, blind)
        if bad is None:
            print("Level done.")
            return
        board, _, _ = parse(bad)
        if not board.pieces:
            print("No trays left - level done?")
            return
        bcol = half_colors(bad)
        stripes = {h: trim(stripes[h], bcol[h]) for h in HALVES}
        print(f"The board isn't what the solution expects - solving again from here "
              f"({len(board.pieces)} trays, the halves show {bcol}).")
        sol = solve_level(board, stripes)
    print("Couldn't find a way to go on.")


if __name__ == "__main__":
    {"snap": snap, "read": read, "solve": solve_cmd, "plan": solve_cmd, "play": play, "follow": follow}[
        sys.argv[1] if len(sys.argv) > 1 else "read"]()
