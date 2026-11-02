#!/usr/bin/env python3
"""
Sand Block solver.

The board is a sliding-block puzzle. One drag moves one tray any number of
cells in a single direction; it stops where the pointer is released, and it
cannot cross a wall or another tray. Turning a corner takes a second drag.
A tray left on the matching bucket (the two cells under that bucket) fills
and disappears, and the bucket's next stripe becomes the color it wants.

Search is iterated width, from Lipovetzky & Geffner: a breadth-first search
that only keeps positions in which some tray stands somewhere it has not
stood before, repeated until a tray of the bucket's color is on that bucket.
At every step the tie-break is how many trays still cage the other bucket's
color, so the short plan is one that does not bury the next delivery.
The full plan is built before the first drag, so the clock is not running yet.
Once the clock is running, a replan searches only the next tray and plays it.
Clearing that tray opens space, and the tray after it is an easier search.

  .venv/bin/python sand_solver.py solve [window.png]
  .venv/bin/python sand_solver.py play

play drives iPhone Mirroring. Push the pointer into a screen corner to abort.
"""
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

import sand_hole_bot as sb

COLS = sb.COLS
DIRS = (-COLS, COLS, -1, 1)
PLAN_BUDGET = 8.0
HERE = os.path.dirname(os.path.abspath(__file__))


class Straight(sb.Model):
    """Same board as the old model, but a move is one straight slide."""

    def moves(self, pos, occ):
        out = []
        for t, p in enumerate(pos):
            sh = self.shape[t]
            rest = occ ^ sh.mask[p]
            for d in DIRS:
                cur = p
                while True:
                    q = cur + d
                    if q not in sh.mask:
                        break
                    if d == 1 and q % COLS == 0:
                        break
                    if d == -1 and cur % COLS == 0:
                        break
                    if sh.mask[q] & rest:
                        break
                    out.append((t, p, q, rest | sh.mask[q]))
                    cur = q
        return out

    def reach(self, pos, occ, t):
        sh, p = self.shape[t], pos[t]
        rest = occ ^ sh.mask[p]
        out = []
        for d in DIRS:
            cur = p
            while True:
                q = cur + d
                if q not in sh.mask:
                    break
                if d == 1 and q % COLS == 0:
                    break
                if d == -1 and cur % COLS == 0:
                    break
                if sh.mask[q] & rest:
                    break
                out.append(q)
                cur = q
        return out, rest


def colinear(p, q, r):
    pr, pc = divmod(p, COLS)
    qr, qc = divmod(q, COLS)
    rr, rc = divmod(r, COLS)
    return (pr == qr == rr) or (pc == qc == rc)


def shorten(model, goal, moves, deadline):
    """Drop slides the delivery can do without, and join two slides of the
    same tray that continue in one straight line."""

    def works(ms):
        end = sb.replay(model, ms)
        return end is not None and goal.h(*end) == 0

    moves = list(moves)
    k = 0
    while k + 1 < len(moves):
        (g, p, q), (g2, p2, q2) = moves[k], moves[k + 1]
        if g == g2 and q == p2 and colinear(p, q, q2):
            moves[k:k + 2] = [] if p == q2 else [(g, p, q2)]
        else:
            k += 1
    k = 0
    while k < len(moves) and time.time() < deadline:
        g, p, q = moves[k]
        cand = moves[:k] + moves[k + 1:]
        for j in range(k, len(cand)):
            if cand[j][0] == g and cand[j][1] == q:
                if cand[j][2] == p:
                    del cand[j]
                else:
                    cand[j] = (g, p, cand[j][2])
                break
        if works(cand):
            moves = cand
        else:
            k += 1
    return moves


def slide_of(src, dst):
    """(dr, dc) translating src cells onto dst. Both sets are the same shape."""
    (r0, c0), (r1, c1) = min(src), min(dst)
    return (r1 - r0, c1 - c0)


def straight(src, dst):
    dr, dc = slide_of(src, dst)
    if dr != 0 and dc != 0:
        return False
    moved = {(r + dr, c + dc) for r, c in src}
    return moved == set(dst)


def next_delivery(board, bcol, budget=None, extend=True):
    budget = PLAN_BUDGET if budget is None else budget
    model = Straight(board)
    score = sb.blocking_score(model, bcol)
    goals = {n: sb.Goal(model, bcol[n], bkt["targets"])
             for n, bkt in sb.BUCKETS.items() if bcol[n]}
    found = sb.race({n: sb.solve(model, g, score) for n, g in goals.items()}, budget)
    # The longer retry is only for the plan built before the clock starts.
    if extend and not any(found.values()):
        found = sb.race({n: sb.solve(model, g, score) for n, g in goals.items()},
                        min(budget * 3, 12.0))
    best = None
    for n, moves in found.items():
        if not moves:
            continue
        moves = shorten(model, goals[n], moves, time.time() + 0.4)
        steps = sb.to_steps(model, moves)
        if any(not straight(src, dst) for _, src, dst in steps):
            continue
        if best is None or len(steps) < len(best[1]):
            best = (n, steps)
    return best


def solve_level(board, stripes, log=print, budget=None, limit=None, extend=True):
    """Return a list of ('slide', color, src, dst) and
    ('fill', bucket, color, cells, bucket_colors_after). None if stuck.

    `limit` stops after that many deliveries. On the clock, limit is 1:
    play the next tray, then search again on the clearer board."""
    level = sb.Level(board, stripes)
    plan = []
    t0 = time.time()
    searches = 0
    while level.board.pieces:
        # A tray already standing on its bucket (a replan caught it mid-fill,
        # or the search's last slide put it there and we loop back) just drains.
        gone = level.settle()
        if gone:
            plan.append(("fill", _bucket_of(gone, level), gone, level.bcol()))
            log(f"  already on a bucket: {', '.join(c for c, _ in gone)}   "
                f"{len(level.board.pieces):2d} left")
            continue
        bcol = level.bcol()
        if not any(bcol.values()):
            log(f"stuck, {len(level.board.pieces)} trays, buckets want {bcol}")
            return plan or None
        t1 = time.time()
        found = next_delivery(level.board, bcol, budget, extend=extend)
        if not found:
            log(f"stuck, {len(level.board.pieces)} trays, buckets want {bcol}")
            return plan or None
        bucket, steps = found
        used = 0
        for color, src, dst in steps:
            i = sb.find_tray(level.board, color, src)
            if i is None or not straight(src, dst):
                break
            level.board.move(i, slide_of(src, dst))
            gone = level.settle()
            plan.append(("slide", color, src, dst))
            used += 1
            if gone:
                # This slide parked a tray on a bucket. Later slides in this
                # plan were searched on a board where that tray was still
                # there, so stop and search again from the board we have.
                plan.append(("fill", bucket if len(gone) == 1 else bucket,
                             color, src and dst, level.bcol()))
                # Record every tray that actually vanished, not just the one
                # the search was aiming at. The cells are where it ended.
                plan[-1] = ("fill", _bucket_of(gone, level), gone, level.bcol())
                break
        else:
            # The search promised a delivery but nothing filled. The tray is
            # sitting on the bucket only if the last slide put it there;
            # settle already ran. Nothing to do but search again.
            pass
        searches += 1
        log(f"  {bucket:5s} {bcol[bucket]:8s} {used:3d} slides   "
            f"{len(level.board.pieces):2d} left   ({time.time() - t1:.1f}s)")
        if used == 0:
            log("no legal slide in the plan")
            return plan or None
        if limit is not None and searches >= limit:
            return plan
    log(f"solved: {sum(s[0] == 'slide' for s in plan)} slides, "
        f"{sum(s[0] == 'fill' for s in plan)} trays, {time.time() - t0:.0f}s")
    return plan


def _bucket_of(gone, level):
    """Which bucket just accepted a tray. `gone` is the list settle() returned,
    each item (color, cells). Prefer the bucket whose target those cells hit."""
    hits = []
    for color, cells in gone:
        for name, bkt in sb.BUCKETS.items():
            if set(cells) & set(bkt["targets"]):
                hits.append(name)
    return hits[0] if hits else "?"


def load(path):
    img = np.array(Image.open(path).convert("RGB"))
    board = sb.parse(img)[0]
    stripes = {n: sb.read_stripes(img, b) for n, b in sb.BUCKETS.items()}
    return board, stripes


def show(board):
    abbr = {"brown": "BR", "pink": "PI", "orange": "OR", "yellow": "YE", "cyan": "CY",
            "red": "RE", "purple": "PU", "blue": "BL", "magenta": "MA", "lavender": "LA"}
    occ = {}
    for p in board.pieces:
        for cell in p["cells"]:
            occ[cell] = abbr[p["color"]]
    print("     " + " ".join(f"c{c}" for c in range(sb.COLS)))
    for r in range(sb.ROWS):
        row = []
        for c in range(sb.COLS):
            if (r, c) in occ:
                row.append(occ[(r, c)])
            elif (r, c) in board.walls:
                row.append("##")
            else:
                row.append(" .")
        print(f"r{r:<2} " + " ".join(row))


def write_plan(plan, path):
    with open(path, "w") as f:
        n = 0
        for step in plan:
            if step[0] == "slide":
                _, color, src, dst = step
                n += 1
                dr, dc = slide_of(src, dst)
                way = f"{'down' if dr > 0 else 'up'} {abs(dr)}" if dr else \
                    f"{'right' if dc > 0 else 'left'} {abs(dc)}"
                f.write(f"{n:4d}. {color:8s} {min(src)}  {way}\n")
            else:
                _, bucket, gone, bcol = step
                names = ", ".join(f"{c} {min(cells)}" for c, cells in gone)
                f.write(f"      fill {bucket}: {names}   then {bcol}\n")


# ------------------------------- playback -------------------------------

def _mouse(kind, x, y):
    e = sb.Quartz.CGEventCreateMouseEvent(None, kind, (x, y), sb.Quartz.kCGMouseButtonLeft)
    sb.Quartz.CGEventPost(sb.Quartz.kCGHIDEventTap, e)


def drag_straight(p0, p1, pace=1.0):
    """A tray follows the pointer with a lag, and a flick shorter than a cell
    is ignored. One-cell slides need a real hold; longer slides can move faster.
    Release a little past the target center so the tray snaps into that cell."""
    x0, y0 = p0
    x1, y1 = p1
    _mouse(sb.Quartz.kCGEventMouseMoved, x0, y0)
    time.sleep(0.02 * pace)
    _mouse(sb.Quartz.kCGEventLeftMouseDown, x0, y0)
    time.sleep(0.15 * pace)
    dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    motion = max(0.20, dist / 1400.0) * pace
    steps = max(8, int(dist / 7.0))
    delay = motion / steps
    for k in range(1, steps + 1):
        t = k / steps
        _mouse(sb.Quartz.kCGEventLeftMouseDragged, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        time.sleep(delay)
    time.sleep(0.05 * pace)
    _mouse(sb.Quartz.kCGEventLeftMouseUp, x1, y1)


def center(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (sb.GRID_X + (min(cs) + max(cs) + 1) / 2 * sb.CELL_W,
            sb.GRID_Y + (min(rs) + max(rs) + 1) / 2 * sb.CELL_H)


def path_cells(src, dst):
    """Every cell the tray covers while it slides, including both ends."""
    dr, dc = slide_of(src, dst)
    steps = max(abs(dr), abs(dc))
    sdr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sdc = 0 if dc == 0 else (1 if dc > 0 else -1)
    covered = set(src)
    for k in range(1, steps + 1):
        covered |= {(r + sdr * k, c + sdc * k) for r, c in src}
    return covered


def do_slide(src, dst, win, pace=1.0):
    dr, dc = slide_of(src, dst)
    cx, cy = center(src)
    dx, dy = dc * sb.CELL_W, dr * sb.CELL_H
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    # Past the center, still short of the next cell boundary (half a cell).
    ox = dx / length * 0.34 * sb.CELL_W
    oy = dy / length * 0.34 * sb.CELL_H
    drag_straight(sb.to_screen(cx, cy, win),
                  sb.to_screen(cx + dx + ox, cy + dy + oy, win), pace)


def live_bottom(img, name, seq, min_px=2):
    """The stripe actually touching the spout.

    A stripe that is almost spent is only a few pixels tall. The old reader
    wanted 5px, so the last of the purple was skipped and the bucket was
    reported as the color above it. Two pixels of a color this bucket really
    has is enough; the rim highlight (lavender, white) is not in `seq`, so it
    is ignored instead of being announced as a new stripe."""
    allowed = set(seq)
    bkt = sb.BUCKETS[name]
    x = bkt["sample"][0]
    bottom, top = bkt["stripes"]
    run, n = None, 0
    for y in range(bottom, top, -1):
        rgb = img[y:y + 1, x - 12:x + 13].reshape(-1, 3).mean(axis=0)
        c = sb.classify(rgb)
        if c == run:
            n += 1
            continue
        if run in allowed and n >= min_px:
            return run
        run, n = c, 1
    if run in allowed and n >= min_px:
        return run
    return None


def bottom_color(img, name, min_px=2):
    """Lowest sand stripe, used for the startup print."""
    bkt = sb.BUCKETS[name]
    x = bkt["sample"][0]
    bottom, top = bkt["stripes"]
    run, n = None, 0
    found = None
    for y in range(bottom, top, -1):
        rgb = img[y:y + 1, x - 16:x + 17].reshape(-1, 3).mean(axis=0)
        c = sb.classify(rgb)
        if c == run:
            n += 1
        else:
            if run not in sb.NON_PIECE and n >= min_px:
                found = run
                break
            run, n = c, 1
    if found is None and run not in sb.NON_PIECE and n >= min_px:
        found = run
    return found


def bucket_colors(img):
    return {n: bottom_color(img, n) for n in sb.BUCKETS}


TARGETS = {tuple(c) for bkt in sb.BUCKETS.values() for c in bkt["targets"]}


def cell_color(img, r, c):
    return sb.classify(sb.cell_rgb(img, r, c))


def cells_empty(img, cells):
    return all(cell_color(img, r, c) == "empty" for r, c in cells)


def parse_board(img):
    """parse(), plus trays the label reader dropped.

    A tray that is pouring shows "56%" instead of "0%", and sometimes no label
    at all. parse() then turns its cells into walls, so a replan would slide
    another tray into a cell that is still occupied. Any piece-colored cell
    is a tray, label or not."""
    board, colors, _ = sb.parse(img)
    stray = [(r, c) for r, c in list(board.walls) if colors[r][c] not in sb.NON_PIECE]
    if not stray:
        return board
    seen = set()
    for r, c in stray:
        if (r, c) in seen:
            continue
        col = colors[r][c]
        cells, stack = set(), [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in seen or (cr, cc) not in board.walls or colors[cr][cc] != col:
                continue
            seen.add((cr, cc))
            cells.add((cr, cc))
            stack.extend(((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)))
        board.walls -= cells
        board.pieces.append({"color": col, "cells": frozenset(cells)})
    board.rebuild()
    return board


def snapshot():
    img, win = sb.capture(refresh=False)
    return parse_board(img), bucket_colors(img), img, win


def label_mask(img, r, c):
    x0, y0 = sb.cell_origin(r, c)
    patch = img[y0 + 2:y0 + 24, x0 + 2:x0 + 34]
    return np.all(patch > 220, axis=2)


def zero_mask(img, avoid):
    """The '0%' label of some tray that is not pouring, to compare against."""
    avoid = set(avoid)
    for r in range(sb.ROWS):
        for c in range(sb.COLS):
            if (r, c) in avoid or cell_color(img, r, c) in sb.NON_PIECE:
                continue
            if sb.has_label(img, r, c):
                return label_mask(img, r, c)
    return None


def still_zero(img, cells, ref):
    if ref is None:
        return False
    r, c = min(cells)
    mask = label_mask(img, r, c)
    if mask.shape != ref.shape:
        return False
    inter = np.logical_and(mask, ref).sum()
    union = np.logical_or(mask, ref).sum()
    return union >= 8 and inter / union >= 0.72


def color_change(actual, expected, seq):
    """'same', 'next' (the stripe really finished), or 'noise' (a misread)."""
    if actual == expected:
        return "same"
    if not seq or expected not in seq:
        return "noise"
    i = seq.index(expected)
    nxt = seq[i + 1] if i + 1 < len(seq) else None
    if actual == nxt:
        return "next"
    return "noise"


def trays(board):
    return frozenset((p["color"], p["cells"]) for p in board.pieces)


def landed(img, color, src, dst):
    """'ok' the tray is on dst, 'gone' src and dst are both empty (it poured
    and vanished during the drag), 'stay' it is still on src, 'lost' otherwise.
    Pixel colors, not the tray parser: a pouring tray loses its '0%' label and
    the parser drops it, which used to look exactly like a finished fill."""
    src_only = set(src) - set(dst)
    if not cells_empty(img, src_only):
        return "stay"
    dst_colors = [cell_color(img, r, c) for r, c in dst]
    if all(c == color for c in dst_colors):
        return "ok"
    if all(c == "empty" for c in dst_colors):
        return "gone"
    if set(dst) & TARGETS and all(c not in sb.NON_PIECE for c in dst_colors):
        return "ok"
    return "lost"


def grab_image():
    img, win = sb.capture(refresh=False)
    return img, win


def perform_slide(color, src, dst, win):
    """Drag `color` from src to dst. Return (img, win) once it has landed."""
    print(f"  {color:8s} {min(src)} -> {min(dst)}", flush=True)
    for attempt in range(3):
        do_slide(src, dst, win, pace=1.0 + 0.7 * attempt)
        time.sleep(0.05 + 0.06 * attempt)
        img, win = grab_image()
        state = landed(img, color, src, dst)
        if state == "lost":
            # The tray is still sliding; one frame in between cells looks like
            # it left home and never arrived. Look again before giving up.
            time.sleep(0.22)
            img, win = grab_image()
            state = landed(img, color, src, dst)
        if state in ("ok", "gone"):
            return img, win
        if state == "lost":
            print("      that tray left its cell but did not land on the target")
            return None
        print(f"      still in place, slower retry ({attempt + 1})")
    return None


def color_depth(img, name, col, color):
    """How many pixels of `color` are touching the spout in this column.

    A bucket is two cells wide, and the two columns drain separately. The
    middle of the bucket is the wrong place to look."""
    bkt = sb.BUCKETS[name]
    x = sb.GRID_X + col * sb.CELL_W + sb.CELL_W // 2
    bottom, top = bkt["stripes"]
    run, n = None, 0
    for y in range(bottom, top, -1):
        x0, x1 = max(0, x - 8), min(img.shape[1], x + 9)
        c = sb.classify(img[y:y + 1, x0:x1].reshape(-1, 3).mean(axis=0))
        if c == run:
            n += 1
            continue
        if run == color and n >= 2:
            return n
        # A real stripe of a different color is sitting on the spout.
        if run is not None and run not in sb.NON_PIECE and n >= 4:
            return 0
        run, n = c, 1
    if run == color and n >= 2:
        return n
    return 0


def same_bucket_shift(img, cells, color, stripes):
    """Where to slide a half-filled tray so it sits under the sand that is left.

    The tray's own column has run out of `color`, and the other cell of the
    same bucket still has it. One straight slide sideways finishes the pour."""
    covered = {tuple(c) for c in cells}
    options = []
    for name, bkt in sb.BUCKETS.items():
        targets = [tuple(t) for t in bkt["targets"]]
        if not (covered & set(targets)):
            continue
        here = max((color_depth(img, name, c, color) for _, c in covered), default=0)
        if here >= 6:
            return None
        for tr, tc in targets:
            if (tr, tc) in covered:
                continue
            there = color_depth(img, name, tc, color)
            if there < 6 or there <= here:
                continue
            if cell_color(img, tr, tc) != "empty":
                continue
            for sr, sc in covered:
                dr, dc = tr - sr, tc - sc
                if dr and dc:
                    continue
                dst = {(r + dr, c + dc) for r, c in covered}
                if not all(0 <= r < sb.ROWS and 0 <= c < sb.COLS for r, c in dst):
                    continue
                if any(p not in covered and cell_color(img, *p) != "empty" for p in dst):
                    continue
                if (tr, tc) not in dst:
                    continue
                options.append((abs(dr) + abs(dc), dst))
    if not options:
        return None
    return min(options)[1]


def sand_above(img, cells, color, stripes):
    """True when `color` is still the sand directly above this tray."""
    covered = {tuple(c) for c in cells}
    for name, bkt in sb.BUCKETS.items():
        targets = {tuple(c) for c in bkt["targets"]}
        if not (covered & targets):
            continue
        return any(color_depth(img, name, c, color) >= 6 for _, c in covered & targets)
    return True


# Colors of trays that stopped short of 100%. They stay on the board and are
# delivered again only when that color is the bottom stripe of either bucket.
held = set()


def wait_clear(cells, color=None, stripes=None, timeout=8.0):
    """Block until the tray is gone, or the sand directly above it has run out.

    A tray that freezes under 100% is not a slow pour. Its color is no longer
    the stripe touching it. Slide it off and bring it back when that color is
    at the bottom of the left or the right bucket."""
    img, win = grab_image()
    ref = zero_mask(img, cells)
    anchor = min(cells)
    started = time.time()
    end = started + timeout
    saw_pour = False
    last_mask = None
    last_change = started
    while time.time() < end:
        time.sleep(0.15)
        sb.pyautogui.failSafeCheck()
        img, win = grab_image()
        if cells_empty(img, cells):
            time.sleep(0.08)
            img, win = grab_image()
            if cells_empty(img, cells):
                return ("clear", img, win)
        if color and stripes and time.time() - started > 0.6:
            dest = same_bucket_shift(img, cells, color, stripes)
            if dest is not None:
                print("      the other side of this bucket still has that color")
                return ("shift", img, win, dest)
            if not sand_above(img, cells, color, stripes):
                print("      no more of that color right above the tray")
                return ("exhausted", img, win)
        mask = label_mask(img, *anchor)
        pouring = not still_zero(img, cells, ref)
        if pouring:
            saw_pour = True
            same = (last_mask is not None and mask.shape == last_mask.shape
                    and np.array_equal(mask, last_mask))
            if same and time.time() - last_change > 3.5:
                dest = None
                if color and stripes:
                    dest = same_bucket_shift(img, cells, color, stripes)
                if dest is not None:
                    print("      the other side of this bucket still has that color")
                    return ("shift", img, win, dest)
                print("      percentage stopped, that color is not above the tray")
                return ("exhausted", img, win)
            if not same:
                last_mask = mask
                last_change = time.time()
        elif not saw_pour and time.time() - started > 4.0:
            print("      tray is still at 0%, the bucket is not pouring")
            return ("dry", img, win)
    print("      pour did not finish")
    return None


def resolve_pour(cells, color, stripes, win):
    """Wait for a pour. A half-filled tray slides onto the other cell of the
    same bucket when that side still has its color. Returns
    ('gone', img, win), ('moved', img, win, new_cells), or ('drop', None, None, None)."""
    cleared = wait_clear(cells, color=color, stripes=stripes)
    if not cleared or cleared[0] in ("dry", "exhausted"):
        return ("drop", None, None, None)
    if cleared[0] == "shift":
        _, img, win, dest = cleared
        print(f"      sliding the {color} tray onto the other side of the bucket", flush=True)
        moved = perform_slide(color, cells, dest, win)
        if moved is None:
            return ("drop", None, None, None)
        img, win = moved
        if cells_empty(img, set(dest)):
            return ("gone", img, win, None)
        return ("moved", img, win, set(dest))
    _, img, win = cleared[0], cleared[1], cleared[2]
    return ("gone", img, win, None)


def play_plan(plan, board, win, stripes, stalls=0):
    """Play `plan`. A tray that is pouring does not stop the other bucket:
    later slides run as long as they never cross the cells still filling."""
    if stalls > 8:
        print("replanned too many times")
        return 1
    busy = {}  # cell -> (bucket, color) of the tray pouring there
    i = 0
    while i < len(plan):
        sb.pyautogui.failSafeCheck()
        step = plan[i]
        if step[0] == "fill":
            _, bucket, gone, _bcol = step
            print(f"      fill {bucket}: " + ", ".join(c for c, _ in gone), flush=True)
            for color, cells in gone:
                for cell in cells:
                    busy[cell] = (bucket, color)
            i += 1
            continue
        _, color, src, dst = step
        hit = path_cells(src, dst) & set(busy)
        if hit:
            img, win = grab_image()
            hit = {c for c in hit if cell_color(img, *c) != "empty"}
            busy = {c: v for c, v in busy.items() if cell_color(img, *c) != "empty"}
        if hit:
            bucket, pcolor = busy[next(iter(hit))]
            tray = {c for c, v in busy.items() if v == (bucket, pcolor)} or set(hit)
            print(f"      waiting for {sorted(tray)} to finish pouring", flush=True)
            kind, img, win, dest = resolve_pour(tray, pcolor, stripes, win)
            if kind == "moved":
                for c in tray:
                    busy.pop(c, None)
                for c in dest:
                    busy[c] = (bucket, pcolor)
                continue
            if kind == "drop":
                held.add(pcolor)
                print(f"      holding the {pcolor} tray until {pcolor} is at the bottom of a bucket",
                      flush=True)
                return replan(stripes, stalls + 1, force_drop={bucket: pcolor})
            for c in tray:
                busy.pop(c, None)
            if held and _held_at_spout(img, stripes):
                print("      a held color is at the bottom of a bucket", flush=True)
                return replan(stripes, stalls)
            continue
        moved = perform_slide(color, src, dst, win)
        if moved is None:
            return replan(stripes, stalls + 1)
        _, win = moved
        i += 1
    while busy:
        img, win = grab_image()
        left = {c for c in busy if cell_color(img, *c) != "empty"}
        busy = {c: v for c, v in busy.items() if c in left}
        if not busy:
            break
        bucket, pcolor = busy[next(iter(left))]
        tray = {c for c, v in busy.items() if v == (bucket, pcolor)} or set(left)
        kind, img, win, dest = resolve_pour(tray, pcolor, stripes, win)
        if kind == "moved":
            for c in tray:
                busy.pop(c, None)
            for c in dest:
                busy[c] = (bucket, pcolor)
            continue
        if kind != "gone":
            held.add(pcolor)
            print(f"      holding the {pcolor} tray until {pcolor} is at the bottom of a bucket",
                  flush=True)
            return replan(stripes, stalls + 1, force_drop={bucket: pcolor})
        for c in tray:
            busy.pop(c, None)
    board, _, _, win = snapshot()
    if board.pieces:
        print(f"{len(board.pieces)} trays still on the board")
        return replan(stripes, stalls)
    print("level clear")
    return 0


def _held_at_spout(img, stripes):
    """True when a partly filled tray's color is the bottom of either bucket,
    and the plan we are playing is not already pouring that color there."""
    for color in held:
        for n in sb.BUCKETS:
            if bottom_color(img, n, min_px=8) != color:
                continue
            seq = stripes.get(n) or []
            if not seq or seq[0] != color:
                return True
    return False


def _bring_held_back(img, trimmed, skip):
    """A held color that is now the bottom of a bucket becomes that bucket's
    next delivery, whichever bucket it is. The bucket that just ran out is
    skipped so the tray is not put straight back onto an empty spout."""
    for color in list(held):
        for n in sb.BUCKETS:
            if skip.get(n) == color:
                continue
            if bottom_color(img, n, min_px=8) != color:
                continue
            cur = list(trimmed.get(n) or [])
            if not cur or cur[0] != color:
                trimmed[n] = [color] + cur
                print(f"{color} is at the bottom of the {n} bucket, bringing that tray back",
                      flush=True)
            held.discard(color)
            break


def board_sane(board):
    """The out-of-time panel is purple and parses as a board of purple trays."""
    n = len(board.pieces)
    if n < 15:
        return True
    top = max(sum(p["color"] == c for p in board.pieces)
              for c in {p["color"] for p in board.pieces})
    return top <= n * 0.5


def replan(stripes, stalls=0, force_drop=None):
    if stalls > 8:
        print("replanned too many times")
        return 1
    img, win = grab_image()
    board = parse_board(img)
    if not board.pieces:
        print("level clear")
        return 0
    if not board_sane(board):
        print("the board read is not the puzzle, stopping")
        return 1
    force_drop = force_drop or {}
    trimmed, colors = {}, {}
    for n, seq in stripes.items():
        seq = list(seq)
        drop = force_drop.get(n)
        if drop and seq and seq[0] == drop:
            seq = seq[1:]
        bottom = live_bottom(img, n, seq)
        if bottom in seq:
            seq = seq[seq.index(bottom):]
        trimmed[n] = seq
        colors[n] = seq[0] if seq else None
    _bring_held_back(img, trimmed, force_drop)
    colors = {n: (s[0] if s else None) for n, s in trimmed.items()}
    print(f"next tray, from {len(board.pieces)} on the board, buckets {colors}", flush=True)
    show(board)
    # The clock is running. Search only the next tray, then drag it. The
    # board that tray leaves behind is an easier search than this one.
    plan = solve_level(board, trimmed, budget=3.0, limit=1, extend=False)
    if not plan:
        return 1
    return play_plan(plan, board, win, trimmed, stalls)


def play():
    subprocess.run(["osascript", "-e", 'tell application "iPhone Mirroring" to activate'])
    time.sleep(0.4)
    x, y, w, h = sb.window_rect(refresh=True)
    # A pointer left in a screen corner trips the abort check before the first drag.
    _mouse(sb.Quartz.kCGEventMouseMoved, x + w / 2, y + 70)
    img, win = sb.capture(refresh=True)
    board = parse_board(img)
    stripes = {n: sb.read_stripes(img, b) for n, b in sb.BUCKETS.items()}
    colors = bucket_colors(img)
    print(f"{len(board.pieces)} trays, buckets {colors}")
    show(board)
    held.clear()
    if colors.get("left") != "purple" or colors.get("right") != "blue" or len(board.pieces) < 38:
        print("this is not a fresh level, not dragging")
        return 1
    print("solving before the timer starts...")
    plan = solve_level(board, stripes)
    if not plan:
        return 1
    write_plan(plan, os.path.join(HERE, "solver_plan.txt"))
    print(f"plan written, {sum(s[0] == 'slide' for s in plan)} slides. dragging now.",
          flush=True)
    return play_plan(plan, board, win, stripes)


FULL_STRIPES = {
    "left": ["purple", "cyan", "magenta", "orange", "brown", "pink"],
    "right": ["blue", "lavender", "red", "yellow", "brown", "pink"],
}


def resume():
    """Continue from whatever is on screen. A tray sitting on a bucket is
    still being poured, even when the stripe above it has worn down to a line
    the color reader misses."""
    subprocess.run(["osascript", "-e", 'tell application "iPhone Mirroring" to activate'])
    time.sleep(0.2)
    x, y, w, h = sb.window_rect(refresh=True)
    _mouse(sb.Quartz.kCGEventMouseMoved, x + w / 2, y + 70)
    img, win = sb.capture(refresh=True)
    board = parse_board(img)
    stripes = {}
    for n, seq in FULL_STRIPES.items():
        targets = {tuple(c) for c in sb.BUCKETS[n]["targets"]}
        on = [p["color"] for p in board.pieces if set(p["cells"]) & targets and p["color"] in seq]
        # Prefer a tray that is already pouring. Otherwise keep the earliest
        # stripe color that still has trays on the board: a thin stripe is
        # easy to miss and read as the color above it.
        chosen = next((c for c in seq if c in on), None)
        if chosen is None:
            chosen = next((c for c in seq if any(p["color"] == c for p in board.pieces)), None)
        if chosen is None:
            chosen = live_bottom(img, n, seq)
        stripes[n] = seq[seq.index(chosen):] if chosen in seq else list(seq)
    colors = {n: (s[0] if s else None) for n, s in stripes.items()}
    print(f"{len(board.pieces)} trays, pouring {colors}", flush=True)
    show(board)
    plan = solve_level(board, stripes)
    if not plan:
        return 1
    print(f"resumed plan, {sum(s[0] == 'slide' for s in plan)} slides", flush=True)
    return play_plan(plan, board, win, stripes)


def solve_cmd(path):
    board, stripes = load(path)
    print(f"{len(board.pieces)} trays")
    show(board)
    for n, s in sb.sand_split(board, stripes).items():
        print(f"  {n:5s} " + ", ".join(f"{c} {k}" for c, k in s))
    plan = solve_level(board, stripes)
    if not plan:
        return 1
    out = os.path.join(HERE, "solver_plan.txt")
    write_plan(plan, out)
    print(f"wrote {out}")
    return 0


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "solve"
    if cmd == "play":
        sys.exit(play())
    if cmd == "resume":
        sys.exit(resume())
    if cmd == "solve":
        path = next((a for a in args[1:] if not a.startswith("-")), os.path.join(HERE, "window.png"))
        sys.exit(solve_cmd(path))
    print("usage: sand_solver.py solve [image] | play")
    sys.exit(2)


if __name__ == "__main__":
    main()
