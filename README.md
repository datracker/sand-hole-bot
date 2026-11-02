# Sand Hole Bot

A bot that plays the **Sand Hole** puzzle on an iPhone through **macOS iPhone Mirroring** — it reads
the board from the screen, solves the whole level offline, then performs the drags for real.

## How it works

The puzzle is deterministic: a drag slides one tray along a free path (turning corners if need be).
Each half of a bucket pours into the tray right under it while the bottom of that half is the tray's
color, and a full tray vanishes. That means the bot can read the board and the buckets' stripes
**once** and solve the entire level **before the clock starts** — the in-game timer only starts with
the first drag.

Key ideas:

- **Sand is counted in trays, never in pixels** — the sand pictures aren't to scale, so how much
  sand there is comes from the trays.
- Both halves of a bucket are kept draining evenly, and a delivered tray that stops filling slides
  under the bucket's other half if that half shows its color.
- Drags are **closed-loop**: like a finger pulling a tray, the bot watches the screen as it drags
  and turns a corner or lets go only once the tray is actually there (iPhone Mirroring adds lag, so
  a faster finger just leaves the tray behind).
- After each drag it checks: a tray that stopped short gets dragged on from where it is, and if the
  real board no longer matches the plan, the bot re-reads and re-solves from there.

## The solver

- `sand_hole_bot.py` — the bot: screen reading (color palette classification), board model,
  solver, and drag execution.
- `sand_solver.py` — an alternative solver built on the same model. Search is **iterated width**
  (Lipovetzky & Geffner): a breadth-first search that only keeps positions in which some tray
  stands somewhere it has not stood before, repeated until a tray of the bucket's color is on that
  bucket. The tie-break is how many trays still cage the other bucket's color, so the short plan is
  one that does not bury the next delivery. The full plan is built before the first drag; once the
  clock is running, a replan searches only the next tray.

## Requirements

- macOS with **iPhone Mirroring** running the game
- Python 3 and:

  ```sh
  pip install pyautogui numpy pillow pyobjc-framework-Quartz
  ```

- **System Settings → Privacy & Security**: give your terminal **Accessibility** *and*
  **Screen Recording** permission (the bot captures the window with `screencapture` and drives the
  pointer with pyautogui).

## Calibration

All measurements are pixels in `window.png` at the top of `sand_hole_bot.py` (`REF_W`, `RIM_Y`,
`GRID_X/GRID_Y`, `CELL_W/CELL_H`, the `BUCKETS` sample points, and the color `PALETTE`). The board
is an 8 × 11 cell grid with a bucket over each side. To re-calibrate for a different window:

```sh
python sand_hole_bot.py snap   # saves window.png + grid_check.png — measure, tune CALIBRATION
python sand_hole_bot.py read   # prints what the bot sees (add -v for raw RGB values)
```

## Usage

```sh
python sand_hole_bot.py snap              # save window.png + grid_check.png for calibration
python sand_hole_bot.py read [-v]         # print what the bot sees (default command)
python sand_hole_bot.py solve [img.png]   # solve the level on screen (or in a screenshot) and
                                          # save it to solution.pkl + moves.txt (-v lists it)
python sand_hole_bot.py follow [moves]    # play moves.txt (or the file named) exactly as written
python sand_hole_bot.py play [--blind]    # play the saved solution if it fits the board, else
                                          # solve first; re-solves if the game differs from plan.
                                          # --blind only looks at the screen after deliveries
```

`sand_solver.py` has its own entry point:

```sh
python sand_solver.py solve [window.png]  # solve and write solver_plan.txt
python sand_solver.py play                # drive iPhone Mirroring with the plan
```

## Safety

`pyautogui.FAILSAFE` is on: **shove the pointer into a screen corner to abort** at any time.

## Notes

Debug screenshots (`window.png`, `grid_check.png`, …) and runtime artifacts (`solution.pkl`,
`moves.txt`, `solver_plan.txt`) are gitignored local artifacts, regenerated on each run.
