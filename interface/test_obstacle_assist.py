"""Offline tests for obstacle_assist.py: synthetic tile grids for each
case, then replays against REAL logged grids (logs/ is git-ignored, so the
replay half skips cleanly on a fresh checkout).

Run: python interface/test_obstacle_assist.py
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from obstacle_assist import ObstacleAssist

W, H = 31, 17
CX, CY = W // 2, H // 2  # (15, 8), as in the real grids


def make_grid(height=0, direction=1, ceiling_above_top=None, overhang=False):
    """Flat floor at row CY+2 and below. An obstacle `height` tiles tall in
    the column ahead (rows CY+1 upward). ceiling_above_top=k puts a solid
    tile k rows above the obstacle's top. overhang=True instead makes a
    solid tile at row CY-1 ahead with the body rows clear (a walk-under)."""
    solid = [False] * (W * H)
    for row in range(CY + 2, H):
        for col in range(W):
            solid[row * W + col] = True
    col = CX + direction
    for row in range(CY + 2 - height, CY + 2):
        solid[row * W + col] = True
    if ceiling_above_top is not None:
        solid[(CY + 2 - height - ceiling_above_top) * W + col] = True
    if overhang:
        solid[(CY - 1) * W + col] = True
    return {"width": W, "height": H, "solid": solid}


def state(tiles, on_ground=True):
    return {"player": {"on_ground": on_ground}, "tiles": tiles}


RIGHT = {"left": False, "right": True, "jump": False, "use": False}
LEFT = {"left": True, "right": False, "jump": False, "use": False}
IDLE = {"left": False, "right": False, "jump": False, "use": False}

passed = 0


def check(name, cond):
    global passed
    assert cond, f"FAIL: {name}"
    passed += 1
    print(f"  ok  {name}")


print("synthetic grids")
a = ObstacleAssist()
check("flat ground: no jump", a.apply(state(make_grid(0)), RIGHT) == (RIGHT, False))
for h in (1, 2, 3):
    act, assisted = ObstacleAssist().apply(state(make_grid(h)), RIGHT)
    check(f"{h}-tile step ahead: jumps", act["jump"] and assisted)
for h in (4, 5):
    act, assisted = ObstacleAssist().apply(state(make_grid(h)), RIGHT)
    check(f"{h}-tile wall: declines (too tall)", not act["jump"] and not assisted)
# ceiling_above_top=k puts a solid tile k rows above the step top, so k=1 is
# ADJACENT (it merges into the wall, making it taller), and k-1 rows are
# free between step and ceiling.
check("ceiling adjacent to a 2-step merges into a 3-tall wall: still clearable",
      ObstacleAssist().apply(state(make_grid(2, ceiling_above_top=1)), RIGHT)[0]["jump"])
check("ceiling adjacent to a 3-step merges into a 4-tall wall: declines",
      not ObstacleAssist().apply(state(make_grid(3, ceiling_above_top=1)), RIGHT)[0]["jump"])
check("low ceiling with one free row above a 2-step: declines (can't stand on it)",
      not ObstacleAssist().apply(state(make_grid(2, ceiling_above_top=2)), RIGHT)[0]["jump"])
check("ceiling exactly at the headroom limit (3 above) is a problem",
      not ObstacleAssist().apply(state(make_grid(2, ceiling_above_top=3)), RIGHT)[0]["jump"])
check("ceiling just beyond headroom (4 above): jumps",
      ObstacleAssist().apply(state(make_grid(2, ceiling_above_top=4)), RIGHT)[0]["jump"])
check("walk-under overhang (body rows clear): no jump",
      not ObstacleAssist().apply(state(make_grid(0, overhang=True)), RIGHT)[0]["jump"])
check("step on the left, walking right: no jump",
      not ObstacleAssist().apply(state(make_grid(2, direction=-1)), RIGHT)[0]["jump"])
check("step on the left, walking left: jumps",
      ObstacleAssist().apply(state(make_grid(2, direction=-1)), LEFT)[0]["jump"])
check("standing still next to a step: no jump",
      not ObstacleAssist().apply(state(make_grid(2)), IDLE)[0]["jump"])
check("airborne: no new trigger",
      not ObstacleAssist().apply(state(make_grid(2), on_ground=False), RIGHT)[0]["jump"])

brain_jump = dict(RIGHT, jump=True)
act, assisted = ObstacleAssist().apply(state(make_grid(2)), brain_jump)
check("brain already jumping: unchanged and NOT credited to the assist", act["jump"] and not assisted)

original = dict(RIGHT)
ObstacleAssist().apply(state(make_grid(2)), original)
check("input action not mutated", original == RIGHT)

print("hold window")
a = ObstacleAssist()  # hold_ticks=3
flat = state(make_grid(0))
seq = [a.apply(state(make_grid(2)), RIGHT)[0]["jump"]] + [a.apply(flat, RIGHT)[0]["jump"] for _ in range(3)]
check("jump held for exactly 3 ticks after a trigger, then released", seq == [True, True, True, False])
a = ObstacleAssist()
a.apply(state(make_grid(2)), RIGHT)
act, _ = a.apply(flat, dict(RIGHT, jump=True))
check("a brain jump cancels any leftover hold", a._hold == 0)

print("real logged grids")
logs = sorted(glob.glob(str(Path(__file__).parent.parent / "logs" / "*.jsonl")))
if not logs:
    print("  (no logs/ here -- skipping replays)")
else:
    stall = Path(__file__).parent.parent / "logs" / "run_fly_20260929_221614.jsonl"
    if stall.exists():
        rec = json.loads(stall.read_text().splitlines()[40])
        h = ObstacleAssist().obstacle_height(rec["state"]["tiles"], 1)
        act, assisted = ObstacleAssist().apply(rec["state"], RIGHT)
        check(f"the real live stall (5-tall wall, x=33756): height={h}, declines", h == 5 and not act["jump"])

    # Every tick where Phase 2's old 1-tile heuristic really fired in a live
    # run: the new logic must agree wherever its stricter headroom rule is
    # met, and we report every disagreement instead of hiding it.
    fired = agree = 0
    disagreements = []
    for path in logs:
        if "dumb_bot" not in path:
            continue
        for line in open(path):
            rec = json.loads(line)
            act = rec.get("action", {})
            st = rec.get("state", {})
            if not (act.get("jump") and (act.get("left") or act.get("right")) and st.get("tiles")):
                continue
            fired += 1
            probe = dict(act, jump=False)
            got, _ = ObstacleAssist().apply(st, probe)
            if got["jump"]:
                agree += 1
            else:
                tl = st["tiles"]
                d = 1 if act.get("right") else -1
                disagreements.append((Path(path).name, rec["tick"], ObstacleAssist().obstacle_height(tl, d)))
    print(f"  old heuristic fired on {fired} real ticks; new assist also jumps on {agree}")
    for d in disagreements:
        print(f"    differs: {d[0]} tick {d[1]} (obstacle height {d[2]})")
    check("replay ran over real dumb_bot jump ticks", fired > 0)

print(f"\n{passed} checks passed")
