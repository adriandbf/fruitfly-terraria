"""Live probe: how tall a step can the character actually clear?

Measures instead of guessing. Walks one direction at 10 Hz with the assist's
height cap raised so it ATTEMPTS every step it meets, then reports, per
step, whether the character really got over it -- judged by its vertical
RISE, not just by moving past it. (A first version judged only by "x moved
on", and reported clearing a 7-tile step, which the game's own jump
physics rule out (apex ~6.7 tiles): it had walked up natural sloped
terrain. Slopes and Terraria's 1-tile auto step-up (Collision.StepUp) pass
without a jump, so a pass with too little rise is reported as PASSED, not
CLEARED, and does not count toward the limit.)

Bypasses the fly brain on purpose: this measures the character's physics,
not the fly's judgment. No brain, no connectome, no human needed.

Setup: in a world, build 1-wide pillars of increasing height (2, 3, 4, 5, 6
tiles; each with 4+ clear tiles above it) on flat ground to ONE side of the
character, spaced AT LEAST 10 TILES apart. The spacing matters: a jump
carries ~8 tiles horizontally at walking speed, so pillars closer than that
get passed in mid-air off the previous jump and never register as their own
encounter (a 3-tall pillar went missing from a first run this way). Keep the
Terraria window focused (input is ignored otherwise), then:

    python interface/probe_step_clearing.py [--dir right|left] [--seconds 60]

Every tick is logged to logs/probe_step_*.jsonl so any result can be
checked against the raw positions. Stops early if it stalls 5s on one step, or at a ledge (7+ tile drop).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bridge"))
sys.path.insert(0, str(ROOT / "interface"))

from dumb_bot import Bridge
from obstacle_assist import ObstacleAssist

TICK_S = 0.1
STALL_TICKS = 50      # 5s at 10 Hz with no horizontal progress -> stop
PASSED_PX = 48        # advanced this far past a step's x -> got past it
TILE_PX = 16.0
RISE_TOLERANCE = 0.25  # tiles; need to rise ~the step's height to be standing on it
PHYSICS_APEX_TILES = 6.7  # from the decompiled jump constants; see obstacle_assist.py
LOG_DIR = ROOT / "logs"
RELEASE = {"left": False, "right": False, "jump": False, "use": False}


def ledge_ahead(tiles, sign):
    """True if the column right ahead has NO solid tile anywhere from the
    floor row down to the bottom of the visible grid -- i.e. a drop of at
    least ~7 tiles. A first live run walked off the end of a platform after
    the last pillar and died of fall damage (a ~40-tile fall), so the probe
    now stops at a ledge instead. Small steps down are fine; the grid can't
    see deeper than ~7 tiles below the feet, so any drop that reaches the
    bottom of the view counts as one worth refusing."""
    w, h = tiles["width"], tiles["height"]
    cx, cy = w // 2, h // 2
    col = cx + sign
    return not any(tiles["solid"][row * w + col] for row in range(cy + 2, h))


def verdict(e):
    """CLEARED: got past AND rose about the step's height (a real jump or
    step-up onto it). PASSED: got past but rose too little (slope/terrain).
    STALLED: never got past."""
    if not e["passed"]:
        return "STALLED"
    return "CLEARED" if e["rise"] >= e["height"] - RISE_TOLERANCE else "PASSED"


def summarize(encounters):
    print("\n=== step clearing results ===")
    if not encounters:
        print("no steps met -- is there a step in front of the character?")
        return
    for e in encounters:
        v = verdict(e)
        print(f"  height {e['height']}: {v:<7}  rose {e['rise']:.1f} tiles, "
              f"faced it {e['ticks']} grounded ticks, at x={e['x']:.0f}")
        if v == "PASSED":
            print(f"      ^ got past but rose only {e['rise']:.1f} for a {e['height']}-tall step: "
                  f"sloped terrain / engine auto-step, NOT a measured jump -- ignored below")
        if v == "CLEARED" and e["height"] > PHYSICS_APEX_TILES:
            print(f"      ^ WARNING: {e['height']} exceeds the ~{PHYSICS_APEX_TILES}-tile jump apex "
                  f"the game's physics allow -- check the log, this should not be possible")
    cleared = [e["height"] for e in encounters if verdict(e) == "CLEARED"]
    stalled = [e["height"] for e in encounters if verdict(e) == "STALLED"]
    if cleared:
        print(f"\ntallest cleared: {max(cleared)} tiles")
    if stalled:
        print(f"shortest stalled: {min(stalled)} tiles")
    if cleared and stalled and max(cleared) < min(stalled):
        untried = list(range(max(cleared) + 1, min(stalled)))
        if untried:
            print(f"-> heights {untried} were never tried, so the limit is somewhere between "
                  f"{max(cleared)} and {min(stalled) - 1} -- build those steps to pin it down.")
        else:
            print(f"-> limit is {max(cleared)} tiles (cleared {max(cleared)}, stalled at {min(stalled)}).")
    elif cleared and stalled:
        print("-> mixed results (a taller step cleared where a shorter one stalled): "
              "likely headroom, approach alignment or a marginal jump, not a clean height limit.")
    print("(each step is one encounter; build repeats to check consistency)")


def run(direction, seconds, bridge_factory=Bridge, log_path=None):
    # Cap raised well above anything plausible so the assist tries every
    # step; we are measuring where it stops working, not trusting a guess.
    assist = ObstacleAssist({"max_jump_tiles": 12})
    key = "right" if direction == "right" else "left"
    sign = 1 if direction == "right" else -1

    bridge = bridge_factory()
    log = None
    if log_path is not False:
        LOG_DIR.mkdir(exist_ok=True)
        log_path = log_path or LOG_DIR / f"probe_step_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        log = open(log_path, "w")
        print(f"Logging every tick to {log_path}")

    encounters = []
    current = None  # open encounter
    last_x, stalled_ticks, tick = None, 0, 0
    start = time.monotonic()

    def close(e):
        e["rise"] = (e["start_y"] - e["min_y"]) / TILE_PX
        encounters.append(e)

    try:
        st = bridge.request({"cmd": "get_state"})
        if not (st.get("debug") or {}).get("has_focus", True):
            print("WARNING: Terraria window does not have focus -- input will be ignored. "
                  "Click into it, then rerun.")
        while time.monotonic() - start < seconds:
            loop_start = time.monotonic()
            st = bridge.request({"cmd": "get_state"})
            p = st.get("player") or {}
            if p.get("dead") or p.get("ghost"):
                print("character died -- stopping.")
                break

            if st.get("tiles") and p.get("on_ground", True) and ledge_ahead(st["tiles"], sign):
                print(f"ledge ahead at x={p['x']:.0f} (a drop of 7+ tiles) -- stopping before "
                      f"walking off it. No more pillars past this point.")
                break

            action = dict(RELEASE, **{key: True})
            action, assisted = assist.apply(st, action)
            bridge.request({"cmd": "set_action", **action})

            x, y = p["x"], p["y"]
            on_ground = p.get("on_ground", True)
            height = assist.obstacle_height(st["tiles"], sign) if st.get("tiles") else 0

            if log:
                log.write(json.dumps({"tick": tick, "x": x, "y": y, "on_ground": on_ground,
                                      "height_ahead": height, "jump": action["jump"],
                                      "assisted": assisted}) + "\n")
                log.flush()

            if current is not None:
                current["min_y"] = min(current["min_y"], y)
            if height > 0 and on_ground:
                if current is None or current["height"] != height or abs(x - current["start_x"]) > 24:
                    if current is not None:
                        current["passed"] = abs(x - current["start_x"]) > PASSED_PX
                        close(current)
                    current = {"height": height, "x": x, "start_x": x, "start_y": y,
                               "min_y": y, "ticks": 0, "passed": False}
                current["ticks"] += 1
            elif current is not None and abs(x - current["start_x"]) > PASSED_PX:
                current["passed"] = True
                close(current)
                current = None

            moved = last_x is not None and abs(x - last_x) > 2
            stalled_ticks = 0 if moved else stalled_ticks + 1
            last_x = x
            tick += 1
            if stalled_ticks >= STALL_TICKS:
                print(f"stalled for {STALL_TICKS * TICK_S:.0f}s at x={x:.0f} "
                      f"(height ahead: {height}) -- stopping.")
                break
            time.sleep(max(0.0, TICK_S - (time.monotonic() - loop_start)))
    finally:
        bridge.request({"cmd": "set_action", **RELEASE})
        bridge.close()
        if log:
            log.close()
    if current is not None:
        close(current)
    summarize(encounters)
    return encounters


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", choices=["right", "left"], default="right")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()
    try:
        run(args.dir, args.seconds)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
