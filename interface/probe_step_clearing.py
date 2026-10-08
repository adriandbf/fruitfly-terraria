"""Live probe: how tall a step can the character actually clear?

Replaces a guess with a measurement. obstacle_assist.py's max_jump_tiles=3
is a conservative guess that nobody has measured on this character. This
walks one direction at 10 Hz with the assist's height cap raised so it
ATTEMPTS every step it meets, and reports, per step height, whether the
character got over it or stalled against it.

This deliberately bypasses the fly brain -- it measures the character's
physics, not the fly's judgment -- so it needs no brain, no connectome,
and no human. To use it: in a world, build a row of steps of increasing
height (1, 2, 3, 4, 5 tiles, each with 3+ clear tiles above it, and a few
tiles of flat ground between them) in front of the character, make sure the
Terraria window has focus (input is ignored otherwise), and run:

    python interface/probe_step_clearing.py [--dir right|left] [--seconds 40]

It stops early if it stalls against one step for 5 seconds, since hopping
forever against a wall tells you nothing new.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bridge"))
sys.path.insert(0, str(ROOT / "interface"))

from dumb_bot import Bridge
from obstacle_assist import ObstacleAssist

TICK_S = 0.1
STALL_TICKS = 50      # 5s at 10 Hz against one step -> give up on it
CLEARED_PX = 48       # advanced this far past a step's x -> counts as over it
RELEASE = {"left": False, "right": False, "jump": False, "use": False}


def summarize(encounters):
    print("\n=== step clearing results ===")
    if not encounters:
        print("no steps met -- is there a step in front of the character?")
        return
    for e in encounters:
        verdict = "CLEARED" if e["cleared"] else "STALLED"
        print(f"  height {e['height']}: {verdict}  (faced it for {e['ticks']} ticks "
              f"= {e['ticks'] * TICK_S:.1f}s, at x={e['x']:.0f})")
    cleared = [e["height"] for e in encounters if e["cleared"]]
    stalled = [e["height"] for e in encounters if not e["cleared"]]
    if cleared:
        print(f"\ntallest cleared: {max(cleared)} tiles")
    if stalled:
        print(f"shortest stalled: {min(stalled)} tiles")
    if cleared and stalled and max(cleared) < min(stalled):
        untried = [h for h in range(max(cleared) + 1, min(stalled))]
        if untried:
            print(f"-> heights {untried} were never tried, so the limit is somewhere between "
                  f"{max(cleared)} and {min(stalled) - 1} -- build those steps to pin it down.")
        else:
            print(f"-> max_jump_tiles={max(cleared)} fits this character (cleared {max(cleared)}, "
                  f"stalled at {min(stalled)}).")
    elif cleared and stalled:
        print("-> mixed results (a taller step cleared where a shorter one stalled): "
              "likely headroom or approach-run effects, not a clean height limit.")
    print("(each height tried once, in the order met; build repeats to check consistency)")


def run(direction, seconds, bridge_factory=Bridge):
    # Cap raised well above anything plausible so the assist tries every
    # step; we are measuring where it stops working, not trusting the guess.
    assist = ObstacleAssist({"max_jump_tiles": 8})
    key = "right" if direction == "right" else "left"
    sign = 1 if direction == "right" else -1

    bridge = bridge_factory()
    encounters = []
    current = None  # {"height","x","ticks","start_x"}
    last_x, stalled_ticks = None, 0
    start = time.monotonic()
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

            action = dict(RELEASE, **{key: True})
            action, _ = assist.apply(st, action)
            bridge.request({"cmd": "set_action", **action})

            x = p["x"]
            height = assist.obstacle_height(st["tiles"], sign) if st.get("tiles") else 0
            if height > 0 and p.get("on_ground", True):
                if current is None or current["height"] != height or abs(x - current["start_x"]) > 24:
                    if current is not None:
                        current["cleared"] = abs(x - current["start_x"]) > CLEARED_PX
                        encounters.append(current)
                    current = {"height": height, "x": x, "start_x": x, "ticks": 0, "cleared": False}
                current["ticks"] += 1
            elif current is not None and abs(x - current["start_x"]) > CLEARED_PX:
                current["cleared"] = True
                encounters.append(current)
                current = None

            moved = last_x is not None and abs(x - last_x) > 2
            stalled_ticks = 0 if moved else stalled_ticks + 1
            last_x = x
            if stalled_ticks >= STALL_TICKS:
                print(f"stalled for {STALL_TICKS * TICK_S:.0f}s at x={x:.0f} "
                      f"(height ahead: {height}) -- stopping.")
                break
            time.sleep(max(0.0, TICK_S - (time.monotonic() - loop_start)))
    finally:
        bridge.request({"cmd": "set_action", **RELEASE})
        bridge.close()
    if current is not None:
        encounters.append(current)
    summarize(encounters)
    return encounters


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", choices=["right", "left"], default="right")
    ap.add_argument("--seconds", type=float, default=40.0)
    args = ap.parse_args()
    try:
        run(args.dir, args.seconds)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
