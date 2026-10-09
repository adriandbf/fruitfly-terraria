"""Measure how the character actually moves with the current loadout, so
the scripted dodger and fake_bridge.py can use real numbers instead of
guesses. No boss: resets the episode, then plays a fixed key schedule at
10 Hz and logs vx/vy every tick. About 45-60 s of game time.

Live data so far suggested max run speed ~8.2 px/tick (assumed 6), and
unclear wing flight (holding jump+up often gave vy ~0). This settles it.

Usage (Mac, in the arena world, window in front):
  python training/moonlord/probe_movement.py
Writes logs/moonlord/probe_movement_<stamp>.jsonl and prints a summary."""
import json
import statistics as st
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from harness import RELEASE, connect, wait_alive, Clock, _ok  # noqa: E402

# (label, seconds, keys held). "GROUND" = hold down until standing (max 8 s),
# so every test starts from the arena floor, well below the world ceiling
# (probe v2 hit the ceiling in its first second and spoiled the climb tests).
G = ("GROUND", 8.0, {"down": True})
SCHEDULE = [
    ("settle", 1.0, {}),
    ("run_right", 2.5, {"right": True}),
    ("stop", 1.5, {}),
    G, ("soft_climb_0.4s", 0.4, {"jump": True}), ("coast_after_soft", 2.5, {}),
    G, ("fast_climb_0.5s", 0.5, {"jump": True, "up": True}), ("coast_after_fast", 3.0, {}),
    G, ("fast_climb_0.5s_b", 0.5, {"jump": True, "up": True}), ("down_after_fast", 3.0, {"down": True}),
    G, ("jump_only_1.5s", 1.5, {"jump": True}), ("coast_after_jump", 2.0, {}),
    G, ("down_plus_jump", 2.0, {"jump": True, "down": True}),
    G, ("hop", 0.3, {"jump": True}), ("up_only_air", 1.5, {"up": True}),
    G, ("hop2", 0.3, {"jump": True}), ("air_right_no_jump", 2.0, {"right": True}),
    G, ("climb_for_fall", 0.6, {"jump": True, "up": True}), ("long_fall", 4.0, {}),
    G,
]


def main():
    bridge = connect()
    clock = Clock(False)
    wait_alive(bridge, clock, 60)
    _ok(bridge.request({"cmd": "reset_episode"}), "reset_episode")
    out = ROOT / "logs" / "moonlord" / f"probe_movement_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out, "w") as f:
        for label, secs, keys in SCHEDULE:
            for i in range(int(secs * 10)):
                clock.start_tick()
                held = ({"jump": i % 2 == 0} if keys == "pulse" else keys)
                st_ = bridge.request({"cmd": "get_state"})
                if label == "GROUND" and i > 2 and st_["player"]["on_ground"]:
                    break
                bridge.request({"cmd": "set_action", **RELEASE, **held})
                p = st_["player"]
                rec = {"label": label, "i": i, "keys": held, "vx": p["vx"], "vy": p["vy"], "cy": p["cy"],
                       "on_ground": p["on_ground"], "wing_time": p.get("wing_time"),
                       "game_ticks": (st_.get("episode") or {}).get("ticks"), "t": time.time()}
                f.write(json.dumps(rec) + "\n")
                rows.append(rec)
                clock.end_tick()
    bridge.request({"cmd": "set_action", **RELEASE})
    bridge.close()

    print(f"log: {out.relative_to(ROOT)}")
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    for label, rs in by.items():
        if label == "GROUND":
            continue
        vx = [r["vx"] for r in rs]
        vy = [r["vy"] for r in rs]
        rise = (rs[0]["cy"] - min(r["cy"] for r in rs)) / 16
        print(f"{label:20s} vx max|{max(vx, key=abs):6.2f}| med {st.median(vx):6.2f}   "
              f"vy min {min(vy):6.2f} med {st.median(vy):6.2f}   rise {rise:5.1f} tiles   "
              f"wing_time {rs[0]['wing_time']}->{rs[-1]['wing_time']}")
    gt = [b["game_ticks"] - a["game_ticks"] for a, b in zip(rows, rows[1:]) if a["game_ticks"] is not None]
    print(f"game ticks per 10 Hz decision: median {st.median(gt)} (6 = full speed)")


if __name__ == "__main__":
    main()
