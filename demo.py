"""One recorded demo fight: the fly against the Moon Lord, once, then stop.

Same fight as training (Master arena, loadout, Nurse house, whip assist),
but a single episode with no reset afterwards: when the Moon Lord dies (or
the fly does) the keys are released and the game is left as it is.

  python demo.py                 # the best fly, with the live brain viewer
  python demo.py --no-viz        # game only
  python demo.py --params training/moonlord/params/scene3_master_all.json

Steps it runs:
  1. builds the fly's brain (~10 s) and, with the viewer, serves it at
     http://localhost:8765/  -> open it in a browser next to the game
  2. resets the character into the arena and summons the minions
  3. waits for Enter, so you can start recording first, then counts down
  4. spawns the Moon Lord; the fly fights until the fight ends
  5. prints the result; the viewer stays up (end screen) until you press Enter

Every brain slice is also saved to logs/moonlord/<run>_brain.jsonl.gz, so the
brain can be re-rendered later in sync with the game video (viz/replay.py).
Extra arguments go straight to training/moonlord/harness.py.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training" / "moonlord"))
import harness  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=str(harness.DEFAULT_FLY_PARAMS), help="the fly's readout params")
    ap.add_argument("--no-viz", action="store_true", help="don't serve the brain viewer")
    ap.add_argument("--countdown", type=float, default=3.0, help="seconds between Enter and the boss")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for Enter before the boss")
    ap.add_argument("--port", type=int, default=8765)
    args, extra = ap.parse_known_args()
    argv = ["--policy", "fly", "--params-file", args.params, "--episodes", "1",
            "--nurse", "--whip-strategy", "balanced", "--keep-focus", "--max-seconds", "600",
            "--countdown", str(args.countdown), "--record-brain"]
    if not args.no_wait:
        argv.append("--wait-enter")
    if not args.no_viz:
        argv += ["--brain-viz", "--viz-port", str(args.port), "--hold-viewer"]
    harness.main(argv + extra)


if __name__ == "__main__":
    main()
