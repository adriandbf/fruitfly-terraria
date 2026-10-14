"""Play a recorded fight's brain activity (harness.py --record-brain) back in
the brain viewer, in real time, e.g. to re-record the brain at a higher
resolution than the live run or to line it up with a game video.

  python viz/replay.py logs/moonlord/<run>_brain.jsonl.gz
  python viz/replay.py <file> --start 30 --speed 0.5   # from 30 s of fight time, half speed

Then open http://localhost:8765/ . Playback starts when you press Enter
(so you can start your screen recorder first), timed by the fight's own
clock (game seconds), with nurse trips and the end screen included.
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from viz import brain_layout, brain_stream  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--start", type=float, default=0.0, help="fight seconds to start from")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--clock", choices=["wall", "game"], default="wall",
                    help="pace by the wall clock of the live run (default; matches a screen recording of it) "
                         "or by game seconds")
    ap.add_argument("--no-wait", action="store_true", help="start playing immediately")
    args = ap.parse_args(argv)

    frames = []
    with gzip.open(args.recording, "rt") as f:
        for line in f:
            fr = json.loads(line)
            if not fr.get("header"):
                frames.append(fr)
    if not frames:
        sys.exit("no frames in the recording")
    print(f"{len(frames)} frames, {frames[-1].get('t') or 0:.0f} s of fight")

    server = brain_stream.BrainServer(brain_layout.to_json(brain_layout.load()), port=args.port)
    print(f"Brain viewer: open {server.url}")
    if not args.no_wait:
        input("Press Enter to start playback... ")
    frames = [f for f in frames if (f.get("t") or 0) >= args.start or f.get("phase") == "end"]
    key = "wall" if args.clock == "wall" and "wall" in frames[0] else "t"
    t0, start = frames[0].get(key) or 0.0, time.monotonic()
    try:
        for fr in frames:
            due = ((fr.get(key) or t0) - t0) / args.speed
            wait = due - (time.monotonic() - start)
            if wait > 0:
                time.sleep(wait)
            server.publish({k: v for k, v in fr.items() if k != "wall"})
        print("done; Ctrl+C to quit")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
