"""Summarise Moon Lord harness results per policy and mode, plus per-run
checks that read the per-tick logs (did the whip actually swing? was the
window focused?). Usage: python training/moonlord/report.py [--fake] [--run STAMP]"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs" / "moonlord"


def log_checks(path):
    ticks = swings = whip_ticks = unfocused = use_held = blocked = 0
    for line in open(path):
        r = json.loads(line)
        if r.get("phase") != "play":
            continue
        d = (r.get("state") or {}).get("debug") or {}
        ticks += 1
        unfocused += d.get("has_focus") is False
        whip_ticks += (d.get("whip_projectiles") or 0) > 0
        if (r.get("action") or {}).get("use"):
            use_held += 1
            blocked += d.get("item_animation", 1) == 0 and d.get("whip_projectiles", 1) == 0
    return {"ticks": ticks, "use_held": use_held, "ticks_with_whip_out": whip_ticks,
            "use_held_but_idle": blocked, "unfocused_ticks": unfocused}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--run", default=None, help="only this run stamp")
    args = ap.parse_args()
    path = LOG_DIR / ("results_fake.jsonl" if args.fake else "results.jsonl")
    rows = [json.loads(l) for l in open(path)] if path.exists() else []
    if args.run:
        rows = [r for r in rows if r.get("run") == args.run]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["policy"], r["mode"])].append(r)
    for (pol, mode), rs in sorted(groups.items()):
        n = len(rs)
        secs = sorted(r["seconds"] for r in rs)
        fr = [r.get("boss_parts_frac_left") for r in rs if r.get("boss_parts_frac_left") is not None]
        led = collections.Counter()
        for r in rs:
            led.update(r.get("ledger", {}))
        left = f"{100 * sum(fr) / len(fr):.0f}%" if fr else "n/a"
        print(f"{pol:9s} {mode:5s} n={n:3d}  win {sum(r['win'] for r in rs)}/{n}  "
              f"median {secs[n // 2]:6.1f}s  dmg/min {sum(r['damage_per_min'] for r in rs) / n:6.0f}  "
              f"boss HP left {left}  ledger {dict(led)}")
    if not args.fake:
        print("\nper-episode log checks (latest 10):")
        for r in rows[-10:]:
            if r.get("log") and (ROOT / r["log"]).exists():
                print(r["run"], r["policy"], r["outcome"], log_checks(ROOT / r["log"]))


if __name__ == "__main__":
    sys.exit(main())
