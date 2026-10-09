"""Calibration gate for the offline fight: does fake_bridge.py reproduce the
LIVE results of every policy we have live data for?

Tuning on a fake that's wrong teaches the fly the fake's quirks (2026-10-09:
CMA-ES parked the fly beside the fake's motionless hand; live, the hand
swings there and hit it 26 times). So before tuning, the fake has to match
live survival and boss damage for several different policies, and rank
them in the same order.

Live references come from logs/moonlord/results.jsonl, grouped like
report.py (policy + variant + whip strategy) within a time window. Fake
results come from running the same policies through harness.run_episode on
fake_bridge with the same loadout config.

  python training/moonlord/calibrate_fake.py --since 20261009_155000 --until 20261009_172000
  python training/moonlord/calibrate_fake.py --since 20261009_180000 --policies scripted
  ... --fake-config '{"dps_near": 3500}'   # try a change
"""
import argparse
import json
import statistics as S
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "brain", ROOT / "interface"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import harness  # noqa: E402
import policies  # noqa: E402
from fake_bridge import FakeMoonLordBridge  # noqa: E402

RESULTS = ROOT / "logs" / "moonlord" / "results.jsonl"


def live_groups(since, until, results=RESULTS):
    groups = {}
    for line in results.read_text().splitlines():
        r = json.loads(line)
        run = r.get("run", "")
        if r.get("simulated") or run < since or (until and run >= until) or r.get("mode") != "fight":
            continue
        key = (r["policy"], r.get("variant") or "", r.get("whip_strategy") or "nearest")
        groups.setdefault(key, []).append(r)
    return groups


def make(key, table=None):
    pol, variant, _ = key
    if pol != "fly":
        lo = json.loads((HERE / "loadout.json").read_text())
        kw = {}
        if pol == "scripted":
            a = lo.get("arena", {})
            rise = min(a.get("rows", 4) * a.get("row_spacing_tiles", 12), a.get("max_rise_tiles") or 10 ** 6)
            kw = {"arena_half_width": 0.9 * 16 * a.get("half_width_tiles", 90), "arena_height": 0.9 * 16 * rise}
            if variant.startswith("ring"):  # e.g. ring550,950_keep350_dither1
                ring, keep, dither = variant.split("_")
                kw.update(ring=tuple(float(v) for v in ring[4:].split(",")), boss_keep_away=float(keep[4:]),
                          dither_cost=float(dither[6:]))
        return lambda seed=0: policies.make_policy(pol, seed=seed, **kw)
    from brain_backend import make_brain
    import fly_params
    brain, readout = make_brain("table", "none", table_path=table)
    if variant.startswith("tuned_"):
        params = fly_params.load(HERE / "params" / f"{variant[len('tuned_'):]}.json")
        kw = fly_params.policy_kwargs(params)
    else:
        kw = {"vertical": "rule" if "rule" in variant else "brain",
              "commit_ticks": int(variant.split("commit")[1].split("_")[0]) if "commit" in variant else 0}
    return lambda seed=0: policies.FlyPolicy(brain=brain, readout_fn=readout, **kw)


def fake_rows(key, n, fake_cfg, table):
    lo = json.loads((HERE / "loadout.json").read_text())
    cfg = harness.config_from_loadout(lo, "fight", key[2])
    cfg["max_seconds"] = {"fight": 300.0}
    factory = make(key, table)
    rows = []
    for i in range(n):
        seed = 7000 + i
        pol = factory(seed=seed)
        b = FakeMoonLordBridge({"seed": seed, **fake_cfg})
        rows.append(harness.run_episode(b, pol, cfg, seed))
    return rows


def summary(rows):
    secs = [r["seconds"] for r in rows]
    left = [r["boss_parts_frac_left"] for r in rows if r.get("boss_parts_frac_left") is not None]
    return S.median(secs), (S.mean(left) if left else float("nan")), S.mean(r["hits"] for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", default=None)
    ap.add_argument("--policies", default="idle,random,scripted,fly")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--fake-config", default=None)
    ap.add_argument("--table", default=str(ROOT / "brain" / "data" / "response_table_np_none.npz"))
    args = ap.parse_args(argv)
    fake_cfg = json.loads(args.fake_config) if args.fake_config else {}
    want = set(args.policies.split(","))
    out = []
    print(f"{'group':48s} {'live s':>7s} {'fake s':>7s} {'live left':>9s} {'fake left':>9s} {'hits L/F':>9s}")
    for key, rows in sorted(live_groups(args.since, args.until).items()):
        if key[0] not in want or key[0] == "human":
            continue
        ls, ll, lh = summary(rows)
        fs, fl, fh = summary(fake_rows(key, args.episodes, fake_cfg, args.table))
        out.append((key, ls, fs, ll, fl))
        print(f"{':'.join(k for k in key if k):48s} {ls:7.1f} {fs:7.1f} {100 * ll:8.0f}% {100 * fl:8.0f}% {lh:4.1f}/{fh:4.1f}"
              f"   (n live {len(rows)})")
    if len(out) >= 2:
        err = S.mean(abs(f - l) / l for _, l, f, _, _ in out)
        rank_ok = sorted(out, key=lambda o: o[1]) == sorted(out, key=lambda o: o[2])
        dmg_err = S.mean(abs(fl - ll) for *_, ll, fl in out)
        print(f"\nmean survival error {100 * err:.0f}%   mean boss-left error {100 * dmg_err:.0f} points   "
              f"same survival ranking: {rank_ok}")
    return out


if __name__ == "__main__":
    main()
