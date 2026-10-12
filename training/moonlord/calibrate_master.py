"""Offline Master fight vs live Master (2026-10-11): run a policy in fake_bridge
with difficulty=master, the Nurse protocol and the v5 house (right under the
floor), and print survival / boss left / hits per min / damage share by source.
  python training/moonlord/calibrate_master.py --policy idle --fights 20
  python training/moonlord/calibrate_master.py --policy senses3 --params training/moonlord/params/senses3_teacher.json
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
import scene3  # noqa: E402
from fake_bridge import FakeMoonLordBridge  # noqa: E402

LIVE = {"idle": (20.6, 0.74, 28.8), "fly": (59.2, 0.46, 14.9)}
MASTER_FAKE = {"difficulty": "master", "house_depth_tiles": 6}


def master_cfg(max_s=240.0):
    lo = json.loads((HERE / "loadout.json").read_text())
    cfg = harness.config_from_loadout(lo, "fight", "balanced")
    cfg["max_seconds"] = {"fight": max_s}
    cfg.pop("loadout_items", None)
    cfg["nurse"] = {**lo["nurse"], "enabled": True}
    return cfg


def run(policy_factory, seeds, fake_extra=None, cfg=None):
    cfg = cfg or master_cfg()
    rows = []
    for s in seeds:
        b = FakeMoonLordBridge({"seed": s, **MASTER_FAKE, **(fake_extra or {})})
        b.request({"cmd": "build_nurse_house", "depth_tiles": 6})
        r = harness.run_episode(b, policy_factory(), cfg, s)
        r["damage_by"] = b.request({"cmd": "get_episode"}).get("episode", {}).get("damage_by", {})
        rows.append(r)
    return rows


def summary(rows):
    secs = S.median(r["seconds"] for r in rows)
    left = S.mean(r["boss_parts_frac_left"] for r in rows)
    hpm = S.mean(60 * r["hits"] / max(r["seconds"], 1) for r in rows)
    tot = {}
    for r in rows:
        for k, v in r["damage_by"].items():
            tot[k] = tot.get(k, 0) + v
    t = sum(tot.values()) or 1
    share = {k: round(100 * v / t) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])}
    return secs, left, hpm, sum(r["win"] for r in rows), share


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="idle", choices=["idle", "senses3"])
    ap.add_argument("--params", default=str(HERE / "params" / "senses3_teacher.json"))
    ap.add_argument("--fights", type=int, default=20)
    a = ap.parse_args(argv)
    if a.policy == "idle":
        fac = policies.IdlePolicy
    else:
        w = json.loads(Path(a.params).read_text())["weights"]
        import fly_params
        fac = lambda: scene3.V3Policy(w, fly_params.arena_px())  # noqa: E731
    secs, left, hpm, wins, share = summary(run(fac, range(7000, 7000 + a.fights)))
    live = LIVE.get(a.policy)
    print(f"{a.policy}: fake master {secs:.1f} s median, boss {100 * left:.0f}% left, {hpm:.1f} hits/min, "
          f"{wins}/{a.fights} wins; damage share {share}")
    if live:
        print(f"          live master {live[0]:.1f} s, boss {100 * live[1]:.0f}% left, {live[2]:.1f} hits/min")


if __name__ == "__main__":
    main()
