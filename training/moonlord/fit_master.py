"""Fit the offline Master fight's attack knobs (fake_bridge MASTER profile) so
the teacher each live fly imitates takes the same damage MIX offline as that
fly took live -- every attack at once (2026-10-12: eye- and deathray-only
calibration left bolts, spheres, the tongue and body contact far too weak
offline, so training traded them away blindly).

Targets: live damage per minute by source (hit_sources.py; "unclear" big hits
counted as deathray), two live batches on "The Violent Heart":
  ray  : scene3_master_ray  (teacher senses3_master_ray_a), 12 Oct 22:07
  eyes : scene3_master_eyes (teacher senses3_master_eyes_b), 12 Oct 15:16
Loss = sum over batches and sources of log((fake+30)/(live+30))^2.

  python training/moonlord/fit_master.py --generations 25
Writes training/moonlord/fake_fit_master.json (fake_bridge loads it for difficulty=master).
"""
import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "brain", ROOT / "interface"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = HERE / "fake_fit_master.json"
SOURCES = ("deathray", "eye", "bolt", "sphere", "tongue", "contact")
LIVE = {
    "ray": ("senses3_master_ray_a", {"deathray": 531, "eye": 234, "bolt": 203, "sphere": 134, "tongue": 71, "contact": 229}),
    "eyes": ("senses3_master_eyes_b", {"deathray": 402, "eye": 123, "bolt": 209, "sphere": 167, "tongue": 194, "contact": 407}),
}
KNOBS = [  # (key, low, high, integer?)
    ("eye_turn", 0.005, 0.05, False), ("eye_period", 100, 400, True),
    ("bolt_period", 120, 500, True), ("bolt_speed", 6.0, 16.0, False), ("bolt_lead", 0.0, 1.0, False),
    ("bolt_spread", 0.0, 0.3, False),
    ("sphere_period", 150, 500, True), ("sphere_speed", 5.0, 16.0, False), ("sphere_lead", 0.0, 1.0, False),
    ("leech_period", 500, 2500, True), ("leech_speed", 5.0, 14.0, False), ("leech_turn", 0.0, 0.06, False),
    ("contact_radius", 30.0, 80.0, False), ("ray_width", 20.0, 80.0, False),
]

_W = {}


def unit_to_cfg(u):
    out = {}
    for (k, lo, hi, is_int), x in zip(KNOBS, u):
        v = lo + min(1.0, max(0.0, float(x))) * (hi - lo)
        out[k] = int(round(v)) if is_int else round(v, 4)
    return out


def _init():
    import fly_params
    import harness
    lo = json.loads((HERE / "loadout.json").read_text())
    cfg = harness.config_from_loadout(lo, "fight", "balanced")
    cfg["max_seconds"] = {"fight": 90.0}
    cfg.pop("loadout_items", None)
    _W.update(cfg=cfg, arena=fly_params.arena_px(lo),
              teachers={g: json.loads((HERE / "params" / f"{t}.json").read_text())["weights"] for g, (t, _) in LIVE.items()})


def _job(job):
    over, group, seed = job
    import harness
    import scene3
    from fake_bridge import FakeMoonLordBridge
    b = FakeMoonLordBridge({"seed": seed, "difficulty": "master", "fit_master": False, **over})
    r = harness.run_episode(b, scene3.V3Policy(_W["teachers"][group], _W["arena"]), _W["cfg"], seed)
    return group, r["seconds"], dict(b.damage_by)


def evaluate(pool, cands, seeds):
    jobs = [(c, g, s) for c in cands for g in LIVE for s in seeds]
    res = pool.map(_job, jobs)
    k = len(LIVE) * len(seeds)
    out = []
    for i in range(len(cands)):
        part = res[i * k:(i + 1) * k]
        loss, detail = 0.0, {}
        for g, (_, live) in LIVE.items():
            rs = [r for r in part if r[0] == g]
            mins = sum(r[1] for r in rs) / 60.0
            fake = {s: sum(r[2].get(s, 0) for r in rs) / mins for s in SOURCES}
            loss += sum(math.log((fake[s] + 30) / (live[s] + 30)) ** 2 for s in SOURCES)
            detail[g] = {s: round(fake[s]) for s in SOURCES}
        out.append((loss, detail))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--popsize", type=int, default=14)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    from cmaes import CMAES
    from fake_bridge import DEFAULT_CONFIG, FITTED, MASTER
    base = {**DEFAULT_CONFIG, **FITTED, **MASTER}
    x0 = [min(1, max(0, (float(base[k]) - lo) / (hi - lo))) for k, lo, hi, _ in KNOBS]
    seeds = list(range(8000, 8000 + a.seeds))
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, _init) as pool:
        cur = evaluate(pool, [{}], seeds)[0]
        print(f"current master fake: loss {cur[0]:.2f} {cur[1]}  [{time.time() - t0:.0f}s]", flush=True)
        for g, (_, live) in LIVE.items():
            print(f"   live {g}: {live}")
        es = CMAES(x0, 0.25, popsize=a.popsize, seed=0)
        best = (cur[0], {}, cur[1])
        for gen in range(a.generations):
            cands = [unit_to_cfg(x) for x in es.ask()]
            res = evaluate(pool, cands, seeds)
            es.tell([r[0] for r in res])
            i = int(np.argmin([r[0] for r in res]))
            if res[i][0] < best[0]:
                best = (res[i][0], cands[i], res[i][1])
            print(f"gen {gen:2d} best {res[i][0]:.2f} (so far {best[0]:.2f})  [{time.time() - t0:.0f}s]", flush=True)
        fresh = list(range(8500, 8500 + 2 * a.seeds))
        mean_cfg = unit_to_cfg(np.clip(es.mean, 0, 1))
        chk = evaluate(pool, [best[1], mean_cfg], fresh)
    pick = 0 if chk[0][0] <= chk[1][0] else 1
    cfg = [best[1], mean_cfg][pick]
    print(f"fresh seeds: best {chk[0][0]:.2f}, mean {chk[1][0]:.2f} -> {'best' if pick == 0 else 'mean'}: {chk[pick][1]}")
    OUT.write_text(json.dumps({"config": cfg, "loss_fresh": chk[pick][0], "fake_per_min": chk[pick][1],
                               "live_per_min": {g: live for g, (_, live) in LIVE.items()},
                               "fitted": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
