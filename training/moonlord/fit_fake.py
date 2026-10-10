"""Fit the offline fight's attack knobs (aim lead, spread, projectile life,
fire rates, speeds, homing, contact size) so fake_bridge.py reproduces
LIVE results across several policies at once -- the calibration gate in
calibrate_fake.py, turned into a search.

The measured parts of the fake (boss chase speed, hand motion, regen, max
HP, damage per hit, distance-dependent boss damage) stay fixed; only the
attack knobs that couldn't be read straight off the logs are fitted.

Loss per policy group = (log fake/live median survival)^2
                        + 4 * (fake - live mean boss-HP-left)^2
summed over the FIT groups; HOLD-OUT groups are reported, not fitted.

  python training/moonlord/fit_fake.py --generations 20 --workers 2
Writes training/moonlord/fake_fit.json; fake_bridge.py loads it when present.
"""
import argparse
import json
import math
import multiprocessing as mp
import statistics as S
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "brain", ROOT / "interface"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = HERE / "fake_fit.json"

KNOBS = [  # (key, low, high, integer?)
    ("bolt_lead", 0.0, 1.0, False), ("bolt_spread", 0.0, 0.3, False), ("bolt_life", 60, 600, True),
    ("bolt_period", 120, 420, True), ("bolt_speed", 6.0, 16.0, False),
    ("sphere_lead", 0.0, 1.0, False), ("sphere_period", 150, 420, True), ("sphere_speed", 5.0, 14.0, False),
    ("eye_turn", 0.01, 0.12, False), ("eye_speed", 6.0, 12.0, False), ("eye_period", 100, 320, True),
    ("eye_life", 120, 900, True),
    ("contact_radius", 30.0, 80.0, False), ("hand_speed", 4.0, 20.0, False),
    ("leech_period", 600, 2000, True), ("leech_speed", 6.0, 14.0, False), ("leech_turn", 0.0, 0.06, False),
    ("leech_grab_ticks", 10, 80, True), ("leech_damage", 50, 200, True),
    # dps_* are NOT fitted: measured straight from live logs (fake_bridge DEFAULT_CONFIG). Fitting them let
    # the fit trade boss damage at range for attack knobs, and teachers then learned to flee 2000 px away.
]

# Live references (2026-10-09). small = small world, Spooky + Berserker's Glove;
# large = large world, 24-tile rows, Spooky + Worm Scarf. (key, since, until)
FIT = [(("idle", "", "balanced"), "20261009_155000", "20261009_172000"),
       (("random", "", "balanced"), "20261009_155000", "20261009_172000"),
       (("fly", "vertical_brain_commit10", "balanced"), "20261009_155000", "20261009_172000"),
       (("fly", "tuned_cma_table_defaults", "balanced"), "20261009_155000", "20261009_172000"),
       (("scripted", "", "balanced"), "20261009_180000", "20261009_190000"),
       # 2026-10-09 19:05 batch: the two new behaviours the first fit got wrong
       (("scripted", "ring650,1050_keep400_dither1", "balanced"), "20261009_190000", None),
       (("fly", "tuned_cma_v3_keepaway", "balanced"), "20261009_190000", "20261009_200000"),
       # 21:19 batch: the 8-direction probe fought further out than anything before
       (("sectors", "", "balanced"), "20261009_200000", None)]
HOLDOUT = [(("scripted", "", "nearest"), "20261009_155000", "20261009_172000"),
           (("fly", "vertical_rule_commit10", "balanced"), "20261009_155000", "20261009_172000"),
           (("scripted", "", "balanced"), "20261009_155000", "20261009_172000")]


def unit_to_cfg(u):
    out = {}
    for (k, lo, hi, is_int), x in zip(KNOBS, u):
        v = lo + min(1.0, max(0.0, float(x))) * (hi - lo)
        out[k] = int(round(v)) if is_int else round(v, 4)
    return out


def cfg_to_unit(cfg):
    from fake_bridge import DEFAULT_CONFIG
    return [min(1.0, max(0.0, (float(cfg.get(k, DEFAULT_CONFIG[k])) - lo) / (hi - lo))) for k, lo, hi, _ in KNOBS]


def live_refs(groups):
    import calibrate_fake as C
    refs = []
    for key, since, until in groups:
        rows = C.live_groups(since, until).get(key, [])
        if rows:
            s, left, _ = C.summary(rows)
            refs.append((key, s, left, len(rows)))
    return refs


_W = {}


def _init(table):
    import calibrate_fake as C
    _W["C"], _W["table"], _W["fac"] = C, table, {}


def _job(job):
    key, fake_cfg, seeds = job
    C = _W["C"]
    if key not in _W["fac"]:
        _W["fac"][key] = C.make(key, _W["table"])
    import harness
    from fake_bridge import FakeMoonLordBridge
    lo = json.loads((HERE / "loadout.json").read_text())
    cfg = harness.config_from_loadout(lo, "fight", key[2])
    cfg["max_seconds"] = {"fight": 240.0}
    rows = [harness.run_episode(FakeMoonLordBridge({"seed": s, **fake_cfg}), _W["fac"][key](seed=s), cfg, s)
            for s in seeds]
    return C.summary(rows)


def evaluate(mapper, cands, refs, seeds):
    jobs = [(r[0], c, seeds) for c in cands for r in refs]
    res = mapper(_job, jobs)
    out = []
    for i in range(len(cands)):
        loss, detail = 0.0, []
        for (key, ls, ll, _), (fs, fl, _) in zip(refs, res[i * len(refs):(i + 1) * len(refs)]):
            loss += math.log(max(fs, 1e-3) / ls) ** 2 + 4 * (fl - ll) ** 2
            detail.append((key, ls, fs, ll, fl))
        out.append((loss, detail))
    return out


def show(detail):
    for key, ls, fs, ll, fl in detail:
        print(f"   {':'.join(k for k in key if k):45s} live {ls:5.1f}s {100 * ll:3.0f}%   fake {fs:5.1f}s {100 * fl:3.0f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--table", default=str(ROOT / "brain" / "data" / "response_table_np_none.npz"))
    args = ap.parse_args(argv)
    refs, held = live_refs(FIT), live_refs(HOLDOUT)
    seeds = list(range(9000, 9000 + args.episodes))
    pool = mp.get_context("spawn").Pool(args.workers, _init, (args.table,))
    from cmaes import CMAES
    from fake_bridge import DEFAULT_CONFIG, FITTED
    x0 = cfg_to_unit({**DEFAULT_CONFIG, **FITTED})  # start from the previous fit
    t0 = time.time()
    base = evaluate(pool.map, [{}], refs, seeds)[0]
    print(f"current fake: loss {base[0]:.3f}  [{time.time() - t0:.0f}s]")
    show(base[1])
    es = CMAES(x0, 0.25, popsize=args.popsize, seed=0)
    best = (base[0], {}, base[1])
    for g in range(args.generations):
        xs = es.ask()
        cands = [unit_to_cfg(x) for x in xs]
        res = evaluate(pool.map, cands, refs, seeds)
        es.tell([r[0] for r in res])
        i = int(np.argmin([r[0] for r in res]))
        if res[i][0] < best[0]:
            best = (res[i][0], cands[i], res[i][1])
        print(f"gen {g:2d} best {res[i][0]:.3f} (so far {best[0]:.3f})  [{time.time() - t0:.0f}s]", flush=True)
    # Re-check the best on fresh seeds (fit and hold-out), and the CMA mean.
    fresh = list(range(9500, 9500 + 2 * args.episodes))
    mean_cfg = unit_to_cfg(np.clip(es.mean, 0, 1))
    chk = evaluate(pool.map, [best[1], mean_cfg], refs + held, fresh)
    pool.close()
    pick = 0 if chk[0][0] <= chk[1][0] else 1
    cfg = [best[1], mean_cfg][pick]
    print(f"\nfresh seeds, fit + hold-out groups ({'best' if pick == 0 else 'mean'}):  loss {chk[pick][0]:.3f}")
    show(chk[pick][1])
    errs = [abs(fs - ls) / ls for _, ls, fs, _, _ in chk[pick][1]]
    OUT.write_text(json.dumps({"config": cfg, "loss_fresh": chk[pick][0],
                               "survival_error_mean": S.mean(errs),
                               "groups": [{"group": ":".join(k for k in key if k), "live_s": ls, "fake_s": fs,
                                           "live_left": ll, "fake_left": fl}
                                          for key, ls, fs, ll, fl in chk[pick][1]],
                               "fitted": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
