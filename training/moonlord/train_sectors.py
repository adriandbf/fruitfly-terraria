"""Information-ceiling probe: CMA-ES for SectorPolicy's 22 linear weights on
the fake fight (with per-fight boss randomization). NOT the fly -- it asks
"how well could anything do with direction-resolved looming as its only
input?". Compare: the fly's left/right-summed looming capped every policy
at ~34% boss damage offline, the fly included (2026-10-09).

  python training/moonlord/train_sectors.py --generations 30 --workers 2
Writes training/moonlord/params/sectors_linear.json (harness --policy sectors).
"""
import argparse
import json
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

import fly_params  # noqa: E402

N_W = 22
ATTENTION = HERE / "params" / "cma_v4_multi.json"   # threat horizon, keep-away, surfaces, above_px
_W = {}


def _init(randomize):
    import harness
    lo = json.loads((HERE / "loadout.json").read_text())
    cfg = harness.config_from_loadout(lo, "fight", "balanced")
    cfg["max_seconds"] = {"fight": 240.0}
    _W.update(cfg=cfg, randomize=randomize, kw=fly_params.policy_kwargs(fly_params.load(ATTENTION)))


def _episode(job):
    w, seed = job
    import harness
    from fake_bridge import FakeMoonLordBridge
    from policies import SectorPolicy
    from train_fly_cma import randomized
    pol = SectorPolicy(weights=w, **_W["kw"])
    r = harness.run_episode(FakeMoonLordBridge({"seed": seed, **randomized({}, seed, _W["randomize"])}),
                            pol, _W["cfg"], seed)
    return 1.0 - (r["boss_parts_frac_left"] if r["boss_parts_frac_left"] is not None else 1.0), r["seconds"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--train-seeds", type=int, default=12)
    ap.add_argument("--heldout-seeds", type=int, default=30)
    ap.add_argument("--randomize", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    from cmaes import CMAES
    train = list(range(1000, 1000 + args.train_seeds))
    held = list(range(5000, 5000 + args.heldout_seeds))
    es = CMAES(np.full(N_W, 0.5), 0.3, popsize=args.popsize, seed=1)   # unit [0,1] -> weights [-2, 2]
    best = (-1.0, None)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, _init, (args.randomize,)) as pool:
        for g in range(args.generations):
            ws = [list(map(float, u * 4 - 2)) for u in es.ask()]
            res = pool.map(_episode, [(w, s) for w in ws for s in train])
            sc = [float(np.mean([r[0] for r in res[i * len(train):(i + 1) * len(train)]])) for i in range(len(ws))]
            es.tell([-x for x in sc])
            i = int(np.argmax(sc))
            if sc[i] > best[0]:
                best = (sc[i], ws[i])
            print(f"gen {g:2d} best {100 * sc[i]:5.1f}%  mean {100 * np.mean(sc):5.1f}%  [{time.time() - t0:.0f}s]",
                  flush=True)
        h = pool.map(_episode, [(best[1], s) for s in held])
    prog, secs = float(np.mean([r[0] for r in h])), float(np.median([r[1] for r in h]))
    print(f"held-out: boss damage {100 * prog:.1f}%  median survival {secs:.1f}s")
    out = HERE / "params" / "sectors_linear.json"
    att = fly_params.load(ATTENTION)
    out.write_text(json.dumps({"encoder": att["encoder"], "decoder": att["decoder"], "policy": att["policy"],
                               "weights": best[1],
                               "meta": {"what": "SectorPolicy info-ceiling probe, NOT the fly",
                                        "train_progress": best[0], "heldout_progress": prog,
                                        "heldout_median_s": secs, "randomize": args.randomize,
                                        "fake_fit": json.loads((HERE / "fake_fit.json").read_text()).get("fitted"),
                                        "finished": time.strftime("%Y-%m-%d %H:%M:%S")}}, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
